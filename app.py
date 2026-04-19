from flask import Flask, render_template, jsonify, Response, request
import requests, threading, time, datetime, json, os, tempfile
import websocket
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter, label, center_of_mass

app = Flask(__name__)

NASA_KEY      = "D4fL2nbc434oawadYKVWc0dKB12hCBidGUYJXZpN"
AQICN_KEY     = "f939b0945911f520627cd2820cd7827603d15344"
OWM_KEY       = "a2ad2c01ea42bc578b23d80775bfc974"
EPA_KEY       = "ce469fc9d073425b9bbd9c44bda741f9"
EPA_SITE_ID   = "9348c1f5-60c5-4c35-b4f1-1f0931ab1415"

OPENSKY_CLIENT_ID     = "djmg1988-api-client"
OPENSKY_CLIENT_SECRET = "ca0OzWSHIFafrrRkfrGfN2MWXr6JvMme"
OPENSKY_TOKEN_URL     = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"

OPENSKY_BOUNDS = {
    "lamin": -38.5, "lamax": -37.0,
    "lomin": 144.5, "lomax": 146.5
}

OBS_LAT  = -37.7939
OBS_LON  = 145.3214
OBS_ELEV = 119

# ---------------------------------------------------------------------------
# Ares — AIS constants
# ---------------------------------------------------------------------------

AISSTREAM_KEY = "f2544a3031b5f0b310381437e8ebc5f7c8589ff7"
ARES_BOUNDS   = [[[23.0, 55.0], [27.5, 61.0]]]  # Strait of Hormuz bounding box

# ---------------------------------------------------------------------------
# Ares — SAR / Sentinel-1 constants
# ---------------------------------------------------------------------------

COPERNICUS_CLIENT_ID     = "sh-5a8c9fdf-b738-4b55-8103-2feb34b909ca"
COPERNICUS_CLIENT_SECRET = "iY4Re2PYth5kt5m6VGPqdMMnUlnQDyWG"
COPERNICUS_INSTANCE_ID   = "sh-5f8b630b-b083-49ed-b340-b8f01ecb81c4"

SAR_BBOX          = [55.0, 23.0, 61.0, 27.5]  # [lon_min, lat_min, lon_max, lat_max]
SAR_WIDTH         = 1024
SAR_HEIGHT        = 512
SAR_REFRESH_HOURS = 6


_SHIP_TYPE_RANGES = [
    (range(20, 30), "Wing in Ground"),
    (range(30, 31), "Fishing"),
    (range(31, 33), "Towing"),
    (range(35, 36), "Military"),
    (range(36, 37), "Sailing"),
    (range(37, 38), "Pleasure Craft"),
    (range(40, 50), "High Speed Craft"),
    (range(50, 51), "Pilot Vessel"),
    (range(51, 52), "SAR"),
    (range(52, 53), "Tug"),
    (range(60, 70), "Passenger"),
    (range(70, 80), "Cargo"),
    (range(80, 90), "Tanker"),
    (range(90, 100), "Other"),
]

_MID_COUNTRY = {
    "310": "Bermuda", "311": "Bahamas", "316": "Canada",
    "319": "Cayman Islands", "338": "USA", "339": "USA",
    "366": "USA", "367": "USA", "368": "USA", "369": "USA",
    "303": "USA", "351": "Panama", "352": "Panama", "353": "Panama",
    "354": "Panama", "355": "Panama", "356": "Panama", "357": "Panama",
    "370": "Panama", "371": "Panama", "372": "Panama", "373": "Panama",
    "232": "UK", "233": "UK", "234": "UK", "235": "UK",
    "209": "Cyprus", "210": "Cyprus", "212": "Cyprus",
    "229": "Malta", "248": "Malta", "249": "Malta",
    "403": "Saudi Arabia", "408": "Iraq",
    "413": "China", "414": "China",
    "416": "Taiwan",
    "422": "Iran", "447": "Iran",
    "440": "South Korea", "441": "South Korea",
    "451": "Qatar", "455": "Bahrain",
    "456": "Kuwait", "457": "Kuwait",
    "461": "UAE", "462": "UAE", "463": "UAE",
    "470": "UAE", "471": "UAE", "472": "UAE",
    "477": "Hong Kong",
    "503": "Australia",
    "525": "Indonesia",
    "538": "Marshall Islands",
    "564": "Singapore", "565": "Singapore",
    "620": "Comoros",
    "636": "Liberia", "637": "Liberia",
}


def ship_type_label(type_code):
    """Return human-readable ship type label for an AIS type code."""
    for r, label in _SHIP_TYPE_RANGES:
        if type_code in r:
            return label
    return "Unknown"


def is_military(type_code):
    """Return True if the AIS type code indicates a military vessel."""
    return ship_type_label(type_code) == "Military"


def mmsi_to_country(mmsi_str):
    """Return country name for an MMSI string based on its 3-digit MID prefix."""
    mid = str(mmsi_str)[:3]
    return _MID_COUNTRY.get(mid, "Unknown")


def prune_stale_ships(ships_dict, max_age=1800):
    """Remove ships not seen in max_age seconds. Mutates ships_dict in place."""
    cutoff = time.time() - max_age
    stale = [mmsi for mmsi, s in ships_dict.items() if s.get("last_seen", 0) < cutoff]
    for mmsi in stale:
        del ships_dict[mmsi]


def detect_ships_cfar(arr, guard_size=5, bg_size=20, threshold=6.0):
    if arr.size == 0:
        return []
    if arr.mean() < 0:
        arr = 10.0 ** (arr.astype(np.float64) / 10.0)
    arr = arr.astype(np.float64)
    w_total  = guard_size + bg_size * 2 + 1
    w_guard  = guard_size
    sum_total = uniform_filter(arr, w_total) * (w_total ** 2)
    sum_guard = uniform_filter(arr, w_guard) * (w_guard ** 2)
    ring_area = max(w_total ** 2 - w_guard ** 2, 1)
    bg_mean   = (sum_total - sum_guard) / ring_area
    bg_mean   = np.maximum(bg_mean, 1e-10)
    detections = (arr / bg_mean) > threshold
    labeled, num = label(detections)
    if num == 0:
        return []
    centroids = []
    for i in range(1, num + 1):
        size = int(np.sum(labeled == i))
        if 2 <= size <= 500:
            cy, cx = center_of_mass(detections, labeled, i)
            centroids.append((float(cy), float(cx)))
    return centroids


# ---------------------------------------------------------------------------
# Ares — SAR / Sentinel-1 backend
# ---------------------------------------------------------------------------

def get_sar_token():
    now = time.time()
    if sar_token["value"] and sar_token["expires"] > now + 60:
        return sar_token["value"]
    try:
        resp = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data={
                "grant_type":    "client_credentials",
                "client_id":     COPERNICUS_CLIENT_ID,
                "client_secret": COPERNICUS_CLIENT_SECRET,
            },
            timeout=10
        )
        resp.raise_for_status()
        d = resp.json()
        sar_token["value"]   = d["access_token"]
        sar_token["expires"] = now + d.get("expires_in", 600)
        return sar_token["value"]
    except Exception as e:
        print(f"[SAR] Token error: {e}")
        return None


def _download_sar_geotiff():
    token = get_sar_token()
    if not token:
        raise RuntimeError("SAR token unavailable")
    now    = datetime.datetime.utcnow()
    t_to   = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    t_from = (now - datetime.timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "input": {
            "bounds": {
                "bbox":       SAR_BBOX,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type":       "sentinel-1-grd",
                "dataFilter": {
                    "timeRange":       {"from": t_from, "to": t_to},
                    "acquisitionMode": "IW",
                    "polarization":    "DV",
                    "resolution":      "HIGH"
                },
                "processing": {
                    "backCoeff":    "SIGMA0_ELLIPSOID",
                    "orthorectify": True,
                    "demInstance":  "COPERNICUS"
                }
            }]
        },
        "output": {
            "width":     SAR_WIDTH,
            "height":    SAR_HEIGHT,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
        },
        "evalscript": (
            "//VERSION=3\n"
            "function setup() {\n"
            "  return { input: ['VV'], output: { bands: 1, sampleType: 'FLOAT32' } };\n"
            "}\n"
            "function evaluatePixel(sample) { return [sample.VV]; }"
        )
    }
    resp = requests.post(
        "https://sh.dataspace.copernicus.eu/api/v1/process",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60
    )
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(suffix=".tiff", delete=False)
    tmp.write(resp.content)
    tmp.close()
    return tmp.name, t_to


def _detect_and_store():
    tmp_path = None
    try:
        print("[SAR] Starting scene download...")
        tmp_path, scene_ts = _download_sar_geotiff()
        with rasterio.open(tmp_path) as ds:
            arr       = ds.read(1)
            transform = ds.transform
        centroids = detect_ships_cfar(arr)
        results = []
        for cy, cx in centroids:
            lon, lat = rasterio.transform.xy(transform, cy, cx)
            lon, lat = float(lon), float(lat)
            if SAR_BBOX[1] <= lat <= SAR_BBOX[3] and SAR_BBOX[0] <= lon <= SAR_BBOX[2]:
                results.append({"lat": round(lat, 5), "lon": round(lon, 5), "source": "SAR", "scene": scene_ts})
        with sar_lock:
            sar_ships.clear()
            sar_ships.extend(results)
            sar_status["last_run"]   = time.time()
            sar_status["scene_date"] = scene_ts
            sar_status["count"]      = len(results)
            sar_status["error"]      = None
        print(f"[SAR] Detection complete: {len(results)} ships")
    except Exception as e:
        with sar_lock:
            sar_status["last_run"] = time.time()
            sar_status["error"]    = str(e)
        print(f"[SAR] Detection error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _run_sar_refresh():
    while True:
        _detect_and_store()
        time.sleep(SAR_REFRESH_HOURS * 3600)

cache = {}
ares_ships = {}        # keyed by MMSI string, stores latest state per vessel
ares_lock  = threading.Lock()
sar_token  = {"value": None, "expires": 0}
sar_ships  = []
sar_lock   = threading.Lock()
sar_status = {"last_run": None, "scene_date": None, "count": 0, "error": None}
opensky_token = {"value": None, "expires": 0}

# ---------------------------------------------------------------------------
# OpenSky OAuth2
# ---------------------------------------------------------------------------

def get_opensky_token():
    now = time.time()
    if opensky_token["value"] and opensky_token["expires"] > now + 60:
        return opensky_token["value"]
    try:
        r = requests.post(OPENSKY_TOKEN_URL, data={
            "grant_type":    "client_credentials",
            "client_id":     OPENSKY_CLIENT_ID,
            "client_secret": OPENSKY_CLIENT_SECRET,
        }, timeout=10)
        r.raise_for_status()
        data = r.json()
        opensky_token["value"]   = data["access_token"]
        opensky_token["expires"] = now + data.get("expires_in", 1800)
        return opensky_token["value"]
    except Exception as e:
        print(f"OpenSky token error: {e}")
        return None

def refresh_flights():
    while True:
        try:
            token = get_opensky_token()
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            r = requests.get(
                "https://opensky-network.org/api/states/all",
                params=OPENSKY_BOUNDS, headers=headers, timeout=10
            )
            r.raise_for_status()
            cache["flights"] = {"data": r.json(), "ts": time.time(), "error": None}
        except Exception as e:
            old = cache.get("flights", {}).get("data")
            cache["flights"] = {"data": old, "ts": time.time(), "error": str(e)}
        time.sleep(30)

# ---------------------------------------------------------------------------
# Generic refresh
# ---------------------------------------------------------------------------

def refresh(key, url, interval, params=None, headers=None, startup_delay=0):
    if startup_delay:
        time.sleep(startup_delay)
    while True:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            cache[key] = {"data": r.json(), "ts": time.time(), "error": None}
        except Exception as e:
            old = cache.get(key, {}).get("data")
            cache[key] = {"data": old, "ts": time.time(), "error": str(e)}
        time.sleep(interval)

def start(key, url, interval, params=None, headers=None, startup_delay=0):
    t = threading.Thread(target=refresh, args=(key, url, interval),
                         kwargs={"params": params, "headers": headers, "startup_delay": startup_delay}, daemon=True)
    t.start()

# ---------------------------------------------------------------------------
# EPA refresh — no query params needed, returns latest
# ---------------------------------------------------------------------------

def refresh_epa():
    time.sleep(15)
    while True:
        try:
            r = requests.get(
                f"https://gateway.api.epa.vic.gov.au/environmentMonitoring/v1/sites/{EPA_SITE_ID}/parameters",
                headers={"X-API-Key": EPA_KEY, "User-Agent": "curl/7.88.1", "Accept": "application/json"},
                timeout=30
            )
            r.raise_for_status()
            cache["epa_air"] = {"data": r.json(), "ts": time.time(), "error": None}
        except Exception as e:
            old = cache.get("epa_air", {}).get("data")
            cache["epa_air"] = {"data": old, "ts": time.time(), "error": str(e)}
        time.sleep(3600)

# ---------------------------------------------------------------------------
# Starlink TLE refresh — proxy CelesTrak through Flask
# ---------------------------------------------------------------------------

def refresh_starlink_tle():
    time.sleep(20)
    while True:
        try:
            # Fetch 500 Starlink TLEs from Ivan's API (paginated, 100 per page)
            all_tles = []
            for page in range(1, 6):
                r = requests.get(
                    "https://tle.ivanstanojevic.me/api/tle/",
                    params={"search": "starlink", "page-size": 100, "page": page},
                    headers={"User-Agent": "curl/7.88.1"},
                    timeout=30
                )
                r.raise_for_status()
                data = r.json()
                members = data.get("member", [])
                if not members:
                    break
                for sat in members:
                    name = sat.get("name", "STARLINK")
                    l1   = sat.get("line1", "")
                    l2   = sat.get("line2", "")
                    if l1 and l2:
                        all_tles.append(f"{name}\n{l1}\n{l2}")
                time.sleep(0.5)  # be polite

            if all_tles:
                cache["starlink_tle"] = {"data": "\n".join(all_tles), "ts": time.time(), "error": None}
        except Exception as e:
            old = cache.get("starlink_tle", {}).get("data")
            cache["starlink_tle"] = {"data": old, "ts": time.time(), "error": str(e)}
        time.sleep(21600)  # refresh every 6 hours



def fetch_horizons_body(target, center='@sun'):
    """Fetch current ecliptic XY position from JPL Horizons."""
    now  = datetime.datetime.utcnow()
    stop = now + datetime.timedelta(hours=1)
    # Horizons requires format: 'YYYY-Mon-DD HH:MM'
    fmt  = '%Y-%b-%d %H:%M'
    params = {
        'format':     'json',
        'COMMAND':    f"'{target}'",
        'OBJ_DATA':   'NO',
        'MAKE_EPHEM': 'YES',
        'EPHEM_TYPE': 'VECTORS',
        'CENTER':     "'@sun'",
        'START_TIME': f"'{now.strftime(fmt)}'",
        'STOP_TIME':  f"'{stop.strftime(fmt)}'",
        'STEP_SIZE':  "'1h'",
        'VEC_TABLE':  '2',
        'CSV_FORMAT': 'YES'
    }
    r = requests.get('https://ssd.jpl.nasa.gov/api/horizons.api', params=params, timeout=30)
    r.raise_for_status()
    text = r.json().get('result','')
    # Parse $$SOE ... $$EOE block
    lines    = text.split('\n')
    in_block = False
    for line in lines:
        if '$$SOE' in line: in_block = True; continue
        if '$$EOE' in line: break
        if in_block and line.strip():
            parts = line.split(',')
            if len(parts) >= 6:
                try:
                    x    = float(parts[2].strip())
                    y    = float(parts[3].strip())
                    z    = float(parts[4].strip())
                    dist = (x**2 + y**2 + z**2)**0.5 / 1.496e8
                    return {'x': x/1.496e8, 'y': y/1.496e8, 'z': z/1.496e8, 'dist_au': dist}
                except: pass
    return None

def refresh_horizons():
    time.sleep(5)
    # target_id: (label, color, size_px, category)
    bodies = {
        '199':     ('Mercury',        '#b5b5b5', 5,  'planet'),
        '299':     ('Venus',          '#e8cda0', 6,  'planet'),
        '399':     ('Earth',          '#4fa3e0', 7,  'planet'),
        '301':     ('Moon',           '#c8c8c8', 3,  'moon'),
        '499':     ('Mars',           '#c1440e', 6,  'planet'),
        '599':     ('Jupiter',        '#c88b3a', 12, 'planet'),
        '699':     ('Saturn',         '#e4d191', 10, 'planet'),
        '799':     ('Uranus',         '#7de8e8', 8,  'planet'),
        '899':     ('Neptune',        '#5b7fde', 8,  'planet'),
        '999':     ('Pluto',          '#c2a379', 4,  'dwarf'),
        '2000001': ('Ceres',          '#a0a0a0', 4,  'dwarf'),
        '-31':     ('Voyager 1',      '#00ff9d', 4,  'spacecraft'),
        '-32':     ('Voyager 2',      '#00d4ff', 4,  'spacecraft'),
        '-143205': ('Starman/Roadster','#ff6b35',4,  'spacecraft'),
        '-227':    ('New Horizons',   '#c77dff', 4,  'spacecraft'),
        '-74':     ('Mars Recon. Orb.','#ff6b35',3,  'spacecraft'),
        '-48':     ('Parker Solar Probe','#ffb703',3,'spacecraft'),
        '-151':    ('JWST',           '#00d4ff', 3,  'spacecraft'),
        '2000433': ('433 Eros',       '#ffb703', 4,  'neo'),
        '2004179': ('4179 Toutatis',  '#ffb703', 3,  'neo'),
        '2001036': ('1036 Ganymed',   '#ffb703', 3,  'neo'),
        '2000887': ('887 Alinda',     '#ffb703', 3,  'neo'),
        '2025143': ('25143 Itokawa',  '#ffb703', 3,  'neo'),
        '2001620': ('1620 Geographos','#ffb703', 3,  'neo'),
        '2001866': ('1866 Sisyphus',  '#ffb703', 3,  'neo'),
        '2003552': ('3552 Don Quixote','#ffb703',3,  'neo'),
        '2001980': ('1980 Tezcatlipoca','#ffb703',3, 'neo'),
        '2005011': ('5011 Ptah',      '#ffb703', 3,  'neo'),
    }
    while True:
        results = {}
        for target_id, meta in bodies.items():
            try:
                pos = fetch_horizons_body(target_id)
                if pos:
                    results[target_id] = {
                        'name':     meta[0],
                        'color':    meta[1],
                        'size':     meta[2],
                        'category': meta[3],
                        **pos
                    }
                time.sleep(0.3)
            except Exception as e:
                print(f"Horizons error {target_id}: {e}")
        if results:
            cache['horizons'] = {'data': results, 'ts': time.time(), 'error': None}
        time.sleep(86400)  # refresh daily

# ---------------------------------------------------------------------------
# Ares — aisstream.io WebSocket thread
# ---------------------------------------------------------------------------

def _ares_handle_message(ws, raw):
    try:
        data     = json.loads(raw)
        msg_type = data.get("MessageType", "")
        meta     = data.get("MetaData", {})
        mmsi     = str(meta.get("MMSI", "")).strip()
        if not mmsi:
            return

        with ares_lock:
            if mmsi not in ares_ships:
                ares_ships[mmsi] = {"mmsi": mmsi, "last_seen": time.time()}

            if msg_type == "PositionReport":
                pr      = data.get("Message", {}).get("PositionReport", {})
                heading = pr.get("TrueHeading")
                if heading is None or heading == 511:   # 511 = not available in AIS spec
                    cog = pr.get("Cog")
                    heading = cog if (cog is not None and cog != 360.0) else None
                speed = pr.get("SpeedOverGround")
                if speed == 102.3:                       # 102.3 = not available in AIS spec
                    speed = None
                lat = pr.get("Latitude")
                lon = pr.get("Longitude")
                ares_ships[mmsi].update({
                    "lat":       lat if (lat is not None and lat != 91.0) else None,
                    "lon":       lon if (lon is not None and lon != 181.0) else None,
                    "speed":     round(speed, 1) if speed is not None else None,
                    "heading":   int(heading) if heading is not None else None,
                    "last_seen": time.time(),
                })
                if "country" not in ares_ships[mmsi]:
                    ares_ships[mmsi]["country"] = mmsi_to_country(mmsi)

            elif msg_type == "ShipStaticData":
                sd        = data.get("Message", {}).get("ShipStaticData", {})
                type_code = sd.get("Type", 0)
                name      = (sd.get("Name") or meta.get("ShipName") or "").strip() or mmsi
                ares_ships[mmsi].update({
                    "name":      name,
                    "type":      ship_type_label(type_code),
                    "military":  is_military(type_code),
                    "country":   mmsi_to_country(mmsi),
                    "last_seen": time.time(),
                })

    except Exception as e:
        print(f"[Ares] Message error: {e}")


def _ares_on_open(ws):
    sub = json.dumps({
        "APIKey":             AISSTREAM_KEY,
        "BoundingBoxes":      ARES_BOUNDS,
        "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
    })
    ws.send(sub)
    print("[Ares] Connected to aisstream.io")


def _run_ares():
    delay = 30  # start cautious after the 429
    while True:
        try:
            ws = websocket.WebSocketApp(
                "wss://stream.aisstream.io/v0/stream",
                on_message = _ares_handle_message,
                on_open    = _ares_on_open,
                on_error   = lambda ws, e: print(f"[Ares] WS error: {e}"),
                on_close   = lambda ws, c, m: print(f"[Ares] WS closed: {c}"),
            )
            ws.run_forever(ping_interval=30, ping_timeout=10)
            delay = 30  # reset on clean disconnect
        except Exception as e:
            print(f"[Ares] Connection error: {e}")
        with ares_lock:
            prune_stale_ships(ares_ships)
        print(f"[Ares] Reconnecting in {delay}s...")
        time.sleep(delay)
        delay = min(delay * 2, 3600)  # cap at 1 hour


def init_cache():
    # Ares
    threading.Thread(target=_run_ares, daemon=True).start()
    # SAR
    threading.Thread(target=_run_sar_refresh, daemon=True).start()

    # Sky Watch
    threading.Thread(target=refresh_flights, daemon=True).start()
    start("iss_pos",    "http://api.open-notify.org/iss-now.json", 30)
    start("iss_crew",   "http://api.open-notify.org/astros.json", 21600)
    start("iss_passes", "http://api.open-notify.org/iss-pass.json", 3600,
          params={"lat": OBS_LAT, "lon": OBS_LON, "n": 5})
    start("starlink", "https://api.celestrak.com/SOCRATES/query.php", 600, params={
        "OBJECT": "STARLINK", "OBS_LAT": OBS_LAT, "OBS_LON": OBS_LON,
        "OBS_ELEVATION": OBS_ELEV, "DAYS": 2, "MAX_MATCHES": 10, "FORMAT": "JSON"
    })

    # Space Weather
    start("kp",                "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json", 600)
    start("solar_wind_plasma", "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json", 300)
    start("solar_wind_mag",    "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json", 300)
    start("sw_alerts",         "https://services.swpc.noaa.gov/products/alerts.json", 300)
    start("xray",              "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json", 300)
    start("gravwaves",         "https://gracedb.ligo.org/api/superevents/?format=json&order=-created&page_size=10", 300)

    # Solar System

    today = datetime.date.today()
    start("neo", "https://api.nasa.gov/neo/rest/v1/feed", 21600, params={
        "start_date": today.isoformat(),
        "end_date":   (today + datetime.timedelta(days=7)).isoformat(),
        "api_key":    NASA_KEY
    })
    start("fireballs", "https://ssd-api.jpl.nasa.gov/fireball.api", 3600, params={"limit": 20})

    # Solar System — Horizons positions (planets, spacecraft, NEOs, Starman)
    threading.Thread(target=refresh_horizons, daemon=True).start()
    start("apod_week", "https://api.nasa.gov/planetary/apod", 21600, params={
        "api_key":    NASA_KEY,
        "start_date": (today - datetime.timedelta(days=7)).isoformat(),
        "end_date":   (today - datetime.timedelta(days=1)).isoformat()
    })

    # Earth — Weather
    start("weather_current", "https://api.open-meteo.com/v1/forecast", 1800, params={
        "latitude":            OBS_LAT,
        "longitude":           OBS_LON,
        "current":             "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m,wind_gusts_10m,uv_index,surface_pressure,cloud_cover,visibility",
        "hourly":              "temperature_2m,relative_humidity_2m,precipitation_probability,cloud_cover,uv_index,wind_speed_10m,surface_pressure,visibility,weather_code",
        "daily":               "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,uv_index_max,sunrise,sunset",
        "timezone":            "Australia/Melbourne",
        "forecast_days":       8
    }, startup_delay=10)

    # Earth — Air Quality (EPA)
    threading.Thread(target=refresh_epa, daemon=True).start()
    threading.Thread(target=refresh_starlink_tle, daemon=True).start()

    # Earth — Earthquakes Victoria (Geoscience Australia)
    # Last 24 hours — all magnitudes
    start("quakes_vic_day", "https://earthquake.usgs.gov/fdsnws/event/1/query", 300, params={
        "format":    "geojson",
        "starttime": (datetime.datetime.utcnow() - datetime.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S"),
        "minlatitude":  -39.2,
        "maxlatitude":  -33.9,
        "minlongitude": 140.9,
        "maxlongitude": 150.0,
        "orderby":   "time"
    })
    # Last 30 days — M0.5+
    start("quakes_vic_month", "https://earthquake.usgs.gov/fdsnws/event/1/query", 3600, params={
        "format":    "geojson",
        "starttime": (datetime.datetime.utcnow() - datetime.timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S"),
        "minlatitude":  -39.2,
        "maxlatitude":  -33.9,
        "minlongitude": 140.9,
        "maxlongitude": 150.0,
        "minmagnitude": 0.5,
        "orderby":   "time"
    })

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def cached(key):
    entry = cache.get(key)
    if not entry or not entry["data"]:
        return jsonify({"error": entry["error"] if entry else "Not yet loaded"}), 503
    return jsonify(entry["data"])

# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/skywatch")
def skywatch():
    return render_template("skywatch.html")

@app.route("/spaceweather")
def spaceweather():
    return render_template("spaceweather.html")

@app.route("/solarsystem")
def solarsystem():
    return render_template("solarsystem.html")

@app.route("/deepspace")
def deepspace():
    return render_template("deepspace.html")

@app.route("/earth")
def earth():
    return render_template("earth.html")

# ---------------------------------------------------------------------------
# API — Sky Watch
# ---------------------------------------------------------------------------

@app.route("/api/flights")
def api_flights():
    return cached("flights")

@app.route("/api/iss")
def api_iss():
    return cached("iss_pos")

@app.route("/api/iss/crew")
def api_iss_crew():
    return cached("iss_crew")

@app.route("/api/iss/passes")
def api_iss_passes():
    return cached("iss_passes")

@app.route("/api/starlink/tle")
def api_starlink_tle():
    entry = cache.get("starlink_tle")
    if not entry or not entry["data"]:
        return "Not yet loaded", 503
    return entry["data"], 200, {"Content-Type": "text/plain"}


    return cached("starlink")

# ---------------------------------------------------------------------------
# API — Space Weather
# ---------------------------------------------------------------------------

@app.route("/api/spaceweather/kp")
def api_kp():
    return cached("kp")

@app.route("/api/spaceweather/solar_wind")
def api_solar_wind():
    entry_p = cache.get("solar_wind_plasma")
    entry_m = cache.get("solar_wind_mag")
    return jsonify({
        "plasma": entry_p["data"] if entry_p else None,
        "mag":    entry_m["data"] if entry_m else None
    })

@app.route("/api/spaceweather/alerts")
def api_sw_alerts():
    return cached("sw_alerts")

@app.route("/api/spaceweather/xray")
def api_xray():
    return cached("xray")

@app.route("/api/gravwaves")
def api_gravwaves():
    return cached("gravwaves")

@app.route("/api/sdo/<image_id>")
def api_sdo(image_id):
    allowed = ['0171','0304','0193','0094','HMIB']
    if image_id not in allowed:
        return "Not found", 404
    url = f"https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_{image_id}.jpg"
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "curl/7.88.1"})
        r.raise_for_status()
        from flask import Response
        return Response(r.content, mimetype="image/jpeg",
                        headers={"Cache-Control": "max-age=900"})
    except Exception as e:
        return str(e), 503

# ---------------------------------------------------------------------------
# API — Solar System
# ---------------------------------------------------------------------------

@app.route("/api/solarsystem")
def api_solarsystem():
    return cached("horizons")




@app.route("/api/neo")
def api_neo():
    return cached("neo")

@app.route("/api/fireballs")
def api_fireballs():
    return cached("fireballs")

# ---------------------------------------------------------------------------
# API — Deep Space
# ---------------------------------------------------------------------------

@app.route("/api/apod")
def api_apod():
    return cached("apod")

@app.route("/api/apod/week")
def api_apod_week():
    return cached("apod_week")

# ---------------------------------------------------------------------------
# API — Earth
# ---------------------------------------------------------------------------

@app.route("/api/weather")
def api_weather():
    return cached("weather_current")

@app.route("/api/epa")
def api_epa():
    return cached("epa_air")

@app.route("/api/earthquakes/vic/day")
def api_quakes_vic_day():
    # Refresh starttime dynamically so it's always last 24hrs
    try:
        r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", params={
            "format":         "geojson",
            "starttime":      (datetime.datetime.utcnow() - datetime.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S"),
            "minlatitude":    -39.2,
            "maxlatitude":    -33.9,
            "minlongitude":   140.9,
            "maxlongitude":   150.0,
            "orderby":        "time"
        }, timeout=10)
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 503

@app.route("/api/earthquakes/vic/month")
def api_quakes_vic_month():
    return cached("quakes_vic_month")

@app.route("/api/owm_key")
def api_owm_key():
    return jsonify({"key": OWM_KEY})


# ---------------------------------------------------------------------------
# Ares — Routes
# ---------------------------------------------------------------------------

@app.route("/ares")
def page_ares():
    return render_template("ares.html")


@app.route("/api/ares/ships")
def api_ares_ships():
    with ares_lock:
        ships = [s for s in ares_ships.values() if s.get("lat") is not None and s.get("lon") is not None]
    return jsonify({"ships": ships, "count": len(ships), "ts": time.time(), "error": None})


@app.route("/api/ares/sar-tile")
def api_ares_sar_tile():
    token = get_sar_token()
    if not token:
        return "SAR auth unavailable", 503
    try:
        params = dict(request.args)
        resp = requests.get(
            f"https://sh.dataspace.copernicus.eu/ogc/wms/{COPERNICUS_INSTANCE_ID}",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
            timeout=20
        )
        return Response(resp.content, content_type=resp.headers.get("content-type", "image/png"))
    except Exception as e:
        return f"SAR tile error: {e}", 502


@app.route("/api/ares/sar-ships")
def api_ares_sar_ships():
    with sar_lock:
        ships  = list(sar_ships)
        status = dict(sar_status)
    return jsonify({"ships": ships, "count": len(ships), "status": status})


@app.route("/api/ares/sar-status")
def api_ares_sar_status():
    with sar_lock:
        status = dict(sar_status)
    return jsonify(status)


if __name__ == "__main__":
    init_cache()
    app.run(host="0.0.0.0", port=5003, debug=False)
