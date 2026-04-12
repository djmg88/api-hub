import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions we're about to write
from app import ship_type_label, is_military, mmsi_to_country, prune_stale_ships


def test_ship_type_label_tanker():
    assert ship_type_label(80) == "Tanker"
    assert ship_type_label(84) == "Tanker"
    assert ship_type_label(89) == "Tanker"

def test_ship_type_label_cargo():
    assert ship_type_label(70) == "Cargo"
    assert ship_type_label(79) == "Cargo"

def test_ship_type_label_military():
    assert ship_type_label(35) == "Military"

def test_ship_type_label_passenger():
    assert ship_type_label(60) == "Passenger"
    assert ship_type_label(69) == "Passenger"

def test_ship_type_label_unknown():
    assert ship_type_label(0) == "Unknown"
    assert ship_type_label(999) == "Unknown"

def test_is_military_true():
    assert is_military(35) is True

def test_is_military_false():
    assert is_military(80) is False
    assert is_military(70) is False
    assert is_military(0) is False

def test_mmsi_to_country_iran():
    assert mmsi_to_country("422123456") == "Iran"

def test_mmsi_to_country_uae():
    assert mmsi_to_country("470123456") == "UAE"

def test_mmsi_to_country_usa():
    assert mmsi_to_country("368123456") == "USA"

def test_mmsi_to_country_panama():
    assert mmsi_to_country("352123456") == "Panama"

def test_mmsi_to_country_unknown():
    assert mmsi_to_country("000000000") == "Unknown"

def test_prune_stale_ships_removes_old():
    ships = {
        "111": {"last_seen": time.time() - 2000},  # stale
        "222": {"last_seen": time.time() - 100},   # fresh
    }
    prune_stale_ships(ships, max_age=1800)
    assert "111" not in ships
    assert "222" in ships

def test_prune_stale_ships_keeps_fresh():
    ships = {
        "333": {"last_seen": time.time() - 60},
    }
    prune_stale_ships(ships, max_age=1800)
    assert "333" in ships

def test_prune_stale_ships_empty():
    ships = {}
    prune_stale_ships(ships, max_age=1800)
    assert ships == {}
