from ground_control.mdns import instance_name


def test_an_app_is_named_after_home_assistant():
    assert instance_name("local-kiosk-satellite-ground-control", app=True) == \
        "Kiosk Satellite Ground Control (Home Assistant)"


def test_a_long_hostname_fits_in_an_mdns_label():
    name = instance_name("a-very-long-hostname-for-a-docker-container-somewhere", app=False)
    assert len(name.encode()) <= 63 and name.endswith(")")
    assert instance_name("nas", app=False) == "Kiosk Satellite Ground Control (nas)"


def test_only_the_default_route_address_is_advertised():
    from ground_control.mdns import pick

    ips = ["10.10.100.3", "10.11.12.30", "192.168.3.3"]
    assert pick(ips, "192.168.3.3") == ["192.168.3.3"]
    assert pick(ips, None) == ips
    assert pick(ips, "172.17.0.2") == ips
    # An address set by hand is the one advertised.
    assert pick(ips, "192.168.3.3", "10.10.100.3") == ["10.10.100.3"]


def test_a_bad_mdns_address_stops_the_start(monkeypatch, tmp_path):
    import pytest

    from ground_control import config

    monkeypatch.setenv("OPTIONS_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv("MDNS_ADDRESS", "kitchen")
    with pytest.raises(ValueError, match="MDNS_ADDRESS"):
        config.load()
    monkeypatch.setenv("MDNS_ADDRESS", "192.168.1.20")
    assert config.load().mdns_address == "192.168.1.20"
    monkeypatch.setenv("MDNS_ADDRESS", "")
    assert config.load().mdns_address is None
