"""Unit tests for HD map format exporters (GeoJSON, Lanelet2 OSM, OpenDRIVE XML)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from nearfield360.hdmap.formats import MapFormatExporter
from nearfield360.hdmap.models import (
    HDVectorMap,
    Lanelet,
    MapTile,
    ParkingSpaceFeature,
    Point2D,
    Polyline2D,
)


def _sample_map() -> HDVectorMap:
    left = Polyline2D(points=[Point2D(x=0.0, y=2.0), Point2D(x=20.0, y=2.0)])
    right = Polyline2D(points=[Point2D(x=0.0, y=-2.0), Point2D(x=20.0, y=-2.0)])
    center = Polyline2D(points=[Point2D(x=0.0, y=0.0), Point2D(x=20.0, y=0.0)])
    lane = Lanelet(
        lanelet_id="L-101",
        left_boundary=left,
        right_boundary=right,
        centerline=center,
        speed_limit_mps=3.0,
    )

    slot_poly = Polyline2D(
        points=[
            Point2D(x=10.0, y=5.0),
            Point2D(x=12.5, y=5.0),
            Point2D(x=12.5, y=10.0),
            Point2D(x=10.0, y=10.0),
        ],
        is_closed=True,
    )
    space = ParkingSpaceFeature(
        slot_id="BAY-01",
        boundary=slot_poly,
        center=Point2D(x=11.25, y=7.5),
    )

    tile = MapTile(
        tile_id="TILE_0_0",
        x_min=0.0,
        y_min=-10.0,
        x_max=50.0,
        y_max=40.0,
        lanelets=[lane],
        parking_spaces=[space],
    )
    return HDVectorMap(map_id="MAP-SAMPLE-01", tiles=[tile])


def test_export_geojson_schema() -> None:
    sample = _sample_map()
    geojson = MapFormatExporter.to_geojson(sample)

    assert geojson["type"] == "FeatureCollection"
    assert geojson["properties"]["map_id"] == "MAP-SAMPLE-01"
    assert len(geojson["features"]) == 2

    lane_feat = next(
        f for f in geojson["features"] if f["properties"]["feature_type"] == "lanelet_centerline"
    )
    assert lane_feat["properties"]["lanelet_id"] == "L-101"
    assert lane_feat["geometry"]["type"] == "LineString"
    assert len(lane_feat["geometry"]["coordinates"]) == 2

    space_feat = next(
        f for f in geojson["features"] if f["properties"]["feature_type"] == "parking_space"
    )
    assert space_feat["properties"]["slot_id"] == "BAY-01"
    assert space_feat["geometry"]["type"] == "Polygon"


def test_export_lanelet2_osm() -> None:
    sample = _sample_map()
    osm_xml = MapFormatExporter.to_lanelet2_osm(sample)

    assert "<osm" in osm_xml
    assert "</osm>" in osm_xml
    root = ET.fromstring(osm_xml)  # noqa: S314
    assert root.tag == "osm"
    assert root.attrib["version"] == "0.6"

    # Verify ways and relations exist
    ways = root.findall("way")
    relations = root.findall("relation")
    assert len(ways) >= 3
    assert len(relations) == 1

    rel = relations[0]
    types = [t.attrib["v"] for t in rel.findall("tag") if t.attrib["k"] == "type"]
    assert "lanelet" in types


def test_export_opendrive_xml() -> None:
    sample = _sample_map()
    od_xml = MapFormatExporter.to_opendrive_xml(sample)

    assert "<OpenDRIVE>" in od_xml
    assert "</OpenDRIVE>" in od_xml
    root = ET.fromstring(od_xml)  # noqa: S314
    assert root.tag == "OpenDRIVE"

    header = root.find("header")
    assert header is not None
    assert header.attrib["name"] == "MAP-SAMPLE-01"

    roads = root.findall("road")
    assert len(roads) == 1
    assert roads[0].attrib["name"] == "L-101"
