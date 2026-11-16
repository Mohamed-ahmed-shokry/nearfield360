"""Standardized HD map format exporters: OpenDRIVE XML, Lanelet2 OSM XML, and GeoJSON."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from nearfield360.hdmap.models import HDVectorMap, Lanelet, ParkingSpaceFeature, Polyline2D


class MapFormatExporter:
    """Exports HDVectorMap representations to standard automotive and GIS exchange formats."""

    @staticmethod
    def to_geojson(vector_map: HDVectorMap) -> dict[str, Any]:
        """Export map to RFC 7946 GeoJSON FeatureCollection dictionary."""
        features: list[dict[str, Any]] = []

        # Deduplicate lanelets, parking spaces, and obstacles across tiles
        seen_lanes: dict[str, Lanelet] = {}
        seen_spaces: dict[str, ParkingSpaceFeature] = {}

        for t in vector_map.tiles:
            for lane in t.lanelets:
                seen_lanes[lane.lanelet_id] = lane
            for space in t.parking_spaces:
                seen_spaces[space.slot_id] = space

        # Lanelet centerlines
        features.extend(
            {
                "type": "Feature",
                "id": f"lane_center_{lane.lanelet_id}",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [p.as_tuple() for p in lane.centerline.points],
                },
                "properties": {
                    "feature_type": "lanelet_centerline",
                    "lanelet_id": lane.lanelet_id,
                    "speed_limit_mps": lane.speed_limit_mps,
                    "lanelet_type": lane.lanelet_type.value,
                    "turn_direction": lane.turn_direction.value,
                    "predecessors": lane.predecessor_ids,
                    "successors": lane.successor_ids,
                },
            }
            for lane in seen_lanes.values()
        )

        # Parking spaces
        for space in seen_spaces.values():
            poly_coords = [p.as_tuple() for p in space.boundary.points]
            if not space.boundary.is_closed or poly_coords[0] != poly_coords[-1]:
                poly_coords.append(poly_coords[0])

            features.append(
                {
                    "type": "Feature",
                    "id": f"space_{space.slot_id}",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [poly_coords],
                    },
                    "properties": {
                        "feature_type": "parking_space",
                        "slot_id": space.slot_id,
                        "slot_type": space.slot_type.value,
                        "heading_rad": space.heading_rad,
                        "width_m": space.width_m,
                        "length_m": space.length_m,
                    },
                }
            )

        return {
            "type": "FeatureCollection",
            "properties": {
                "map_id": vector_map.map_id,
                "facility_id": vector_map.facility_id,
                "level_id": vector_map.level_id,
                "total_tiles": len(vector_map.tiles),
            },
            "features": features,
        }

    @staticmethod
    def to_lanelet2_osm(vector_map: HDVectorMap) -> str:
        """Export map to Lanelet2 OSM XML schema string."""
        osm = ET.Element("osm", version="0.6", generator="NearField360 HDMap Engine")

        node_id_counter = 1
        way_id_counter = 1000
        relation_id_counter = 10000

        seen_lanes: dict[str, Lanelet] = {}
        for t in vector_map.tiles:
            for lane in t.lanelets:
                seen_lanes[lane.lanelet_id] = lane

        def _add_way(poly: Polyline2D, role_tag: str) -> int:
            nonlocal node_id_counter, way_id_counter
            way_id = way_id_counter
            way_id_counter += 1

            way_elem = ET.SubElement(osm, "way", id=str(way_id))
            for pt in poly.points:
                nid = node_id_counter
                node_id_counter += 1
                ET.SubElement(
                    osm,
                    "node",
                    id=str(nid),
                    lat=str(pt.y),
                    lon=str(pt.x),
                    x=f"{pt.x:.3f}",
                    y=f"{pt.y:.3f}",
                )
                ET.SubElement(way_elem, "nd", ref=str(nid))

            ET.SubElement(way_elem, "tag", k="type", v="line_thin")
            ET.SubElement(way_elem, "tag", k="role", v=role_tag)
            return way_id

        for lane in seen_lanes.values():
            left_way_id = _add_way(lane.left_boundary, "left")
            right_way_id = _add_way(lane.right_boundary, "right")
            center_way_id = _add_way(lane.centerline, "centerline")

            rel_id = relation_id_counter
            relation_id_counter += 1
            rel_elem = ET.SubElement(osm, "relation", id=str(rel_id))
            ET.SubElement(rel_elem, "member", type="way", role="left", ref=str(left_way_id))
            ET.SubElement(rel_elem, "member", type="way", role="right", ref=str(right_way_id))
            ET.SubElement(rel_elem, "member", type="way", role="centerline", ref=str(center_way_id))

            ET.SubElement(rel_elem, "tag", k="type", v="lanelet")
            ET.SubElement(rel_elem, "tag", k="subtype", v="road")
            ET.SubElement(rel_elem, "tag", k="lanelet_id", v=lane.lanelet_id)
            ET.SubElement(rel_elem, "tag", k="speed_limit", v=f"{lane.speed_limit_mps:.1f}")

        ET.indent(osm, space="  ")
        return ET.tostring(osm, encoding="unicode")

    @staticmethod
    def to_opendrive_xml(vector_map: HDVectorMap) -> str:
        """Export map to ASAM OpenDRIVE XML schema string."""
        od = ET.Element("OpenDRIVE")
        header = ET.SubElement(
            od,
            "header",
            revMajor="1",
            revMinor="7",
            name=vector_map.map_id,
            version="1.0",
            date="2026-11-16",
        )
        ET.SubElement(
            header,
            "geoReference",
        ).text = "+proj=tmerc +lat_0=0.0 +lon_0=0.0 +k=1.0 +x_0=0 +y_0=0 +datum=WGS84"

        seen_lanes: dict[str, Lanelet] = {}
        for t in vector_map.tiles:
            for lane in t.lanelets:
                seen_lanes[lane.lanelet_id] = lane

        road_id = 1
        for lane in seen_lanes.values():
            pts = lane.centerline.points
            road_len = lane.centerline.length_m
            road = ET.SubElement(
                od,
                "road",
                id=str(road_id),
                name=lane.lanelet_id,
                length=f"{road_len:.3f}",
                junction="-1",
            )
            road_id += 1

            # planView
            plan_view = ET.SubElement(road, "planView")
            ET.SubElement(
                plan_view,
                "geometry",
                s="0.0",
                x=f"{pts[0].x:.3f}",
                y=f"{pts[0].y:.3f}",
                hdg="0.0",
                length=f"{road_len:.3f}",
            )

            # lanes
            lanes = ET.SubElement(road, "lanes")
            lane_section = ET.SubElement(lanes, "laneSection", s="0.0")
            center = ET.SubElement(lane_section, "center")
            ET.SubElement(center, "lane", id="0", type="driving", level="false")

        ET.indent(od, space="  ")
        return ET.tostring(od, encoding="unicode")
