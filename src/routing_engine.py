"""
Graph-Based Road Network Routing & Dynamic Hazard Detour Engine.
Constructs a topological arterial road graph for Delhi-NCR using NetworkX.
Implements Dijkstra shortest-path computation with real-time road blockage
and flood inundation simulation.
Follows the routing and graph algorithms in Section 5.4 / 7.2 of the project specification.
"""

import math
from typing import List, Dict, Tuple, Optional, Any
import networkx as nx
from src.spatial_engine import haversine_distance, estimate_travel_time_minutes


# Key arterial network vertices across Delhi-NCR
ROAD_GRAPH_NODES = {
    "KASHMERE_GATE": {"coords": (28.6670, 77.2250), "label": "Kashmere Gate ISBT Junction"},
    "YAMUNA_BRIDGE": {"coords": (28.6610, 77.2400), "label": "Old Yamuna Bridge Causeway"},
    "CIVIL_LINES": {"coords": (28.6750, 77.2200), "label": "Civil Lines / Ring Road North"},
    "NDLS_CP": {"coords": (28.6360, 77.2180), "label": "Connaught Place / NDLS Arterial"},
    "ITO_JUNCTION": {"coords": (28.6300, 77.2400), "label": "ITO Vikas Marg Intersection"},
    "AIIMS_SOUTH": {"coords": (28.5672, 77.2100), "label": "AIIMS Ansari Nagar Flyover"},
    "LAJPAT_NAGAR": {"coords": (28.5700, 77.2400), "label": "Lajpat Nagar Ring Road Link"},
    "OKHLA_IND": {"coords": (28.5350, 77.2710), "label": "Okhla Industrial Access Junction"},
    "DDU_HARI_NAGAR": {"coords": (28.6278, 77.1065), "label": "Hari Nagar West Corridor"},
    "PASCHIM_VIHAR": {"coords": (28.6700, 77.1120), "label": "Paschim Vihar Outer Ring Road"},
    "ROHINI_SEC14": {"coords": (28.7125, 77.1150), "label": "Rohini Institutional Node"},
    "MUNDKA_IND": {"coords": (28.6812, 77.0321), "label": "Mundka Industrial Rohtak Road"},
    "DWARKA_SEC8": {"coords": (28.5833, 77.0500), "label": "Dwarka Expressway Link Node"}
}

# Major arterial corridors with default lengths (km) and corridor names
ROAD_GRAPH_EDGES = [
    ("KASHMERE_GATE", "YAMUNA_BRIDGE", 2.1, "Yamuna Khadar Lowland Corridor", True),
    ("YAMUNA_BRIDGE", "ITO_JUNCTION", 3.8, "Vikas Marg Riverfront Artery", True),
    ("CIVIL_LINES", "KASHMERE_GATE", 1.5, "North Ring Road Segment", False),
    ("CIVIL_LINES", "ROHINI_SEC14", 11.2, "GT Karnal Road Expressway", False),
    ("KASHMERE_GATE", "NDLS_CP", 4.5, "SP Mukherjee Marg Artery", False),
    ("NDLS_CP", "ITO_JUNCTION", 2.3, "DDU Marg Corridor", False),
    ("ITO_JUNCTION", "LAJPAT_NAGAR", 7.2, "Ring Road Southbound", False),
    ("LAJPAT_NAGAR", "AIIMS_SOUTH", 3.4, "Inner Ring Road Flyover Network", False),
    ("LAJPAT_NAGAR", "OKHLA_IND", 6.8, "Mathura Road / Okhla Link", False),
    ("AIIMS_SOUTH", "DDU_HARI_NAGAR", 11.5, "Mahatma Gandhi Marg Westbound", False),
    ("DDU_HARI_NAGAR", "DWARKA_SEC8", 8.4, "Pankha Road Expressway", False),
    ("DDU_HARI_NAGAR", "PASCHIM_VIHAR", 5.6, "Outer Ring Road West", False),
    ("PASCHIM_VIHAR", "MUNDKA_IND", 8.2, "Rohtak Road NH-10 Highway", True),
    ("PASCHIM_VIHAR", "ROHINI_SEC14", 6.1, "Outer Ring Road North Corridor", False),
    ("ROHINI_SEC14", "MUNDKA_IND", 9.8, "Ranikhera Connecting Road", False),
    ("NDLS_CP", "AIIMS_SOUTH", 8.1, "Aurobindo Marg North-South Link", False),
    ("NDLS_CP", "DDU_HARI_NAGAR", 12.0, "Patel Road / Shadipur Flyover", False),
    ("KASHMERE_GATE", "PASCHIM_VIHAR", 12.8, "Rohtak Road Inbound Connector", False)
]


class DelhiRoadGraphNetwork:
    """
    Manages the road graph network and dynamically calculates routes
    with simulated hazards/flooding road blocks.
    """
    def __init__(self):
        self.graph = nx.Graph()
        self.active_blockages = set()
        self._build_graph()

    def _build_graph(self):
        self.graph.clear()
        for node_id, data in ROAD_GRAPH_NODES.items():
            self.graph.add_node(node_id, coords=data["coords"], label=data["label"])
            
        for u, v, dist, name, can_flood in ROAD_GRAPH_EDGES:
            is_blocked = (u, v) in self.active_blockages or (v, u) in self.active_blockages
            weight = 999999.0 if is_blocked else dist
            self.graph.add_edge(u, v, weight=weight, distance_km=dist, name=name, can_flood=can_flood, is_blocked=is_blocked)

    def set_blockage(self, u: str, v: str, blocked: bool = True):
        """Toggle blockage on a specific arterial corridor."""
        edge = tuple(sorted([u, v]))
        if blocked:
            self.active_blockages.add(edge)
        else:
            self.active_blockages.discard(edge)
        self._build_graph()

    def find_nearest_node(self, lat: float, lon: float) -> str:
        """Find the closest road network intersection node to a given GPS point."""
        closest_node = None
        min_dist = float("inf")
        for node_id, data in ROAD_GRAPH_NODES.items():
            n_lat, n_lon = data["coords"]
            d = haversine_distance(lat, lon, n_lat, n_lon)
            if d < min_dist:
                min_dist = d
                closest_node = node_id
        return closest_node

    def calculate_road_route(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float
    ) -> Dict[str, Any]:
        """
        Calculates turn-by-turn road route between two coordinates using
        Dijkstra shortest path on the road graph network.
        """
        start_node = self.find_nearest_node(start_lat, start_lon)
        end_node = self.find_nearest_node(end_lat, end_lon)

        # Baseline connector distances from exact points to network nodes
        start_conn = haversine_distance(start_lat, start_lon, ROAD_GRAPH_NODES[start_node]["coords"][0], ROAD_GRAPH_NODES[start_node]["coords"][1])
        end_conn = haversine_distance(end_lat, end_lon, ROAD_GRAPH_NODES[end_node]["coords"][0], ROAD_GRAPH_NODES[end_node]["coords"][1])

        if start_node == end_node:
            direct_dist = haversine_distance(start_lat, start_lon, end_lat, end_lon)
            return {
                "success": True,
                "path_nodes": [start_node],
                "route_coordinates": [[start_lat, start_lon], ROAD_GRAPH_NODES[start_node]["coords"], [end_lat, end_lon]],
                "total_distance_km": round(direct_dist, 2),
                "estimated_eta_min": estimate_travel_time_minutes(direct_dist),
                "detour_active": False,
                "corridors_traversed": ["Local Access Road"]
            }

        try:
            path = nx.shortest_path(self.graph, source=start_node, target=end_node, weight="weight")
            
            # Check if any edge in path is marked as blocked (inf weight)
            network_dist = 0.0
            corridors = []
            route_coords = [[start_lat, start_lon]]
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_data = self.graph[u][v]
                network_dist += edge_data["distance_km"]
                corridors.append(edge_data["name"])
                route_coords.append(ROAD_GRAPH_NODES[u]["coords"])
                
            route_coords.append(ROAD_GRAPH_NODES[path[-1]]["coords"])
            route_coords.append([end_lat, end_lon])

            total_dist = round(network_dist + start_conn + end_conn, 2)
            eta = estimate_travel_time_minutes(total_dist)

            return {
                "success": True,
                "path_nodes": path,
                "route_coordinates": route_coords,
                "total_distance_km": total_dist,
                "estimated_eta_min": eta,
                "detour_active": len(self.active_blockages) > 0,
                "corridors_traversed": corridors
            }
        except nx.NetworkXNoPath:
            # Fallback to straight-line if isolated by blockages
            direct_dist = haversine_distance(start_lat, start_lon, end_lat, end_lon)
            return {
                "success": False,
                "path_nodes": [],
                "route_coordinates": [[start_lat, start_lon], [end_lat, end_lon]],
                "total_distance_km": round(direct_dist, 2),
                "estimated_eta_min": estimate_travel_time_minutes(direct_dist) + 15.0,
                "detour_active": True,
                "corridors_traversed": ["Hazard Blockage: Alternate Air / Amphibious Transit Required"]
            }


# Singleton router
_road_router = None

def get_road_router() -> DelhiRoadGraphNetwork:
    global _road_router
    if _road_router is None:
        _road_router = DelhiRoadGraphNetwork()
    return _road_router
