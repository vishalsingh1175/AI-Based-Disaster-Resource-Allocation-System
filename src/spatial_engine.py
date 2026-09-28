"""
Spatial calculation engine for Disaster Resource Allocation.
Implements Haversine distance calculations, proximity radius queries,
k-nearest available resource retrieval, and rule-based priority scoring.
"""

import math
from typing import List, Dict, Any, Optional
import pandas as pd


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth in kilometers.
    """
    R = 6371.0  # Earth's radius in kilometers

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) *
         math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return round(R * c, 2)


def estimate_travel_time_minutes(distance_km: float, avg_speed_kmh: float = 35.0, dispatch_delay_min: float = 3.0) -> float:
    """
    Estimate response travel time in minutes based on distance and emergency transit speed.
    Adds a baseline dispatch delay (preparation & sirens turn-on).
    """
    travel_time_hours = distance_km / max(avg_speed_kmh, 5.0)
    total_minutes = (travel_time_hours * 60.0) + dispatch_delay_min
    return round(total_minutes, 1)


def calculate_rule_based_priority(severity: str, affected_population: int) -> float:
    """
    Baseline heuristic priority score (0.0 to 100.0) for Micro Project Milestone.
    Designed to be extensible: Phase 2 will replace this with Scikit-learn ML model output.
    """
    severity_weights = {
        "Critical": 50.0,
        "High": 35.0,
        "Medium": 20.0,
        "Low": 10.0
    }
    base = severity_weights.get(severity, 15.0)
    
    # Population impact factor (logarithmic scale up to 40 pts)
    pop_factor = min(40.0, math.log10(max(affected_population, 1)) * 10.0)
    
    score = base + pop_factor
    return round(min(score, 100.0), 1)


def find_nearest_resources(
    incident_lat: float,
    incident_lon: float,
    resources_df: pd.DataFrame,
    resource_type: Optional[str] = None,
    only_available: bool = True,
    max_results: int = 5,
    max_radius_km: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Find nearest emergency resources to an incident location.
    
    Returns a sorted list of resource dicts with calculated distance and ETA.
    """
    filtered_df = resources_df.copy()

    if only_available:
        filtered_df = filtered_df[filtered_df["availability_status"].isin(["Available", "Limited"])]

    if resource_type and resource_type != "All":
        filtered_df = filtered_df[filtered_df["resource_type"] == resource_type]

    if filtered_df.empty:
        return []

    results = []
    for _, res in filtered_df.iterrows():
        dist = haversine_distance(incident_lat, incident_lon, res["latitude"], res["longitude"])
        
        if max_radius_km and dist > max_radius_km:
            continue
            
        eta = estimate_travel_time_minutes(dist)
        
        results.append({
            "resource_id": res["resource_id"],
            "name": res["name"],
            "resource_type": res["resource_type"],
            "latitude": res["latitude"],
            "longitude": res["longitude"],
            "base_station": res["base_station"],
            "capacity": res["capacity"],
            "availability_status": res["availability_status"],
            "contact_number": res.get("contact_number", "N/A"),
            "distance_km": dist,
            "eta_minutes": eta
        })

    # Sort ascending by distance
    results.sort(key=lambda x: x["distance_km"])
    return results[:max_results]


def compute_system_coverage_stats(incidents_df: pd.DataFrame, resources_df: pd.DataFrame, threshold_radius_km: float = 7.0) -> Dict[str, Any]:
    """
    Calculate high-level GIS coverage metrics for the entire jurisdiction.
    """
    total_incidents = len(incidents_df)
    if total_incidents == 0:
        return {"covered_incidents": 0, "coverage_pct": 100.0, "avg_min_distance_km": 0.0}

    covered_count = 0
    min_distances = []

    available_resources = resources_df[resources_df["availability_status"].isin(["Available", "Limited"])]

    for _, inc in incidents_df.iterrows():
        if available_resources.empty:
            min_dist = float("inf")
        else:
            dists = [
                haversine_distance(inc["latitude"], inc["longitude"], r_lat, r_lon)
                for r_lat, r_lon in zip(available_resources["latitude"], available_resources["longitude"])
            ]
            min_dist = min(dists) if dists else float("inf")

        min_distances.append(min_dist if min_dist != float("inf") else 0.0)
        if min_dist <= threshold_radius_km:
            covered_count += 1

    coverage_pct = round((covered_count / total_incidents) * 100.0, 1)
    avg_min_dist = round(sum(min_distances) / max(len(min_distances), 1), 2)

    return {
        "total_incidents": total_incidents,
        "covered_incidents": covered_count,
        "uncovered_incidents": total_incidents - covered_count,
        "coverage_pct": coverage_pct,
        "avg_min_distance_km": avg_min_dist,
        "threshold_radius_km": threshold_radius_km
    }
