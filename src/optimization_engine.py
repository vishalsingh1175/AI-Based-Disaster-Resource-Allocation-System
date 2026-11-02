"""
Global Multi-Incident Resource Allocation Optimizer.
Formulates and solves the global emergency dispatch bipartite matching problem
using the Hungarian Algorithm (scipy.optimize.linear_sum_assignment).
Maximizes total priority served while minimizing cumulative transit time across all
active disaster zones in Delhi-NCR.
Follows the optimization formulations in Section 7.3 of the project specification.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from src.spatial_engine import haversine_distance, estimate_travel_time_minutes


def run_global_fleet_optimization(
    incidents_df: pd.DataFrame,
    resources_df: pd.DataFrame,
    time_weight_alpha: float = 0.6,
    priority_weight_beta: float = 0.4
) -> Dict[str, Any]:
    """
    Solves the global optimal assignment between all active unassigned incidents
    and all available emergency resources.
    
    Cost Function:
        C(i, j) = alpha * ETA(i, j) - beta * PriorityScore(i)
    """
    # Filter active incidents and available resources
    unassigned_incidents = incidents_df[incidents_df["status"].isin(["Active", "Awaiting Response"])].copy()
    available_resources = resources_df[resources_df["availability_status"].isin(["Available", "Limited"])].copy()

    if unassigned_incidents.empty or available_resources.empty:
        return {
            "success": False,
            "message": "Insufficient active incidents or available resources for global optimization.",
            "assignments": [],
            "total_distance_km": 0.0,
            "avg_eta_min": 0.0,
            "unmet_incidents_count": len(unassigned_incidents),
            "priority_served_pct": 0.0
        }

    n_inc = len(unassigned_incidents)
    n_res = len(available_resources)

    # Cost matrix of shape (n_inc, n_res)
    cost_matrix = np.zeros((n_inc, n_res))
    dist_matrix = np.zeros((n_inc, n_res))
    eta_matrix = np.zeros((n_inc, n_res))

    inc_list = unassigned_incidents.to_dict("records")
    res_list = available_resources.to_dict("records")

    for i, inc in enumerate(inc_list):
        p_score = float(inc.get("priority_score", 50.0))
        for j, res in enumerate(res_list):
            dist = haversine_distance(inc["latitude"], inc["longitude"], res["latitude"], res["longitude"])
            eta = estimate_travel_time_minutes(dist)
            dist_matrix[i, j] = dist
            eta_matrix[i, j] = eta
            # Cost function: higher ETA increases cost, higher priority decreases cost
            cost_matrix[i, j] = (time_weight_alpha * eta) - (priority_weight_beta * p_score)

    # Solve bipartite matching using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments = []
    total_dist = 0.0
    total_eta = 0.0
    total_priority = sum([float(inc.get("priority_score", 50.0)) for inc in inc_list])
    served_priority = 0.0

    for r_idx, c_idx in zip(row_ind, col_ind):
        inc = inc_list[r_idx]
        res = res_list[c_idx]
        dist = float(dist_matrix[r_idx, c_idx])
        eta = float(eta_matrix[r_idx, c_idx])

        total_dist += dist
        total_eta += eta
        served_priority += float(inc.get("priority_score", 50.0))

        assignments.append({
            "incident_id": inc["incident_id"],
            "incident_title": inc["title"],
            "incident_severity": inc["severity_level"],
            "incident_priority": inc.get("priority_score", 50.0),
            "incident_coords": (inc["latitude"], inc["longitude"]),
            "resource_id": res["resource_id"],
            "resource_name": res["name"],
            "resource_type": res["resource_type"],
            "resource_coords": (res["latitude"], res["longitude"]),
            "distance_km": round(dist, 2),
            "eta_minutes": round(eta, 1),
            "base_station": res["base_station"]
        })

    assigned_count = len(assignments)
    avg_eta = round(total_eta / max(assigned_count, 1), 1)
    priority_pct = round((served_priority / max(total_priority, 1.0)) * 100.0, 1)

    return {
        "success": True,
        "message": f"Optimal global match computed for {assigned_count} emergency pairs.",
        "assignments": assignments,
        "total_distance_km": round(total_dist, 2),
        "avg_eta_min": avg_eta,
        "assigned_count": assigned_count,
        "unmet_incidents_count": max(0, n_inc - assigned_count),
        "priority_served_pct": priority_pct
    }
