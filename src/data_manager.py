"""
Data loading, validation, and cleaning module.
Ensures data integrity for GIS mapping and downstream allocation pipelines.
Satisfies Micro Project Milestone 1 (Validated disaster/resource data).
"""

import os
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from src.spatial_engine import calculate_rule_based_priority


# Geographic bounding box for Delhi-NCR validation
DELHI_NCR_BOUNDS = {
    "min_lat": 28.30,
    "max_lat": 28.95,
    "min_lon": 76.80,
    "max_lon": 77.55
}


def load_datasets(
    incidents_path: str = "data/delhi_incidents.csv",
    resources_path: str = "data/delhi_resources.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw incidents and resources CSV files.
    """
    if not os.path.exists(incidents_path) or not os.path.exists(resources_path):
        raise FileNotFoundError("Dataset files not found. Please run data/seed_data.py first.")

    incidents_df = pd.read_csv(incidents_path)
    resources_df = pd.read_csv(resources_path)
    return incidents_df, resources_df


def validate_and_clean_datasets(
    incidents_df: pd.DataFrame,
    resources_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Perform structural and geographic validation on incidents and resources.
    Returns cleaned DataFrames and a comprehensive audit report for the review panel.
    """
    audit = {
        "incidents_total": len(incidents_df),
        "resources_total": len(resources_df),
        "incidents_valid_coords": 0,
        "resources_valid_coords": 0,
        "duplicate_incident_ids": 0,
        "duplicate_resource_ids": 0,
        "missing_values_imputed": 0,
        "validation_passed": True,
        "logs": []
    }

    # 1. Deduplication Check
    dup_inc = incidents_df.duplicated(subset=["incident_id"]).sum()
    dup_res = resources_df.duplicated(subset=["resource_id"]).sum()
    audit["duplicate_incident_ids"] = int(dup_inc)
    audit["duplicate_resource_ids"] = int(dup_res)

    clean_inc = incidents_df.drop_duplicates(subset=["incident_id"]).copy()
    clean_res = resources_df.drop_duplicates(subset=["resource_id"]).copy()

    # 2. Coordinates Range & Bounding Box Check
    def check_coords(df: pd.DataFrame, name: str) -> Tuple[pd.DataFrame, int]:
        valid_mask = (
            df["latitude"].between(DELHI_NCR_BOUNDS["min_lat"], DELHI_NCR_BOUNDS["max_lat"]) &
            df["longitude"].between(DELHI_NCR_BOUNDS["min_lon"], DELHI_NCR_BOUNDS["max_lon"])
        )
        valid_count = int(valid_mask.sum())
        invalid_count = len(df) - valid_count
        if invalid_count > 0:
            audit["logs"].append(f"Warning: {invalid_count} records in {name} fell outside standard Delhi-NCR bounds.")
        return df[valid_mask].copy(), valid_count

    clean_inc, audit["incidents_valid_coords"] = check_coords(clean_inc, "incidents")
    clean_res, audit["resources_valid_coords"] = check_coords(clean_res, "resources")

    # 3. Missing Value Handling & Type Coercion
    clean_inc["affected_population"] = clean_inc["affected_population"].fillna(0).astype(int)
    clean_inc["notes"] = clean_inc["notes"].fillna("No additional operational notes.")
    clean_res["capacity"] = clean_res["capacity"].fillna(1).astype(int)
    clean_res["contact_number"] = clean_res["contact_number"].fillna("Control Room 112")

    # 4. Compute Micro-stage Rule-based Priority Score for Incidents
    clean_inc["priority_score"] = clean_inc.apply(
        lambda row: calculate_rule_based_priority(row["severity_level"], row["affected_population"]),
        axis=1
    )

    # Sort incidents descending by priority
    clean_inc = clean_inc.sort_values(by="priority_score", ascending=False).reset_index(drop=True)

    audit["validation_passed"] = (
        audit["duplicate_incident_ids"] == 0 and
        audit["incidents_valid_coords"] == len(clean_inc) and
        audit["resources_valid_coords"] == len(clean_res)
    )

    return clean_inc, clean_res, audit


def create_and_validate_incident(
    incident_dict: Dict[str, Any]
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validate and format a newly reported incident from the UI intake form.
    Returns (is_valid, message, formatted_dict).
    """
    lat = float(incident_dict.get("latitude", 0.0))
    lon = float(incident_dict.get("longitude", 0.0))

    # Bounding box check
    if not (DELHI_NCR_BOUNDS["min_lat"] <= lat <= DELHI_NCR_BOUNDS["max_lat"] and
            DELHI_NCR_BOUNDS["min_lon"] <= lon <= DELHI_NCR_BOUNDS["max_lon"]):
        return False, f"Coordinates ({lat}, {lon}) fall outside the Delhi-NCR bounds.", None

    pop = int(incident_dict.get("affected_population", 0))
    severity = incident_dict.get("severity_level", "Medium")
    priority = calculate_rule_based_priority(severity, pop)

    formatted = {
        "incident_id": incident_dict.get("incident_id", f"INC-DEL-{pd.Timestamp.now().strftime('%H%M%S')}"),
        "title": incident_dict.get("title", "Unspecified Incident"),
        "incident_type": incident_dict.get("incident_type", "Flood"),
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "location_name": incident_dict.get("location_name", "Delhi-NCR Sector"),
        "severity_level": severity,
        "affected_population": pop,
        "reported_damage": incident_dict.get("reported_damage", "Initial field assessment in progress."),
        "status": incident_dict.get("status", "Active"),
        "timestamp": incident_dict.get("timestamp", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")),
        "reported_by": incident_dict.get("reported_by", "Field Operations Unit"),
        "notes": incident_dict.get("notes", "Submitted via Operations Intake Console."),
        "priority_score": priority
    }

    return True, "Incident validated successfully.", formatted

