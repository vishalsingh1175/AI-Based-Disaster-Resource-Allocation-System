"""
Data models and schemas for AI-Based Disaster Resource Allocation System.
Follows the core entities specified in the project synopsis (Section 5.2).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
from datetime import datetime


class IncidentType(str, Enum):
    FLOOD = "Flood"
    FIRE = "Fire / Chemical Hazard"
    STRUCTURAL_COLLAPSE = "Structural Collapse"
    EARTHQUAKE = "Earthquake"
    MEDICAL_OUTBREAK = "Medical Outbreak"
    STORM = "Severe Storm"


class SeverityLevel(str, Enum):
    CRITICAL = "Critical"  # Immediate life threat
    HIGH = "High"          # Substantial risk to life/property
    MEDIUM = "Medium"      # Moderate impact, stable for short term
    LOW = "Low"            # Non-emergency / minor disruption


class IncidentStatus(str, Enum):
    ACTIVE = "Active"
    ASSIGNED = "Assigned"
    CONTAINED = "Contained"
    RESOLVED = "Resolved"


class ResourceType(str, Enum):
    AMBULANCE = "Ambulance Unit (CATS / Hospital)"
    RESCUE_TEAM = "NDRF / Civil Defence Rescue Team"
    FIRE_BRIGADE = "Delhi Fire Service Unit"
    FOOD_WATER = "Emergency Food & Water Relief"
    SHELTER = "Relief Shelter & Camp"


class ResourceStatus(str, Enum):
    AVAILABLE = "Available"
    LIMITED = "Limited"
    BUSY_DEPLOYED = "Busy / Deployed"
    UNAVAILABLE = "Unavailable"


@dataclass
class DisasterIncident:
    incident_id: str
    title: str
    incident_type: IncidentType
    latitude: float
    longitude: float
    location_name: str
    severity_level: SeverityLevel
    affected_population: int
    reported_damage: str
    status: IncidentStatus
    timestamp: str
    reported_by: str = "Emergency Control Room"
    notes: Optional[str] = None

    @property
    def coordinates(self) -> tuple:
        return (self.latitude, self.longitude)


@dataclass
class EmergencyResource:
    resource_id: str
    name: str
    resource_type: ResourceType
    latitude: float
    longitude: float
    base_station: str
    capacity: int
    availability_status: ResourceStatus
    contact_number: str
    dispatch_lead_time_min: int = 5
    equipment_details: Optional[str] = None

    @property
    def coordinates(self) -> tuple:
        return (self.latitude, self.longitude)


@dataclass
class AllocationRecommendation:
    incident_id: str
    resource_id: str
    resource_name: str
    resource_type: ResourceType
    distance_km: float
    estimated_arrival_minutes: float
    priority_score: float
    recommendation_rank: int
    status: str = "Proposed"
