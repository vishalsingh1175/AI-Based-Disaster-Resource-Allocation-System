"""
Seed dataset generator for Delhi-NCR disaster incidents and emergency resources.
Generates realistic spatial coordinates, incident attributes, and resource capacities.
"""

import os
import pandas as pd

def generate_datasets(output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)

    incidents = [
        {
            "incident_id": "INC-DEL-001",
            "title": "Yamuna River Khadar Lowland Flooding",
            "incident_type": "Flood",
            "latitude": 28.6652,
            "longitude": 77.2341,
            "location_name": "Old Railway Bridge / Yamuna Khadar",
            "severity_level": "Critical",
            "affected_population": 3400,
            "reported_damage": "Submerged shanties, road embankment breached, residents stranded on rooftops",
            "status": "Active",
            "timestamp": "2026-09-28 08:30:00",
            "reported_by": "Central Water Commission & DDMA",
            "notes": "Urgent boat rescue and medical evacuation needed"
        },
        {
            "incident_id": "INC-DEL-002",
            "title": "Industrial Warehouse Fire & Chemical Smoke",
            "incident_type": "Fire / Chemical Hazard",
            "latitude": 28.6812,
            "longitude": 77.0321,
            "location_name": "Mundka Industrial Area Phase-I",
            "severity_level": "Critical",
            "affected_population": 450,
            "reported_damage": "Multi-storey plastics manufacturing godown engulfed, toxic fumes spreading",
            "status": "Active",
            "timestamp": "2026-09-28 09:15:00",
            "reported_by": "Delhi Police PCR",
            "notes": "Explosion hazard. Requires foam tenders and hazmat breathing kits"
        },
        {
            "incident_id": "INC-DEL-003",
            "title": "Historic Residential Building Collapse",
            "incident_type": "Structural Collapse",
            "latitude": 28.6562,
            "longitude": 77.2310,
            "location_name": "Chandni Chowk / Ballimaran",
            "severity_level": "High",
            "affected_population": 120,
            "reported_damage": "3-storey heritage building partially collapsed into narrow lane",
            "status": "Active",
            "timestamp": "2026-09-28 10:05:00",
            "reported_by": "Local Residents / Beat Constable",
            "notes": "Narrow alleyway restricts large fire engines; specialized canine & acoustic search needed"
        },
        {
            "incident_id": "INC-DEL-004",
            "title": "Commercial Plaza Basement Waterlogging & Electrical Short",
            "incident_type": "Flood",
            "latitude": 28.6740,
            "longitude": 77.1085,
            "location_name": "Paschim Vihar District Centre",
            "severity_level": "Medium",
            "affected_population": 85,
            "reported_damage": "Basement parking submerged under 4 feet water, transformer sparks reported",
            "status": "Active",
            "timestamp": "2026-09-28 11:20:00",
            "reported_by": "Security Control Room",
            "notes": "Power isolated; de-watering high-capacity pumps required"
        },
        {
            "incident_id": "INC-DEL-005",
            "title": "Multi-Storey Apartment Kitchen Gas Explosion",
            "incident_type": "Fire / Chemical Hazard",
            "latitude": 28.7180,
            "longitude": 77.1215,
            "location_name": "Rohini Sector 18 DDA Flats",
            "severity_level": "High",
            "affected_population": 210,
            "reported_damage": "Floors 3 and 4 damaged, fire spreading to external AC ducts",
            "status": "Active",
            "timestamp": "2026-09-28 12:45:00",
            "reported_by": "Resident Welfare Association",
            "notes": "Hydraulic ladder and burn unit ambulances dispatched"
        },
        {
            "incident_id": "INC-DEL-006",
            "title": "Drain Overflow & Urban Waterlogging",
            "incident_type": "Flood",
            "latitude": 28.5822,
            "longitude": 77.0611,
            "location_name": "Dwarka Sector 14 Underpass",
            "severity_level": "Low",
            "affected_population": 30,
            "reported_damage": "Underpass completely flooded, 3 vehicles stalled",
            "status": "Active",
            "timestamp": "2026-09-28 13:10:00",
            "reported_by": "Traffic Police Control Room",
            "notes": "No casualties; traffic diversions activated"
        },
        {
            "incident_id": "INC-DEL-007",
            "title": "Underground Metro Pedestrian Subway Water Seepage",
            "incident_type": "Structural Collapse",
            "latitude": 28.6315,
            "longitude": 77.2167,
            "location_name": "Connaught Place Radial Road 3",
            "severity_level": "Medium",
            "affected_population": 180,
            "reported_damage": "Ceiling plaster falling, heavy water seepage endangering commuters",
            "status": "Active",
            "timestamp": "2026-09-28 14:00:00",
            "reported_by": "DMRC Station Controller",
            "notes": "Subway cordoned off; structural inspection team requested"
        },
        {
            "incident_id": "INC-DEL-008",
            "title": "Contaminated Water Supply Gastro Outbreak",
            "incident_type": "Medical Outbreak",
            "latitude": 28.7250,
            "longitude": 77.1720,
            "location_name": "Jahangirpuri Block EE",
            "severity_level": "High",
            "affected_population": 520,
            "reported_damage": "Over 70 patients hospitalized within 6 hours, pipeline leak identified",
            "status": "Active",
            "timestamp": "2026-09-28 14:40:00",
            "reported_by": "Chief Medical Officer (North)",
            "notes": "Mobile medical camps, clean drinking water tankers required urgently"
        },
        {
            "incident_id": "INC-DEL-009",
            "title": "Interstate Bus Terminal Severe Heatstroke & Stampede",
            "incident_type": "Medical Outbreak",
            "latitude": 28.6495,
            "longitude": 77.3155,
            "location_name": "Anand Vihar ISBT & Railway Staging Area",
            "severity_level": "Medium",
            "affected_population": 350,
            "reported_damage": "Crowd surge on foot-over bridge, dehydration and faintings",
            "status": "Active",
            "timestamp": "2026-09-28 15:15:00",
            "reported_by": "ISBT Terminal Supervisor",
            "notes": "Mobile rehydration squads and quick-response ambulances deployed"
        },
        {
            "incident_id": "INC-DEL-010",
            "title": "Transformer Fire & Packaging Unit Blaze",
            "incident_type": "Fire / Chemical Hazard",
            "latitude": 28.5355,
            "longitude": 77.2711,
            "location_name": "Okhla Industrial Area Phase-II",
            "severity_level": "High",
            "affected_population": 190,
            "reported_damage": "Cardboard packaging unit on fire threatening neighboring garment factory",
            "status": "Active",
            "timestamp": "2026-09-28 15:50:00",
            "reported_by": "Okhla Fire Post",
            "notes": "Cooling operations underway; 4 fire tenders active"
        }
    ]

    resources = [
        {
            "resource_id": "RES-AMB-001",
            "name": "CATS Advanced Life Support Unit #12",
            "resource_type": "Ambulance Unit (CATS / Hospital)",
            "latitude": 28.5672,
            "longitude": 77.2100,
            "base_station": "AIIMS Trauma Centre, Ansari Nagar",
            "capacity": 4,
            "availability_status": "Available",
            "contact_number": "+91-11-2659-3333",
            "dispatch_lead_time_min": 4,
            "equipment_details": "Ventilator, Defibrillator, Emergency Trauma Meds"
        },
        {
            "resource_id": "RES-AMB-002",
            "name": "DDU Hospital Rapid Ambulance Fleet A",
            "resource_type": "Ambulance Unit (CATS / Hospital)",
            "latitude": 28.6278,
            "longitude": 77.1065,
            "base_station": "Deen Dayal Upadhyay Hospital, Hari Nagar",
            "capacity": 6,
            "availability_status": "Available",
            "contact_number": "+91-11-2549-4400",
            "dispatch_lead_time_min": 3,
            "equipment_details": "BLS Kits, Oxygen Cylinders, 2 Paramedics"
        },
        {
            "resource_id": "RES-AMB-003",
            "name": "Sanjay Gandhi Memorial Ambulance Squad",
            "resource_type": "Ambulance Unit (CATS / Hospital)",
            "latitude": 28.6945,
            "longitude": 77.0988,
            "base_station": "Sanjay Gandhi Memorial Hospital, Mangolpuri",
            "capacity": 5,
            "availability_status": "Available",
            "contact_number": "+91-11-2792-2500",
            "dispatch_lead_time_min": 5,
            "equipment_details": "Trauma Stabilizers, Stretcher Hoists"
        },
        {
            "resource_id": "RES-AMB-004",
            "name": "LNJP Emergency Ambulance Unit #04",
            "resource_type": "Ambulance Unit (CATS / Hospital)",
            "latitude": 28.6366,
            "longitude": 77.2415,
            "base_station": "Lok Nayak Hospital, Delhi Gate",
            "capacity": 3,
            "availability_status": "Limited",
            "contact_number": "+91-11-2323-3000",
            "dispatch_lead_time_min": 7,
            "equipment_details": "Burn Dressing Kits, ICU On Wheels"
        },
        {
            "resource_id": "RES-AMB-005",
            "name": "Max Healthcare Emergency Support #07",
            "resource_type": "Ambulance Unit (CATS / Hospital)",
            "latitude": 28.7011,
            "longitude": 77.1550,
            "base_station": "Max Hospital, Shalimar Bagh",
            "capacity": 4,
            "availability_status": "Busy / Deployed",
            "contact_number": "+91-11-6642-2222",
            "dispatch_lead_time_min": 15,
            "equipment_details": "Cardiac Care, Advanced Ventilator"
        },
        {
            "resource_id": "RES-FIRE-001",
            "name": "DFS Paschim Vihar Station Unit 1",
            "resource_type": "Delhi Fire Service Unit",
            "latitude": 28.6698,
            "longitude": 77.1120,
            "base_station": "Paschim Vihar Fire Station (Jwalaheri)",
            "capacity": 8,
            "availability_status": "Available",
            "contact_number": "+91-11-2525-0101",
            "dispatch_lead_time_min": 3,
            "equipment_details": "Water Tender (4500L), High Pressure Hose, Cutting Tools"
        },
        {
            "resource_id": "RES-FIRE-002",
            "name": "DFS Connaught Place Headquarters Tender",
            "resource_type": "Delhi Fire Service Unit",
            "latitude": 28.6304,
            "longitude": 77.2195,
            "base_station": "CP Fire Station, Barakhamba Road",
            "capacity": 12,
            "availability_status": "Available",
            "contact_number": "+91-11-2341-2222",
            "dispatch_lead_time_min": 4,
            "equipment_details": "54m Hydraulic Turntable Ladder, Foam Tender"
        },
        {
            "resource_id": "RES-FIRE-003",
            "name": "DFS Rohini Sector 5 Quick Response Tender",
            "resource_type": "Delhi Fire Service Unit",
            "latitude": 28.7125,
            "longitude": 77.1150,
            "base_station": "Rohini Fire Station",
            "capacity": 6,
            "availability_status": "Available",
            "contact_number": "+91-11-2704-0101",
            "dispatch_lead_time_min": 4,
            "equipment_details": "Multi-Purpose Tender, Thermal Imaging Camera"
        },
        {
            "resource_id": "RES-FIRE-004",
            "name": "DFS Okhla Industrial Heavy Tender",
            "resource_type": "Delhi Fire Service Unit",
            "latitude": 28.5312,
            "longitude": 77.2750,
            "base_station": "Okhla Phase-I Fire Station",
            "capacity": 10,
            "availability_status": "Limited",
            "contact_number": "+91-11-2681-1101",
            "dispatch_lead_time_min": 6,
            "equipment_details": "Chemical Foam Crash Tender, Breathing Apparatus Bank"
        },
        {
            "resource_id": "RES-NDRF-001",
            "name": "NDRF 8th Bn RRC Flood Rescue Troop",
            "resource_type": "NDRF / Civil Defence Rescue Team",
            "latitude": 28.5833,
            "longitude": 77.0500,
            "base_station": "NDRF Regional Response Centre, Sector 8 Dwarka",
            "capacity": 45,
            "availability_status": "Available",
            "contact_number": "+91-11-2805-0100",
            "dispatch_lead_time_min": 8,
            "equipment_details": "12 Inflatable Motor Boats (IRB), Lifebuoys, Deep Divers"
        },
        {
            "resource_id": "RES-NDRF-002",
            "name": "Delhi Civil Defence West Rescue Battalion",
            "resource_type": "NDRF / Civil Defence Rescue Team",
            "latitude": 28.6540,
            "longitude": 77.1230,
            "base_station": "District Magistrate Office, Raja Garden",
            "capacity": 30,
            "availability_status": "Available",
            "contact_number": "+91-11-2510-1234",
            "dispatch_lead_time_min": 5,
            "equipment_details": "Hydraulic Spreaders, Chainsaws, Acoustic Search Detectors"
        },
        {
            "resource_id": "RES-NDRF-003",
            "name": "NDRF Yamuna Flood Special Task Force",
            "resource_type": "NDRF / Civil Defence Rescue Team",
            "latitude": 28.6720,
            "longitude": 77.2280,
            "base_station": "Civil Lines Disaster Staging Post",
            "capacity": 50,
            "availability_status": "Available",
            "contact_number": "+91-11-2391-4455",
            "dispatch_lead_time_min": 5,
            "equipment_details": "Assault Boats, High-Volume Water Evacuation Pumps"
        },
        {
            "resource_id": "RES-REL-001",
            "name": "DDMA Central Food & Potable Water Depot",
            "resource_type": "Emergency Food & Water Relief",
            "latitude": 28.6410,
            "longitude": 77.1850,
            "base_station": "Pusa Institute Agriculture Campus Staging Hub",
            "capacity": 5000,
            "availability_status": "Available",
            "contact_number": "+91-11-2584-7700",
            "dispatch_lead_time_min": 10,
            "equipment_details": "10 Potable Water Tankers (5000L each), 10,000 Packed Meal Rations"
        },
        {
            "resource_id": "RES-REL-002",
            "name": "DDMA East Delhi Relief Supplies Depot",
            "resource_type": "Emergency Food & Water Relief",
            "latitude": 28.6250,
            "longitude": 77.3010,
            "base_station": "Yamuna Sports Complex Hub, Surajmal Vihar",
            "capacity": 3500,
            "availability_status": "Available",
            "contact_number": "+91-11-2216-9900",
            "dispatch_lead_time_min": 10,
            "equipment_details": "Dry Rations, Chlorine Tablets, Baby Food Kits"
        },
        {
            "resource_id": "RES-SHL-001",
            "name": "DUSIB Emergency Relief Camp #1",
            "resource_type": "Relief Shelter & Camp",
            "latitude": 28.6670,
            "longitude": 77.2250,
            "base_station": "Kashmere Gate ISBT Ground",
            "capacity": 1200,
            "availability_status": "Available",
            "contact_number": "+91-11-2386-5050",
            "dispatch_lead_time_min": 15,
            "equipment_details": "Waterproof Weather Tents, 800 Cots, Sanitation Blocks"
        },
        {
            "resource_id": "RES-SHL-002",
            "name": "Janakpuri Community Disaster Shelter",
            "resource_type": "Relief Shelter & Camp",
            "latitude": 28.6220,
            "longitude": 77.0850,
            "base_station": "Janakpuri B-Block Complex",
            "capacity": 800,
            "availability_status": "Available",
            "contact_number": "+91-11-2555-8888",
            "dispatch_lead_time_min": 15,
            "equipment_details": "Air-Cooled Hall, Backup Generator, Community Kitchen"
        }
    ]

    incidents_df = pd.DataFrame(incidents)
    resources_df = pd.DataFrame(resources)

    incidents_path = os.path.join(output_dir, "delhi_incidents.csv")
    resources_path = os.path.join(output_dir, "delhi_resources.csv")

    incidents_df.to_csv(incidents_path, index=False)
    resources_df.to_csv(resources_path, index=False)

    print(f"Generated {len(incidents_df)} incidents -> {incidents_path}")
    print(f"Generated {len(resources_df)} resources -> {resources_path}")

if __name__ == "__main__":
    generate_datasets()
