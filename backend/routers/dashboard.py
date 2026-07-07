"""
Dashboard router. Serves per-dealership metrics for the rollout simulator
and the troubleshooting flow definition for the frontend.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import get_connection
from simulator.dealerships import (
    seed_dealerships,
    run_all_dealership_evals,
    get_dashboard_metrics,
)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


class SeedRequest(BaseModel):
    manual_id: int


@router.post("/seed")
def seed_and_run(req: SeedRequest):
    """
    Seed the five simulated dealerships and run the eval harness on each.
    Safe to call once. Subsequent calls return the existing run IDs.
    """
    conn = get_connection()
    existing = conn.execute(
        "SELECT id FROM eval_runs WHERE dealership_id IS NOT NULL LIMIT 1"
    ).fetchone()
    conn.close()

    if existing:
        conn = get_connection()
        run_ids = [
            r["id"] for r in conn.execute(
                "SELECT id FROM eval_runs WHERE dealership_id IS NOT NULL"
            ).fetchall()
        ]
        conn.close()
        return {"status": "already_seeded", "run_ids": run_ids}

    dealership_ids = seed_dealerships(req.manual_id)
    run_ids = run_all_dealership_evals(req.manual_id)
    return {"status": "seeded", "dealership_ids": dealership_ids, "run_ids": run_ids}


@router.get("/metrics")
def get_metrics():
    """Retrieve computed per-dealership metrics for the dashboard."""
    conn = get_connection()
    manual = conn.execute("SELECT id FROM manuals LIMIT 1").fetchone()
    conn.close()
    if not manual:
        raise HTTPException(status_code=400, detail="No manual ingested yet.")

    metrics = get_dashboard_metrics(manual["id"])
    return {"dealerships": metrics}


@router.get("/troubleshoot-flow")
def get_troubleshoot_flow():
    """
    Return the static troubleshooting decision tree for 'unit will not start'.
    Confidence scores and page citations are grounded in the Honda E/ES3500 manual.
    """
    flow = {
        "failure_mode": "Unit Will Not Start",
        "nodes": [
            {
                "id": "root",
                "title": "Unit Will Not Start",
                "description": (
                    "The generator does not start when the engine switch is turned to START. "
                    "Use this flow to narrow down the likely cause."
                ),
                "confidence": 0.95,
                "cited_page": 16,
                "cited_text": (
                    "Troubleshooting chart for AC output circuit and starting system. "
                    "Refer to page 16."
                ),
                "question": "Does the engine crank (attempt to turn over) when you press START?",
                "branches": [
                    {"answer": "Yes, engine cranks but does not fire", "next_node_id": "cranks_no_fire"},
                    {"answer": "No, engine does not crank at all", "next_node_id": "no_crank"},
                ],
                "result": None,
            },
            {
                "id": "cranks_no_fire",
                "title": "Engine Cranks, Will Not Fire",
                "description": (
                    "The starter motor is turning the engine but combustion does not occur. "
                    "This points to a fuel delivery or ignition problem."
                ),
                "confidence": 0.88,
                "cited_page": 41,
                "cited_text": (
                    "The fuel cut-off solenoid closes the carburetor main jet when the engine "
                    "is switched OFF. A stuck or failed solenoid prevents fuel flow on restart."
                ),
                "question": "Is the engine switch in the ON position and is there fuel in the tank?",
                "branches": [
                    {"answer": "No, one of those conditions is not met", "next_node_id": "basic_checklist"},
                    {"answer": "Yes, switch is ON and fuel is present", "next_node_id": "fuel_solenoid"},
                ],
                "result": None,
            },
            {
                "id": "no_crank",
                "title": "Engine Will Not Crank",
                "description": (
                    "The starter does not engage. On ES3500 models this is typically a battery "
                    "or electrical issue. On E3500 models check the recoil starter."
                ),
                "confidence": 0.82,
                "cited_page": 17,
                "cited_text": (
                    "Maintenance schedule: battery inspection is required at every 6-month interval "
                    "or 100 hours of operation, whichever comes first."
                ),
                "question": "Is this an ES3500 model with an electric starter?",
                "branches": [
                    {"answer": "Yes, ES3500 with electric starter", "next_node_id": "battery_check"},
                    {"answer": "No, E3500 recoil start only", "next_node_id": "recoil_starter"},
                ],
                "result": None,
            },
            {
                "id": "basic_checklist",
                "title": "Basic Startup Checklist",
                "description": (
                    "Confirm the engine switch is ON, the fuel valve is open, and the fuel "
                    "level is adequate before proceeding to component diagnosis."
                ),
                "confidence": 0.97,
                "cited_page": 41,
                "cited_text": (
                    "The fuel cut-off solenoid is energized when the engine switch is at OFF, "
                    "closing the carburetor main jet. With the switch at ON the valve opens."
                ),
                "question": None,
                "branches": None,
                "result": {
                    "issue": "Operator condition not met",
                    "likely_cause": (
                        "Engine switch in wrong position or fuel tank empty."
                    ),
                    "next_diagnostic_step": (
                        "Set engine switch to ON. Confirm fuel valve is open. "
                        "Fill tank if low. Retry startup."
                    ),
                    "can_escalate": False,
                },
            },
            {
                "id": "fuel_solenoid",
                "title": "Fuel Cut-Off Solenoid",
                "description": (
                    "The fuel cut-off solenoid may be stuck closed, preventing fuel from reaching "
                    "the carburetor even when the switch is ON."
                ),
                "confidence": 0.81,
                "cited_page": 27,
                "cited_text": (
                    "Using an ohmmeter, check for continuity between the two wire leads. "
                    "Replace the solenoid valve if there is no continuity."
                ),
                "question": None,
                "branches": None,
                "result": {
                    "issue": "Fuel cut-off solenoid suspected",
                    "likely_cause": (
                        "Solenoid valve stuck closed or open circuit in solenoid wiring."
                    ),
                    "next_diagnostic_step": (
                        "Remove the wire protector tube (page 19). Disconnect solenoid connectors. "
                        "Check continuity across solenoid leads with an ohmmeter. "
                        "Replace solenoid if no continuity is found (page 27)."
                    ),
                    "can_escalate": True,
                },
            },
            {
                "id": "battery_check",
                "title": "Battery Charge Check (ES3500)",
                "description": (
                    "A discharged or failed battery will prevent the starter motor from cranking "
                    "the engine on electric-start ES3500 models."
                ),
                "confidence": 0.90,
                "cited_page": 37,
                "cited_text": (
                    "Battery tray kit components: battery, battery cable (positive), "
                    "battery cable (negative), battery set plate, guard pipe."
                ),
                "question": "Does the battery appear charged? (Check for voltage above 12V.)",
                "branches": [
                    {"answer": "Battery is discharged or unknown", "next_node_id": "battery_replace"},
                    {"answer": "Battery measures above 12V", "next_node_id": "starter_motor"},
                ],
                "result": None,
            },
            {
                "id": "recoil_starter",
                "title": "Recoil Starter (E3500)",
                "description": (
                    "On E3500 models, a seized or broken recoil starter prevents cranking. "
                    "This requires mechanical inspection."
                ),
                "confidence": 0.72,
                "cited_page": 19,
                "cited_text": (
                    "Unit removal chart: to remove the engine, remove fuel tank, wire protector "
                    "tube, 4-P coupler, belt cover, and generator adjustment components first."
                ),
                "question": None,
                "branches": None,
                "result": {
                    "issue": "Recoil starter likely seized or broken",
                    "likely_cause": (
                        "Recoil rope broken, starter pawl stuck, or engine mechanically seized."
                    ),
                    "next_diagnostic_step": (
                        "Inspect the recoil starter assembly for broken rope or stuck pawl. "
                        "Attempt to rotate engine by hand. If engine is seized, escalate to "
                        "an authorized service center."
                    ),
                    "can_escalate": True,
                },
            },
            {
                "id": "battery_replace",
                "title": "Charge or Replace Battery",
                "description": (
                    "A discharged battery must be charged or replaced before further diagnosis. "
                    "Refer to the maintenance schedule for battery service intervals."
                ),
                "confidence": 0.94,
                "cited_page": 17,
                "cited_text": (
                    "Maintenance schedule: inspect and service the battery every 6 months "
                    "or 100 hours. Use only genuine Honda-recommended parts."
                ),
                "question": None,
                "branches": None,
                "result": {
                    "issue": "Battery discharged or failed",
                    "likely_cause": (
                        "Battery below minimum voltage. May need charging or replacement."
                    ),
                    "next_diagnostic_step": (
                        "Charge battery to 12.6V minimum. If battery will not hold charge, "
                        "replace with Honda-approved battery per maintenance schedule (page 17). "
                        "After charging, retry engine start."
                    ),
                    "can_escalate": False,
                },
            },
            {
                "id": "starter_motor",
                "title": "Starter Motor and Wiring",
                "description": (
                    "Battery is charged but engine still will not crank. Check the starter "
                    "motor terminals, the R1 and R2 rectifiers, and wiring connections."
                ),
                "confidence": 0.75,
                "cited_page": 26,
                "cited_text": (
                    "R1 (for non-ES3500) and R2 (for ES3500 K1 starting circuit) rectifiers: "
                    "disconnect coupler from rectifier, check continuity with ohmmeter. "
                    "Replace if continuity does not match the chart."
                ),
                "question": None,
                "branches": None,
                "result": {
                    "issue": "Starter motor circuit fault",
                    "likely_cause": (
                        "Failed R1 or R2 rectifier, broken starter terminal, or wiring fault."
                    ),
                    "next_diagnostic_step": (
                        "Remove control box cover (6 screws). Disconnect R2 coupler. "
                        "Test continuity per diagram on page 26. "
                        "Inspect starter motor terminals for corrosion or looseness."
                    ),
                    "can_escalate": True,
                },
            },
        ],
    }
    return flow
