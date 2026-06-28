"""Load sample data into a tenant store and a RAG engine.

Used both by the standalone seed scripts (against real Mongo/Chroma) and by
the API at startup (against the in-memory fallbacks) so a fresh clone has a
fully populated, queryable system with no manual steps.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from sdk.models import Tenant
from sdk.rag_engine import Document, RAGEngine
from sdk.tenant_store import TenantIsolatedStore

SAMPLE_DIR = Path(__file__).parent / "sample_data"

# Collections
MACHINES = "machines"
SENSOR_LOGS = "sensor_logs"
FAILURE_CASES = "failure_cases"
SPARE_PARTS = "spare_parts"

# RAG collections
KB_STANDARDS = "vibration_standards"
KB_MANUALS = "manuals"
KB_PROCEDURES = "maintenance_procedures"


def _load(name: str) -> Any:
    return json.loads((SAMPLE_DIR / name).read_text(encoding="utf-8"))


def generate_sensor_logs(machines: list[dict], readings_per_machine: int = 100, seed: int = 7):
    """Deterministically synthesize a week of readings with injected anomalies."""
    rng = random.Random(seed)
    logs: list[dict] = []
    for m in machines:
        ranges = m["normal_ranges"]
        # Last machine of each type trends into an anomaly toward the end.
        for i in range(readings_per_machine):
            frac = i / readings_per_machine
            anomalous = frac > 0.85 and m["machine_id"] in {"pump_001", "gearbox_005"}
            vib_lo, vib_hi = ranges["vibration_mm_s"]
            base_vib = rng.uniform(vib_lo, vib_hi)
            if anomalous:
                base_vib = vib_hi * (1.4 + frac)  # drift well past the limit
            t_lo, t_hi = ranges["temperature_c"]
            temp = rng.uniform(t_lo, t_hi) + (12 if anomalous else 0)
            p_lo, p_hi = ranges["pressure_bar"]
            pressure = rng.uniform(p_lo, p_hi) if p_hi > 0 else 0.0
            r_lo, r_hi = ranges["rpm"]
            logs.append(
                {
                    "machine_id": m["machine_id"],
                    "timestamp": f"2026-06-{20 + i // 24:02d}T{i % 24:02d}:00:00Z",
                    "vibration_mm_s": round(base_vib, 2),
                    "temperature_c": round(temp, 1),
                    "pressure_bar": round(pressure, 2),
                    "rpm": rng.randint(r_lo, r_hi),
                }
            )
    return logs


def seed_store(store: TenantIsolatedStore, tenant: Tenant) -> dict[str, int]:
    """Populate the document store for one tenant. Returns inserted counts."""
    machines = _load("machines.json")
    failures = _load("failure_cases.json")
    kb = _load("knowledge_base.json")
    logs = generate_sensor_logs(machines)

    for m in machines:
        store.insert(MACHINES, m, tenant)
    for log in logs:
        store.insert(SENSOR_LOGS, log, tenant)
    for fc in failures:
        store.insert(FAILURE_CASES, fc, tenant)
    for part in kb["spare_parts"]:
        store.insert(SPARE_PARTS, part, tenant)

    return {
        MACHINES: len(machines),
        SENSOR_LOGS: len(logs),
        FAILURE_CASES: len(failures),
        SPARE_PARTS: len(kb["spare_parts"]),
    }


def seed_rag(rag: RAGEngine) -> dict[str, int]:
    """Index the knowledge base into the RAG engine. Returns indexed counts."""
    kb = _load("knowledge_base.json")
    failures = _load("failure_cases.json")
    counts: dict[str, int] = {}

    counts[KB_STANDARDS] = rag.index_documents(
        [Document(text=d["text"], source=d["source"]) for d in kb["vibration_standards"]],
        KB_STANDARDS,
    )
    counts[KB_MANUALS] = rag.index_documents(
        [
            Document(text=d["text"], source=d["source"], metadata={"machine_type": d["machine_type"]})
            for d in kb["manuals"]
        ],
        KB_MANUALS,
    )
    # Procedures + failure cases share a collection used by diagnostics/recommender.
    proc_docs = [
        Document(text=d["text"], source=d["source"], metadata={"fault": d["fault"]})
        for d in kb["maintenance_procedures"]
    ]
    fc_docs = [
        Document(
            text=(
                f"{fc['machine_type']} symptoms: {fc['symptoms']}. "
                f"Root cause: {fc['root_cause']}. Resolution: {fc['resolution']}. "
                f"Downtime {fc['downtime_hours']}h. Parts: {', '.join(fc['parts']) or 'none'}."
            ),
            source=f"failure_case:{fc['case_id']}",
            metadata={"machine_type": fc["machine_type"], "root_cause": fc["root_cause"]},
        )
        for fc in failures
    ]
    counts[KB_PROCEDURES] = rag.index_documents(proc_docs + fc_docs, KB_PROCEDURES)
    return counts
