from typing import Dict, Any
import numpy as np
import pandas as pd

def compute_diagnostics(mc: Dict[str,Any], scenario: Dict[str,Any]) -> Dict[str,Any]:
    # Minimal placeholder diagnostics at community level.
    # In a full pathwise engine, check identities per scenario and per hour.
    violations = []
    # Example: matched + export should be <= total PV generation (not tracked in this minimal demo)
    # We'll just report negatives if any kpi distributions have invalid (negative) values.
    for k in ["matched_hh","matched_shop","import_hh","import_shop","export","pros_rev","cons_cost","platform_margin"]:
        x = mc[k]
        if np.any(x < -1e-9):
            violations.append({"kpi": k, "min": float(x.min())})
    viol = pd.DataFrame(violations)
    summary = pd.DataFrame([{"checks": "basic_nonneg", "failures": len(viol)}])
    return {"summary": summary, "violations": viol}
