from typing import Dict, Any, Tuple
import io
import numpy as np
import pandas as pd

def export_household_calibration_xlsx(params: Dict[str,Any]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xl:
        params["mu"].to_excel(xl, sheet_name="mu")
        params["sigma_lnD"].to_frame("sigma_lnD").to_excel(xl, sheet_name="lnD_sigma")
        params["sigma_resid"].to_excel(xl, sheet_name="resid_sigma")
    return buf.getvalue()

def export_shop_calibration_xlsx(params: Dict[str,Any]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xl:
        params["mu"].to_excel(xl, sheet_name="mu")
        params["p_zero"].to_excel(xl, sheet_name="p_zero")
        params["sigma_lnD"].to_frame("sigma_lnD").to_excel(xl, sheet_name="lnD_sigma")
        params["sigma_resid"].to_excel(xl, sheet_name="resid_sigma")
    return buf.getvalue()

def export_pv_calibration_xlsx(params: Dict[str,Any]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xl:
        params["S"].to_excel(xl, sheet_name="S_envelope")
        pd.DataFrame(params["loglogistic"]).to_excel(xl, sheet_name="loglogistic")
        # flatten markov
        mk = []
        for s, v in params["markov"].items():
            mk.append({"season": s, "P": v["P"], "beta_alpha": v["beta"]["alpha"], "beta_beta": v["beta"]["beta"]})
        pd.DataFrame(mk).to_excel(xl, sheet_name="markov_beta", index=False)
    return buf.getvalue()

def export_all_in_one(mc, summary, scenario, scenario_hash) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xl:
        # Summary (cards)
        summary["cards"].to_excel(xl, sheet_name="Summary", index=False)
        # KPIs — Monthly fan (per KPI, bands)
        summary["fan"].to_excel(xl, sheet_name="Monthly_Fan", index=False)
        # Metadata
        meta = pd.DataFrame([{
            "scenario_set_id": scenario_hash,
            **scenario
        }])
        meta.to_excel(xl, sheet_name="Metadata", index=False)
    return buf.getvalue()

def export_fact_tables(mc, summary) -> dict:
    out = {}
    for key in ["matched_hh","matched_shop","import_hh","import_shop","export","pros_rev","cons_cost","platform_margin"]:
        df = pd.DataFrame({"scenario_value": mc[key]})
        csv = df.to_csv(index=False).encode("utf-8")
        out[f"kpi_{key}_scenarios.csv"] = ("text/csv", csv)
    # Quantiles table from cards
    q = summary["cards"].copy()
    csv = q.to_csv(index=False).encode("utf-8")
    out["kpi_quantiles.csv"] = ("text/csv", csv)
    return out
