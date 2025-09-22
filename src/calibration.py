from typing import Dict, Any, Tuple
import io, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import stats

TZ = "Europe/Rome"

# ---------------------------
# Loaders (used when you upload Excel files)
# ---------------------------
def load_households_excel(file) -> pd.DataFrame:
    xl = pd.ExcelFile(file)
    frames = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        # Normalize columns
        tcol = [c for c in df.columns if "timestamp" in c.lower()][0]
        p_candidates = [c for c in df.columns if ("power" in c.lower()) and ("kw" in c.lower())]
        if not p_candidates:
            raise ValueError("Household sheet is missing a power column in kW.")
        pcol = p_candidates[0]
        df = df.rename(columns={tcol: "timestamp", pcol: "power_kW"})
        df["meter"] = sheet
        frames.append(df[["timestamp","power_kW","meter"]])
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out

def load_shops_excel(file) -> pd.DataFrame:
    xl = pd.ExcelFile(file)
    frames = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        tcol = [c for c in df.columns if "timestamp" in c.lower()][0]
        if any("activeenergy_generale" in c.lower() for c in df.columns):
            ecol = [c for c in df.columns if "activeenergy_generale" in c.lower()][0]
        else:
            kwh_cols = [c for c in df.columns if ("kwh" in c.lower())]
            if not kwh_cols:
                raise ValueError("Shop sheet is missing an energy column in kWh.")
            ecol = kwh_cols[0]
        df = df.rename(columns={tcol: "timestamp", ecol: "energy_kWh"})
        df["meter"] = sheet
        frames.append(df[["timestamp","energy_kWh","meter"]])
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out

# ---------------------------
# Calendar helpers (6 clusters for HH/SHOP)
# ---------------------------
def season_of(dt):
    m = dt.month
    if m in (12,1,2): return "Winter"
    if m in (3,4,5): return "Spring"
    if m in (6,7,8): return "Summer"
    return "Autumn"

def daytype_of(dt):
    wd = dt.weekday()
    if wd == 5: return "Saturday"
    if wd == 6: return "Holiday"  # Sunday as Holiday baseline
    return "Weekday"

def cluster_label(dt):
    return f"{season_of(dt)}-{daytype_of(dt)}"  # 6 clusters

# ---------------------------
# Households calibration (robust to demo headers)
# ---------------------------
def calibrate_households(raw: pd.DataFrame) -> Dict[str,Any]:
    assert raw is not None and len(raw) > 0, "No HH data"
    df = raw.copy()

    # Normalize incoming columns (handles both uploaded Excel and demo DataFrame)
    if "timestamp" not in df.columns:
        tcols = [c for c in df.columns if "timestamp" in c.lower()]
        if not tcols:
            raise ValueError("Household data has no timestamp column.")
        df = df.rename(columns={tcols[0]: "timestamp"})
    if "power_kW" not in df.columns:
        pcols = [c for c in df.columns if ("power" in c.lower()) and ("kw" in c.lower())]
        if not pcols:
            raise ValueError("Household data has no power_kW column (kW).")
        df = df.rename(columns={pcols[0]: "power_kW"})
    if "meter" not in df.columns:
        df["meter"] = "METER_001"
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # 15-min kW -> hourly kWh (by summing 4×15min)
    df["kWh_15"] = df["power_kW"] * 0.25
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["cluster"] = pd.to_datetime(df["timestamp"]).apply(cluster_label)

    # Hourly energy by meter/date/hour/cluster
    hour = df.groupby(["meter","date","hour","cluster"], as_index=False)["kWh_15"].sum()

    # Daily totals E_day by meter/date/cluster
    day = (
        hour.groupby(["meter","date","cluster"], as_index=False)["kWh_15"]
        .sum()
        .rename(columns={"kWh_15": "E_day"})
    )

    # Merge hour with E_day (validated)
    merged = hour.merge(day, on=["meter","date","cluster"], how="left", validate="many_to_one")

    # Hourly shares (positive E_day only)
    pos = merged.loc[merged["E_day"].gt(0)].copy()
    pos["share"] = pos["kWh_15"] / pos["E_day"]

    # Baseline μ_c,h = median share per cluster/hour, normalized to sum to 1
    mu = (
        pos.groupby(["cluster","hour"])["share"].median().unstack("hour").fillna(0.0)
    )
    mu = mu.div(mu.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # Day-scaler ln(D) ~ N(0, σ^2)
    day_m = day.copy()
    med = day_m.groupby("cluster")["E_day"].transform("median").replace(0, np.nan)
    eps = np.log(day_m["E_day"] / med)
    sigma_lnD = (
        eps.groupby(day_m["cluster"]).std(ddof=1)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.25)
    )

    # Hourly residual σ on positives
    mu_long = mu.reset_index().melt(id_vars="cluster", var_name="hour", value_name="mu_h")
    merged2 = merged.merge(mu_long, on=["cluster","hour"], how="left")

    # D per day
    day_m = day_m.assign(D=np.exp(np.log(day_m["E_day"] / med).fillna(0.0)))
    merged2 = merged2.merge(day_m[["meter","date","cluster","D"]], on=["meter","date","cluster"], how="left")

    merged2["base"] = merged2["D"].fillna(1.0) * merged2["mu_h"].fillna(0.0)
    mask = (merged2["E_day"].fillna(0) > 0) & (merged2["base"] > 0) & (merged2["kWh_15"] > 0)
    merged_pos = merged2.loc[mask].copy()
    merged_pos["resid"] = np.log(merged_pos["kWh_15"] / merged_pos["base"])
    sig_res = merged_pos.groupby(["cluster","hour"])["resid"].std(ddof=1).unstack("hour").fillna(0.25)

    params = {
        "mu": mu,
        "sigma_lnD": sigma_lnD,
        "sigma_resid": sig_res,
        "hash_base": {
            "mu_head": mu.head().to_dict(),
            "sigma_lnD": sigma_lnD.to_dict(),
            "sigma_resid_head": sig_res.iloc[:, :5].to_dict(),
        }
    }
    return params

def render_household_diagnostics(params: Dict[str,Any]):
    st.markdown("**Households μ profiles (per cluster)**")
    mu = params["mu"]
    fig = px.line(mu.T, x=mu.T.index, y=mu.T.columns, labels={"x":"hour","value":"μ share"}, title="μ by cluster")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Residual σ heatmap**")
    sig = params["sigma_resid"]
    fig2 = px.imshow(sig, aspect="auto", labels=dict(color="σ"), title="Hourly log-residual σ (clusters × hours)")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# Shops calibration (zero-inflated; robust headers)
# ---------------------------
def calibrate_shops(raw: pd.DataFrame) -> Dict[str,Any]:
    assert raw is not None and len(raw)>0, "No SHOP data"
    df = raw.copy()

    # Normalize incoming columns
    if "timestamp" not in df.columns:
        tcols = [c for c in df.columns if "timestamp" in c.lower()]
        if not tcols:
            raise ValueError("Shop data has no timestamp column.")
        df = df.rename(columns={tcols[0]: "timestamp"})
    if "energy_kWh" not in df.columns:
        if any("activeenergy_generale" in c.lower() for c in df.columns):
            ecol = [c for c in df.columns if "activeenergy_generale" in c.lower()][0]
        else:
            kwh_cols = [c for c in df.columns if "kwh" in c.lower()]
            if not kwh_cols:
                raise ValueError("Shop data has no kWh energy column.")
            ecol = kwh_cols[0]
        df = df.rename(columns={ecol: "energy_kWh"})
    if "meter" not in df.columns:
        df["meter"] = "SHOP_001"
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["cluster"] = pd.to_datetime(df["timestamp"]).apply(cluster_label)

    agg = df.groupby(["meter","date","hour","cluster"], as_index=False)["energy_kWh"].sum()
    day = agg.groupby(["meter","date","cluster"], as_index=False)["energy_kWh"].sum().rename(columns={"energy_kWh":"E_day"})
    merged = agg.merge(day, on=["meter","date","cluster"], how="left")

    # Zero mass per hour in each cluster
    p_zero = (merged["energy_kWh"]==0).groupby([merged["cluster"], merged["hour"]]).mean().unstack("hour").fillna(0.0)

    # Shares on positives
    pos = merged[(merged["E_day"]>0) & (merged["energy_kWh"]>0)].copy()
    pos["share"] = pos["energy_kWh"] / pos["E_day"]
    mu_tilde = pos.groupby(["cluster","hour"])["share"].median().unstack("hour").fillna(0.0)

    # Zero-aware normalization
    denom = (mu_tilde * (1 - p_zero)).sum(axis=1).replace(0, np.nan)
    mu = mu_tilde.div(denom, axis=0).fillna(0.0)

    # Day scaler ln(D) ~ N(0, σ^2)
    day_m = day.copy()
    med = day_m.groupby("cluster")["E_day"].transform("median")
    eps = np.log(np.where(day_m["E_day"]>0, day_m["E_day"], np.nan) / med)
    sigma_lnD = pd.Series(eps).groupby(day_m["cluster"]).std(ddof=1).replace([np.inf, -np.inf], np.nan).fillna(0.30)

    # Residual σ on positives
    merged = merged.merge(mu.reset_index().melt(id_vars="cluster", var_name="hour", value_name="mu").rename(columns={"mu":"mu_h"}), on=["cluster","hour"])
    day_m["D"] = np.where(day_m["E_day"]>0, day_m["E_day"]/med, 1.0)
    merged = merged.merge(day_m[["meter","date","cluster","D"]], on=["meter","date","cluster"])
    merged_pos = merged[(merged["E_day"]>0) & (merged["energy_kWh"]>0) & (merged["mu_h"]>0)].copy()
    merged_pos["resid"] = np.log(merged_pos["energy_kWh"]/(merged_pos["D"]*merged_pos["mu_h"]))
    sig_res = merged_pos.groupby(["cluster","hour"])["resid"].std(ddof=1).unstack("hour").fillna(0.30)

    params = {
        "mu": mu,
        "p_zero": p_zero,
        "sigma_lnD": sigma_lnD,
        "sigma_resid": sig_res,
        "hash_base": {
            "mu_head": mu.head().to_dict(),
            "p_zero_head": p_zero.head().to_dict(),
            "sigma_lnD": sigma_lnD.to_dict(),
            "sigma_resid_head": sig_res.iloc[:, :5].to_dict(),
        }
    }
    return params

def render_shop_diagnostics(params: Dict[str,Any]):
    st.markdown("**Shops μ profiles (zero-aware) per cluster**")
    mu = params["mu"]
    fig = px.line(mu.T, x=mu.T.index, y=mu.T.columns, labels={"x":"hour","value":"μ share"}, title="μ by cluster")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Zero mass probability p_zero(c,h)**")
    p0 = params["p_zero"]
    fig2 = px.imshow(p0, aspect="auto", labels=dict(color="p_zero"), title="Zero-mass probability")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Residual σ heatmap**")
    sig = params["sigma_resid"]
    fig3 = px.imshow(sig, aspect="auto", labels=dict(color="σ"), title="Hourly log-residual σ (clusters × hours)")
    st.plotly_chart(fig3, use_container_width=True)
