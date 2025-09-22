from typing import Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import stats

TZ = "Europe/Rome"
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]

def make_demo_pv_json() -> Dict[str, Any]:
    """
    Tiny per-kWp hourly PV sequence (Summer solstice morning → noon),
    timezone-aware, suitable for the 'Use sample data (demo)' button.
    """
    t = pd.date_range("2024-06-21 05:00", periods=6, freq="h", tz=TZ)
    vals = [0.0, 0.12, 0.35, 0.48, 0.42, 0.20]
    recs = [{"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S%z"),
             "energy_kWh_per_kWp": v} for ts, v in zip(t, vals)]
    return {"timezone": TZ, "unit": "kWh per kWp per hour", "records": recs}

def load_pv_json(pv_json: Dict[str,Any]) -> pd.DataFrame:
    """Load per-kWp hourly PV JSON -> DataFrame with tz-aware timestamps."""
    recs = pv_json.get("records", [])
    df = pd.DataFrame(recs)
    if "timestamp" not in df.columns or "energy_kWh_per_kWp" not in df.columns:
        raise ValueError("PV JSON must contain records with 'timestamp' and 'energy_kWh_per_kWp'.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.rename(columns={"energy_kWh_per_kWp": "kWh_per_kWp"})
    df = df.dropna(subset=["timestamp"])
    return df

def season_of(dt):
    m = dt.month
    if m in (12,1,2): return "Winter"
    if m in (3,4,5): return "Spring"
    if m in (6,7,8): return "Summer"
    return "Autumn"

def _default_envelope() -> pd.Series:
    """Return a simple bell-shaped daylight envelope over 24h normalized to sum=1."""
    h = np.arange(24)
    bump = np.exp(-0.5*((h-13)/3.5)**2)
    bump[h < 6] = 0.0
    bump[h > 20] = 0.0
    s = bump / max(bump.sum(), 1e-9)
    return pd.Series(s, index=range(24))

def _fit_loglogistic_to_unit_median(x: np.ndarray) -> Dict[str,float]:
    """Fit Fisk (log-logistic) to positive x and renormalize scale to median≈1; provide defaults if scarce."""
    v = np.asarray(x, dtype=float)
    v = v[(v>0) & np.isfinite(v)]
    if v.size < 6:
        return {"c": 2.0, "scale": 1.0}
    c, loc, scale = stats.fisk.fit(v, floc=0)  # loc=0
    med = np.median(v)
    scale_adj = scale / max(med, 1e-12)
    return {"c": float(c), "scale": float(scale_adj)}

def _markov_from_daily(daily_vals: np.ndarray) -> np.ndarray:
    """Two-state (Cloud/Clear) Markov transition from daily totals via median threshold; robust fallback."""
    vals = np.asarray(daily_vals, dtype=float)
    if vals.size < 3:
        return np.array([[0.7, 0.3],[0.3, 0.7]])
    thr = np.median(vals)
    states = (vals >= thr).astype(int)  # 1=Clear, 0=Cloud
    n00 = np.sum((states[:-1]==0)&(states[1:]==0))
    n01 = np.sum((states[:-1]==0)&(states[1:]==1))
    n10 = np.sum((states[:-1]==1)&(states[1:]==0))
    n11 = np.sum((states[:-1]==1)&(states[1:]==1))
    P = np.array([
        [n00/(n00+n01+1e-9), n01/(n00+n01+1e-9)],
        [n10/(n10+n11+1e-9), n11/(n10+n11+1e-9)]
    ])
    if not np.all(np.isfinite(P)) or np.any(P<0):
        P = np.array([[0.7, 0.3],[0.3, 0.7]])
    return P

def _beta_from_ratios(ratios: np.ndarray) -> Dict[str,float]:
    """Crude MOM Beta fit on (0,1) using rescaled ratios; fallback to alpha=beta=5."""
    x = np.asarray(ratios, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return {"alpha": 5.0, "beta": 5.0}
    m = x.mean()
    y = (x / (1.0 + max(m, 1e-9))).clip(1e-3, 1-1e-3)
    my = y.mean()
    vy = y.var()
    if vy <= 0 or my <= 0 or my >= 1:
        return {"alpha": 5.0, "beta": 5.0}
    a = my*(my*(1-my)/vy - 1)
    b = (1-my)*(my*(1-my)/vy - 1)
    if not np.isfinite(a) or not np.isfinite(b) or a<=0 or b<=0:
        return {"alpha": 5.0, "beta": 5.0}
    return {"alpha": float(a), "beta": float(b)}

def calibrate_pv(pv_json: Dict[str,Any]) -> Dict[str,Any]:
    """
    Build PV parameters for ALL seasons. For missing seasons we provide safe defaults:
    - Envelope S_s,h: default bell daylight profile (sum=1)
    - Log-logistic daily multiplier: c=2, scale=1 (median≈1)
    - 2-state Markov P: [[0.7,0.3],[0.3,0.7]]
    - Beta(alpha=5, beta=5) for clearness
    """
    df = load_pv_json(pv_json)
    df["season"] = df["timestamp"].apply(season_of)
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    # Envelope by season: average hourly per season, daylight normalized
    S_rows = {}
    for s in SEASONS:
        if s in df["season"].unique():
            prof = df[df["season"]==s].groupby("hour")["kWh_per_kWp"].mean().reindex(range(24), fill_value=0.0)
            daylight = prof.copy()
            daylight[daylight < daylight.max()*0.1] = 0.0
            total = daylight.sum()
            if total <= 0:
                S_rows[s] = _default_envelope().values
            else:
                S_rows[s] = (daylight / total).values
        else:
            S_rows[s] = _default_envelope().values
    S = pd.DataFrame(S_rows, index=range(24)).T  # seasons x 24
    S.index.name = "season"

    # Daily totals and multipliers per season (M / median)
    ll_params = {}
    markov = {}
    for s in SEASONS:
        if s in df["season"].unique():
            day = df[df["season"]==s].groupby("date")["kWh_per_kWp"].sum().sort_index()
            ll_params[s] = _fit_loglogistic_to_unit_median(day.values)
            P = _markov_from_daily(day.values)
            part = df[df["season"]==s].copy()
            env = pd.Series(S.loc[s].values, index=range(24)).replace(0, np.nan)
            part["ratio"] = part.apply(lambda r: r["kWh_per_kWp"] / env.get(r["hour"], np.nan), axis=1)
            ratios = part["ratio"].replace([np.inf, -np.inf], np.nan).dropna().values
            beta_ab = _beta_from_ratios(ratios)
            markov[s] = {"P": P.tolist(), "beta": beta_ab}
        else:
            ll_params[s] = {"c": 2.0, "scale": 1.0}
            markov[s] = {"P": [[0.7,0.3],[0.3,0.7]], "beta": {"alpha": 5.0, "beta": 5.0}}

    params = {
        "S": S,                  # DataFrame (seasons x 24)
        "loglogistic": ll_params,
        "markov": markov,
        "hash_base": {
            "S_head": S.iloc[:, :6].to_dict(),
            "ll": ll_params,
            "markov": markov
        }
    }
    return params

def render_pv_diagnostics(params):
    st.markdown("**Seasonal envelopes S_s,h (per kWp)**")
    S = params["S"]
    fig = px.line(S.T, x=S.T.index, y=S.T.columns,
                  labels={"x":"hour","value":"share"},
                  title="PV seasonal envelopes")
    st.plotly_chart(fig, width="stretch")
