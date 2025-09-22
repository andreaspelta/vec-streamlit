from typing import Dict, Any
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import stats

TZ = "Europe/Rome"

def load_pv_json(pv_json: Dict[str,Any]) -> pd.DataFrame:
    recs = pv_json.get("records", [])
    df = pd.DataFrame(recs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.rename(columns={"energy_kWh_per_kWp":"kWh_per_kWp"})
    return df

def season_of(dt):
    m = dt.month
    if m in (12,1,2): return "Winter"
    if m in (3,4,5): return "Spring"
    if m in (6,7,8): return "Summer"
    return "Autumn"

def calibrate_pv(pv_json: Dict[str,Any]) -> Dict[str,Any]:
    df = load_pv_json(pv_json)
    df["season"] = df["timestamp"].apply(season_of)
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    # Seasonal envelope S_s,h: average profile per season normalized to sum to 1 on daylight hours
    prof = df.groupby(["season","hour"])["kWh_per_kWp"].mean().unstack("hour").fillna(0.0)
    # Zero out nighttime (values very small)
    prof = prof.mask(prof < prof.max().quantile(0.1), 0.0)
    S = prof.div(prof.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # Daily totals and M_t (log-logistic with median ~1)
    day = df.groupby(["season","date"])["kWh_per_kWp"].sum().reset_index()
    M_raw = day.groupby("season")["kWh_per_kWp"].apply(lambda x: x / x.median())
    # Fit log-logistic via scipy.fisk (shape c, scale) per season
    ll_params = {}
    for s, x in M_raw.groupby(level=0):
        v = x.values
        v = v[(v>0) & np.isfinite(v)]
        if len(v) < 5:
            ll_params[s] = {"c": 2.0, "scale": 1.0}
        else:
            # Fix location=0; estimate c, loc, scale; force loc=0
            c, loc, scale = stats.fisk.fit(v, floc=0)
            # normalize scale so that median ≈ 1 (median of Fisk = scale * (1)^(1/c) = scale)
            ll_params[s] = {"c": float(c), "scale": float(scale / np.median(v))}
    # Markov day-state (two-state Clear/Cloud) by threshold on daily total
    markov = {}
    for s, grp in day.groupby("season"):
        vals = grp["kWh_per_kWp"].values
        thr = np.median(vals)
        states = (vals >= thr).astype(int)  # 1=Clear, 0=Cloud
        if len(states) < 3:
            P = np.array([[0.7,0.3],[0.3,0.7]])
        else:
            n00 = np.sum((states[:-1]==0)&(states[1:]==0))
            n01 = np.sum((states[:-1]==0)&(states[1:]==1))
            n10 = np.sum((states[:-1]==1)&(states[1:]==0))
            n11 = np.sum((states[:-1]==1)&(states[1:]==1))
            P = np.array([
                [n00/(n00+n01+1e-9), n01/(n00+n01+1e-9)],
                [n10/(n10+n11+1e-9), n11/(n10+n11+1e-9)]
            ])
        # Beta clearness per hour/state: derive alpha,beta from moments on (hourly / envelope share)
        # Approximation: use same beta params for both states per season
        cl = df[df["season"]==s].copy()
        env = S.loc[s].replace(0, np.nan)
        cl["ratio"] = cl.apply(lambda r: r["kWh_per_kWp"] / (env.get(r["hour"], np.nan)), axis=1)
        x = cl["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        x = x[(x>0) & (x<5)]  # crude trimming
        m = x.mean() if len(x)>0 else 1.0
        v = x.var() if len(x)>1 else 0.1
        # MOM for Beta on (0,1) -> rescale ratio to (0,1) by / (1+m)
        y = (x / (1+m)).clip(1e-3, 1-1e-3)
        my = y.mean() if len(y)>0 else 0.5
        vy = y.var() if len(y)>1 else 0.05
        # alpha,beta from mean/var
        if vy<=0 or my<=0 or my>=1:
            a,b = 5.0,5.0
        else:
            a = my*(my*(1-my)/vy - 1)
            b = (1-my)*(my*(1-my)/vy - 1)
            if not np.isfinite(a) or not np.isfinite(b) or a<=0 or b<=0:
                a,b = 5.0,5.0
        markov[s] = {"P": P.tolist(), "beta": {"alpha": float(a), "beta": float(b)}}

    params = {"S": S, "loglogistic": ll_params, "markov": markov,
              "hash_base": {"S_head": S.iloc[:,:6].to_dict(), "ll": ll_params, "markov": markov}}
    return params

def render_pv_diagnostics(params):
    st.markdown("**Seasonal envelopes S_s,h**")
    S = params["S"]
    fig = px.line(S.T, x=S.T.index, y=S.T.columns, labels={"x":"hour","value":"share"}, title="PV seasonal envelopes (per kWp)")
    st.plotly_chart(fig, use_container_width=True)

def make_demo_pv_json():
    # Reuse in utils; keeping alias for clarity
    from src.utils import make_demo_pv_json as demo
    return demo()
