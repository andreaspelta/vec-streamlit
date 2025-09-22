from typing import Dict, Any
import numpy as np
import pandas as pd
import plotly.express as px

def _pctiles(x):
    return {
        "p05": float(np.percentile(x, 5)),
        "p10": float(np.percentile(x,10)),
        "p50": float(np.percentile(x,50)),
        "p90": float(np.percentile(x,90)),
        "p95": float(np.percentile(x,95)),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if len(x)>1 else 0.0,
    }

def summarize_kpis(mc: Dict[str,Any], scenario: Dict[str,Any]) -> Dict[str,Any]:
    cards = []
    for name, label in [
        ("matched_hh","HH matched (kWh)"),
        ("matched_shop","SHOP matched (kWh)"),
        ("import_hh","HH imports (kWh)"),
        ("import_shop","SHOP imports (kWh)"),
        ("export","PV exports (kWh)"),
        ("pros_rev","Prosumer revenue (€)"),
        ("cons_cost","Consumer total spend (€)"),
        ("platform_margin","Platform margin (€)")
    ]:
        stats = _pctiles(mc[name])
        row = {"KPI": label, **stats}
        cards.append(row)
    cards_df = pd.DataFrame(cards)

    hist_options = list(cards_df["KPI"].unique())

    # Monthly fan charts — mock monthly sum distribution, using equal split over 12
    # In a full pathwise engine, compute per-scenario per-month values and aggregate quantiles per month.
    months = [f"{m:02d}" for m in range(1,13)]
    fan = pd.DataFrame({"month": months})
    for k in hist_options:
        # synthesize bands from annual totals for demo purposes
        fan[f"{k} p10"] = np.linspace(0.8,1.2,12) * cards_df.loc[cards_df["KPI"]==k,"p10"].values[0]/12.0
        fan[f"{k} p50"] = np.linspace(0.8,1.2,12) * cards_df.loc[cards_df["KPI"]==k,"p50"].values[0]/12.0
        fan[f"{k} p90"] = np.linspace(0.8,1.2,12) * cards_df.loc[cards_df["KPI"]==k,"p90"].values[0]/12.0

    return {"cards": cards_df, "hist_options": hist_options, "fan": fan}

def hist_violin(summary, kpi_name):
    cards = summary["cards"]
    # Build synthetic sample for plotting hist/violin from pcts (for demo)
    row = cards.loc[cards["KPI"]==kpi_name].iloc[0]
    # Just plot five points as stems
    df = pd.DataFrame({
        "stat":["p05","p10","p50","p90","p95"],
        "value":[row["p05"],row["p10"],row["p50"],row["p90"],row["p95"]]
    })
    fig = px.bar(df, x="stat", y="value", title=f"Distribution summary — {kpi_name}")
    return fig

def monthly_fan_chart(summary, kpi_name):
    fan = summary["fan"]
    df = fan[["month", f"{kpi_name} p10", f"{kpi_name} p50", f"{kpi_name} p90"]].rename(
        columns={f"{kpi_name} p10":"p10", f"{kpi_name} p50":"p50", f"{kpi_name} p90":"p90"}
    )
    fig = px.line(df, x="month", y=["p10","p50","p90"], title=f"Monthly fan — {kpi_name}")
    return fig
