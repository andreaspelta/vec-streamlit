import io
import json
import hashlib
from typing import Dict, Any, Tuple, List
from datetime import datetime
import numpy as np
import pandas as pd
import pytz

TZ = "Europe/Rome"

# -------------------------
# Demo data (small, in-memory)
# -------------------------
def make_demo_loads():
    """Return tiny HH and SHOP dataframes with the required columns."""
    t = pd.date_range("2024-06-21 00:00", periods=8, freq="15min", tz=TZ)
    hh = pd.DataFrame({
        "timestamp (Europe/Rome, ISO8601)": t.strftime("%Y-%m-%d %H:%M:%S%z"),
        "power_kW (15min average)": [0.8,0.7,0.6,0.5,0.4,0.6,0.9,1.2],
        "meter": "METER_001"
    })
    t2 = pd.date_range("2024-06-21 08:00", periods=8, freq="15min", tz=TZ)
    shop = pd.DataFrame({
        "timestamp (Europe/Rome, ISO8601)": t2.strftime("%Y-%m-%d %H:%M:%S%z"),
        "ActiveEnergy_Generale (kWh per 15min)": [0.0,0.0,0.5,0.7,0.8,0.9,0.6,0.4],
        "meter": "SHOP_001"
    })
    return hh, shop

def make_demo_pv_json():
    t = pd.date_range("2024-06-21 05:00", periods=6, freq="h", tz=TZ)  # 'h' to avoid deprecation
    recs = []
    vals = [0.0, 0.12, 0.35, 0.48, 0.42, 0.20]
    for ts, v in zip(t, vals):
        recs.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S%z"),
            "energy_kWh_per_kWp": v
        })
    return {"timezone": TZ, "unit": "kWh per kWp per hour", "records": recs}

def make_demo_prices():
    """Demo prices aligned to Summer (to match demo PV)."""
    hours = pd.date_range("2024-06-21", periods=24, freq="h", tz=TZ)
    pun = pd.DataFrame({"timestamp": hours, "PUN (EUR_per_MWh)": np.linspace(80, 120, len(hours))})
    zones = ["NORD","CNOR","CSUD","SUD","SICI","SARD"]
    rows = []
    for ts in hours:
        for i, z in enumerate(zones):
            rows.append({"timestamp": ts, "zone": z, "zonal_price (EUR_per_MWh)": 90 + i*2 + (ts.hour%3)*1.5})
    zonal = pd.DataFrame(rows)
    pun["timestamp"] = pun["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    zonal["timestamp"] = zonal["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return {"pun": pun, "zonal": zonal}

def make_demo_mapping():
    return pd.DataFrame({"prosumer_id":["P001","P001","P002"], "household_id":["H011","H012","H021"], "zone":["NORD","NORD","CNOR"]})

# -------------------------
# Prices loader
# -------------------------
def load_prices(pun_csv, zonal_csv):
    pun = pd.read_csv(pun_csv)
    z = pd.read_csv(zonal_csv)
    tcol_pun = [c for c in pun.columns if "timestamp" in c.lower()][0]
    pun = pun.rename(columns={tcol_pun: "timestamp"})
    pun["timestamp"] = pd.to_datetime(pun["timestamp"])
    tcol_z = [c for c in z.columns if "timestamp" in c.lower()][0]
    z = z.rename(columns={tcol_z: "timestamp"})
    if "zone" in [c.lower() for c in z.columns]:
        z["timestamp"] = pd.to_datetime(z["timestamp"])
        zonal = z.copy()
    else:
        z["timestamp"] = pd.to_datetime(z["timestamp"])
        zonal = z.melt(id_vars=["timestamp"], var_name="zone", value_name="zonal_price (EUR_per_MWh)")
        zonal["zone"] = zonal["zone"].str.replace(r"\s*\(.*\)", "", regex=True)
    pun["timestamp"] = pun["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    zonal["timestamp"] = zonal["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return {"pun": pun, "zonal": zonal}

# -------------------------
# Scenario hash for reproducibility
# -------------------------
def scenario_hash(hh_params, shop_params, pv_params, prices, retail_setup, mapping, scenario) -> str:
    payload = {
        "hh": hh_params.get("hash_base", None),
        "shop": shop_params.get("hash_base", None),
        "pv": pv_params.get("hash_base", None),
        "prices": {"pun_head": prices["pun"].head().to_dict(), "zonal_head": prices["zonal"].head().to_dict()},
        "retail": retail_setup,
        "mapping_head": mapping.head().to_dict(),
        "scenario": scenario,
    }
    s = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()
