import io
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src import utils, calibration, pv, simulation, kpi, exporters, diagnostics

APP_TITLE = "Virtual Energy Community — Streamlit"
TZ = pytz.timezone("Europe/Rome")

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    page_icon="⚡",
)

# -----------------------------
# Sidebar Navigation
# -----------------------------
PAGES = [
    "Home / Checklist",
    "Data Upload & Calibration",
    "Scenario Builder",
    "Monte Carlo Run",
    "KPI Dashboard",
    "Exports & Reports",
    "Diagnostics",
    "Help",
]
page = st.sidebar.radio("• Workflow", PAGES, index=0)

# Global session state
for key, default in [
    ("hh_raw", None), ("shop_raw", None), ("pv_json", None),
    ("hh_params", None), ("shop_params", None), ("pv_params", None),
    ("prices", None), ("retail_setup", None), ("mapping", None),
    ("scenario", None), ("scenario_hash", None),
    ("mc_results", None), ("kpi_summary", None), ("diag", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

def header():
    left, mid, right = st.columns([0.5, 0.25, 0.25])
    with left:
        st.markdown(f"### {APP_TITLE}")
    with mid:
        if st.session_state.get("scenario"):
            hh_gift = st.session_state["scenario"].get("hh_gift", False)
            seed = st.session_state["scenario"].get("seed", 12345)
            st.markdown(
                f"**hh_gift:** `{str(hh_gift).upper()}`  \n**seed:** `{seed}`"
            )
    with right:
        if st.session_state.get("scenario_hash"):
            st.markdown(
                f"**scenario_set_id**  \n`{st.session_state['scenario_hash'][:16]}…`"
            )

header()

# -----------------------------
# Page 1 — Home / Checklist
# -----------------------------
if page == "Home / Checklist":
    st.info(
        "Use the sidebar to follow the steps. If you want to see the app working "
        "without files, click **Use sample data** below."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        ok1 = st.session_state["hh_params"] is not None and st.session_state["shop_params"] is not None and st.session_state["pv_params"] is not None
        st.metric("Calibration", "Ready" if ok1 else "Pending")
    with c2:
        ok2 = st.session_state["prices"] is not None and st.session_state["retail_setup"] is not None and st.session_state["mapping"] is not None
        st.metric("Scenario", "Ready" if ok2 else "Pending")
    with c3:
        ok3 = st.session_state["mc_results"] is not None
        st.metric("Monte Carlo", "Done" if ok3 else "Not run")

    st.markdown("---")
    if st.button("Use sample data (demo)"):
        # Tiny synthetic samples for a quick demo
        st.session_state["hh_raw"], st.session_state["shop_raw"] = utils.make_demo_loads()
        st.session_state["pv_json"] = pv.make_demo_pv_json()
        st.success("Sample HH/SHOP/PV data loaded.")
        # Calibrate immediately
        st.session_state["hh_params"] = calibration.calibrate_households(st.session_state["hh_raw"])
        st.session_state["shop_params"] = calibration.calibrate_shops(st.session_state["shop_raw"])
        st.session_state["pv_params"] = pv.calibrate_pv(st.session_state["pv_json"])
        st.success("Calibrations completed on sample data.")
        # Minimal prices
        price = utils.make_demo_prices()
        st.session_state["prices"] = price
        # Retail spreads and segments (year-constant)
        st.session_state["retail_setup"] = {
            "spread_hh": 0.10,  # €/kWh
            "spread_shop": 0.12,  # €/kWh
        }
        # Minimal mapping (prosumer→HH)
        st.session_state["mapping"] = utils.make_demo_mapping()
        # Default scenario
        st.session_state["scenario"] = {
            "alpha_hh": 0.5, "phi_hh": 0.2,
            "alpha_shop": 0.5, "phi_shop": 0.2,
            "delta_unm": 0.03, "loss_factor": 0.0,
            "fee_pros": 0.0, "fee_hh": 0.0, "fee_shop": 0.0, "platform_monthly": 0.0,
            "hh_gift": False, "seed": 12345, "S": 200,
        }
        st.session_state["scenario_hash"] = utils.scenario_hash(
            st.session_state["hh_params"], st.session_state["shop_params"],
            st.session_state["pv_params"], st.session_state["prices"],
            st.session_state["retail_setup"], st.session_state["mapping"],
            st.session_state["scenario"]
        )
        st.success("Demo scenario prepared. Go to **Monte Carlo Run**.")

# -----------------------------
# Page 2 — Data Upload & Calibration
# -----------------------------
elif page == "Data Upload & Calibration":
    st.markdown("#### Upload raw data and run the fitting pipelines.")

    tabs = st.tabs(["Households", "Small Shops", "Prosumer PV"])
    # Households
    with tabs[0]:
        st.markdown("**Excel**: One sheet per meter. Columns: "
                    "`timestamp (Europe/Rome, ISO8601)`, `power_kW (15min average)`.")
        f = st.file_uploader("Upload Households Excel", type=["xlsx"], key="hh_upl")
        if f:
            st.session_state["hh_raw"] = calibration.load_households_excel(f)
            st.success(f"Loaded: {len(st.session_state['hh_raw'])} rows.")
        if st.button("Fit Households"):
            st.session_state["hh_params"] = calibration.calibrate_households(st.session_state["hh_raw"])
            st.success("Households calibrated.")
            calibration.render_household_diagnostics(st.session_state["hh_params"])

            buf = exporters.export_household_calibration_xlsx(st.session_state["hh_params"])
            st.download_button("Download households_calibration.xlsx", data=buf, file_name="households_calibration.xlsx")

    # Shops
    with tabs[1]:
        st.markdown("**Excel**: One sheet per meter. Columns: "
                    "`timestamp (Europe/Rome, ISO8601)`, `ActiveEnergy_Generale (kWh per 15min)`.")
        f = st.file_uploader("Upload Small Shops Excel", type=["xlsx"], key="shop_upl")
        if f:
            st.session_state["shop_raw"] = calibration.load_shops_excel(f)
            st.success(f"Loaded: {len(st.session_state['shop_raw'])} rows.")
        if st.button("Fit Shops"):
            st.session_state["shop_params"] = calibration.calibrate_shops(st.session_state["shop_raw"])
            st.success("Shops calibrated.")
            calibration.render_shop_diagnostics(st.session_state["shop_params"])

            buf = exporters.export_shop_calibration_xlsx(st.session_state["shop_params"])
            st.download_button("Download shops_calibration.xlsx", data=buf, file_name="shops_calibration.xlsx")

    # PV
    with tabs[2]:
        st.markdown("**JSON** per-kWp hourly. Fields: `timezone`, `unit`, `records[timestamp, energy_kWh_per_kWp]`.")
        f = st.file_uploader("Upload PV per-kWp JSON", type=["json"], key="pv_upl")
        if f:
            st.session_state["pv_json"] = json.load(f)
            st.success(f"Loaded PV records: {len(st.session_state['pv_json'].get('records', []))}.")
        if st.button("Fit PV (Option-B v3)"):
            st.session_state["pv_params"] = pv.calibrate_pv(st.session_state["pv_json"])
            st.success("PV calibrated.")
            pv.render_pv_diagnostics(st.session_state["pv_params"])
            buf = exporters.export_pv_calibration_xlsx(st.session_state["pv_params"])
            st.download_button("Download PV_calibration.xlsx", data=buf, file_name="pv_calibration.xlsx")

# -----------------------------
# Page 3 — Scenario Builder
# -----------------------------
elif page == "Scenario Builder":
    st.markdown("#### Define prices, spreads, parameters, and mapping (wizard).")
    with st.expander("Prices & Retail (year-constant spreads)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            pun_file = st.file_uploader("Upload PUN hourly CSV", type=["csv"], key="pun_csv")
        with col2:
            zonal_file = st.file_uploader("Upload Zonal hourly CSV (LONG or WIDE)", type=["csv"], key="zonal_csv")
        if pun_file and zonal_file:
            prices = utils.load_prices(pun_file, zonal_file)
            st.session_state["prices"] = prices
            st.success(f"Loaded {len(prices['pun'])} PUN hours and {len(prices['zonal'])} zonal rows.")
        c1, c2 = st.columns(2)
        with c1:
            spread_hh = st.number_input("HH spread over zonal (€/kWh, constant over year)", min_value=0.0, value=0.10, step=0.01)
        with c2:
            spread_shop = st.number_input("SHOP spread over zonal (€/kWh, constant over year)", min_value=0.0, value=0.12, step=0.01)
        st.session_state["retail_setup"] = {"spread_hh": spread_hh, "spread_shop": spread_shop}

    with st.expander("Community parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            alpha_hh = st.slider("α_HH", 0.0, 1.0, 0.5, 0.01)
            alpha_shop = st.slider("α_SHOP", 0.0, 1.0, 0.5, 0.01)
        with c2:
            phi_hh = st.slider("φ_HH", 0.0, 0.99, 0.2, 0.01)
            phi_shop = st.slider("φ_SHOP", 0.0, 0.99, 0.2, 0.01)
        with c3:
            delta_unm = st.number_input("δ_unm (€/kWh) unmatched premium", min_value=0.0, value=0.03, step=0.005)
            loss_factor = st.number_input("ℓ loss factor (0..1)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

        c4, c5, c6, c7 = st.columns(4)
        with c4:
            fee_pros = st.number_input("Monthly fee Prosumer (€)", min_value=0.0, value=0.0, step=1.0)
        with c5:
            fee_hh = st.number_input("Monthly fee HH (€)", min_value=0.0, value=0.0, step=1.0)
        with c6:
            fee_shop = st.number_input("Monthly fee SHOP (€)", min_value=0.0, value=0.0, step=1.0)
        with c7:
            platform_monthly = st.number_input("Platform monthly cost (€)", min_value=0.0, value=0.0, step=1.0)

        c8, c9, c10 = st.columns(3)
        with c8:
            hh_gift = st.toggle("HH-Gift (scenario-level)", value=False, help="If ON: HH matched kWh priced 0 €/kWh for both prosumer and HH; platform gap on HH = 0.")
        with c9:
            seed = st.number_input("Random seed", min_value=0, value=12345, step=1)
        with c10:
            S = st.number_input("# Monte-Carlo scenarios (≤ 10000 on Cloud)", min_value=10, max_value=10000, value=200, step=10)

    with st.expander("Mapping wizard (prosumer → designated HH)", expanded=True):
        st.markdown("Use the wizard to create a simple mapping. You can also upload an optional CSV (prosumer_id, household_id, zone).")
        uploaded_map = st.file_uploader("Upload mapping CSV", type=["csv"], key="map_csv")
        if uploaded_map:
            mapping_df = pd.read_csv(uploaded_map)
        else:
            mapping_df = utils.make_demo_mapping()  # starter
        st.dataframe(mapping_df, use_container_width=True)
        st.session_state["mapping"] = mapping_df

    # PV plant sizes
    with st.expander("Prosumer kWp input", expanded=True):
        st.markdown("Enter kWp per prosumer (used to scale per-kWp PV).")
        kWp_json = st.text_area("JSON mapping {prosumer_id: kWp}", value='{"P001": 5.0, "P002": 3.0}')
        try:
            kWp_map = json.loads(kWp_json)
            st.session_state["kWp_map"] = {str(k): float(v) for k, v in kWp_map.items()}
            st.success("kWp map parsed.")
        except Exception as e:
            st.warning(f"Invalid JSON for kWp map: {e}")

    if st.button("Save scenario"):
        st.session_state["scenario"] = {
            "alpha_hh": alpha_hh, "phi_hh": phi_hh,
            "alpha_shop": alpha_shop, "phi_shop": phi_shop,
            "delta_unm": delta_unm, "loss_factor": loss_factor,
            "fee_pros": fee_pros, "fee_hh": fee_hh, "fee_shop": fee_shop,
            "platform_monthly": platform_monthly,
            "hh_gift": hh_gift, "seed": int(seed), "S": int(S),
        }
        st.session_state["scenario_hash"] = utils.scenario_hash(
            st.session_state["hh_params"], st.session_state["shop_params"],
            st.session_state["pv_params"], st.session_state["prices"],
            st.session_state["retail_setup"], st.session_state["mapping"],
            st.session_state["scenario"]
        )
        st.success(f"Scenario saved. scenario_set_id = {st.session_state['scenario_hash']}")

# -----------------------------
# Page 4 — Monte Carlo Run
# -----------------------------
elif page == "Monte Carlo Run":
    st.markdown("#### Run the simulation and check energy flows.")
    ready = all([
        st.session_state["hh_params"] is not None,
        st.session_state["shop_params"] is not None,
        st.session_state["pv_params"] is not None,
        st.session_state["prices"] is not None,
        st.session_state["retail_setup"] is not None,
        st.session_state["mapping"] is not None,
        st.session_state.get("kWp_map") is not None,
        st.session_state["scenario"] is not None,
    ])
    if not ready:
        st.warning("Please complete calibration and scenario builder steps first.")
    else:
        if st.button("Run Monte Carlo"):
            with st.spinner("Simulating..."):
                mc = simulation.run_mc(
                    hh_params=st.session_state["hh_params"],
                    shop_params=st.session_state["shop_params"],
                    pv_params=st.session_state["pv_params"],
                    prices=st.session_state["prices"],
                    retail_setup=st.session_state["retail_setup"],
                    mapping=st.session_state["mapping"],
                    kWp_map=st.session_state["kWp_map"],
                    scenario=st.session_state["scenario"],
                )
                st.session_state["mc_results"] = mc
                st.session_state["kpi_summary"] = kpi.summarize_kpis(mc, st.session_state["scenario"])
                st.session_state["diag"] = diagnostics.compute_diagnostics(mc, st.session_state["scenario"])
            st.success("Monte Carlo complete.")

        if st.session_state["mc_results"] is not None:
            st.markdown("**Example visual: energy allocation Sankey (aggregated)**")
            fig = simulation.sankey_aggregate(st.session_state["mc_results"])
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Price split panel (example hour)**")
            pfig = simulation.price_panel_example(st.session_state["mc_results"], st.session_state["scenario"])
            st.plotly_chart(pfig, use_container_width=True)

# -----------------------------
# Page 5 — KPI Dashboard
# -----------------------------
elif page == "KPI Dashboard":
    st.markdown("#### Distributional KPIs (P05, P10, P50, P90, P95, mean, sd).")
    if st.session_state["kpi_summary"] is None:
        st.info("Run Monte Carlo first.")
    else:
        st.dataframe(st.session_state["kpi_summary"]["cards"], use_container_width=True)
        st.markdown("---")
        options = st.session_state["kpi_summary"]["hist_options"]
        kpi_name = st.selectbox("Select KPI for histogram/violin", options)
        fig = kpi.hist_violin(st.session_state["kpi_summary"], kpi_name)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Monthly fan chart**")
        mfig = kpi.monthly_fan_chart(st.session_state["kpi_summary"], kpi_name)
        st.plotly_chart(mfig, use_container_width=True)

# -----------------------------
# Page 6 — Exports & Reports
# -----------------------------
elif page == "Exports & Reports":
    if st.session_state["mc_results"] is None:
        st.info("No results yet. Run Monte Carlo first.")
    else:
        st.markdown("**Download consolidated Excel**")
        buf = exporters.export_all_in_one(
            st.session_state["mc_results"],
            st.session_state["kpi_summary"],
            st.session_state["scenario"],
            st.session_state["scenario_hash"],
        )
        st.download_button("Download COMMUNITY_ALL_IN_ONE.xlsx", data=buf, file_name="COMMUNITY_ALL_IN_ONE.xlsx")

        st.markdown("**Parquet/CSV facts**")
        pq = exporters.export_fact_tables(st.session_state["mc_results"], st.session_state["kpi_summary"])
        for name, (mime, binary) in pq.items():
            st.download_button(f"Download {name}", data=binary, file_name=name, mime=mime)

# -----------------------------
# Page 7 — Diagnostics
# -----------------------------
elif page == "Diagnostics":
    if st.session_state["diag"] is None:
        st.info("Run Monte Carlo to see diagnostics.")
    else:
        st.dataframe(st.session_state["diag"]["summary"], use_container_width=True)
        st.markdown("Open a failing scenario path to inspect flows/prices, if any.")
        # Minimal example: show first failing hour (if exists)
        if not st.session_state["diag"]["violations"].empty:
            st.dataframe(st.session_state["diag"]["violations"].head(50), use_container_width=True)

# -----------------------------
# Page 8 — Help
# -----------------------------
elif page == "Help":
    st.markdown("""
**Quick help**

• **Data formats:**  
Households (Excel; 15-min kW); Shops (Excel; 15-min kWh in `ActiveEnergy_Generale`); PV JSON per-kWp hourly; PUN/Zonal hourly CSV in Europe/Rome tz.

• **Calendar anchor:**  
The **hourly simulation index is taken from your PUN/Zonal timestamps** (8760/8784). All series are aligned to this index.

• **Spreads:**  
Year-constant spreads per segment (HH, SHOP). `retail = zonal + spread`.

• **HH-Gift:**  
Scenario-level toggle; when ON, HH matched kWh are priced **0 €/kWh** for both HH and prosumer; platform gap on HH = 0.

• **KPIs:**  
All KPIs are **distributions**; we report P05, P10, P50, P90, P95, mean, sd.

See the Exports section for Excel/Parquet downloads. 
""")
