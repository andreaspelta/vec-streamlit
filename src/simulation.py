from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

# ------------------------
# Helpers for clustering
# ------------------------
def season_of(ts):
    m = ts.month
    if m in (12,1,2): return "Winter"
    if m in (3,4,5): return "Spring"
    if m in (6,7,8): return "Summer"
    return "Autumn"

def daytype_of(ts):
    wd = ts.weekday()
    if wd == 5: return "Saturday"
    if wd == 6: return "Holiday"
    return "Weekday"

def cluster_label(ts):
    return f"{season_of(ts)}-{daytype_of(ts)}"

# ------------------------
# Sampling engines
# ------------------------
def sample_hh(hh_params, hours_idx, S, rng):
    """Return dict of HH loads by household_id (demo: three HH)."""
    hh_ids = sorted(set(["H011","H012","H021"]))
    out = {}
    for hid in hh_ids:
        out[hid] = sample_consumption(hh_params, hours_idx, S, rng)
    return out

def sample_shops(shop_params, hours_idx, S, rng):
    shop_ids = sorted(set(["K101","K102","K103"]))
    out = {}
    for sid in shop_ids:
        out[sid] = sample_consumption(shop_params, hours_idx, S, rng, zero_inflated=True)
    return out

def sample_consumption(params, hours_idx, S, rng, zero_inflated=False):
    """Generic sampler for HH/Shop with day-scaler D (ln ~ Normal) and hourly residuals."""
    mu = params["mu"]              # clusters x 24
    sig_lnD = params["sigma_lnD"]  # per cluster
    sig_res = params["sigma_resid"]# clusters x 24
    if zero_inflated:
        p0 = params["p_zero"]      # clusters x 24

    df = pd.DataFrame(index=hours_idx, columns=["kWh"], dtype=float)
    df["cluster"] = [cluster_label(ts) for ts in hours_idx]
    df["hour"] = [ts.hour for ts in hours_idx]
    df["date"] = [ts.date() for ts in hours_idx]

    # Draw D per day/cluster
    days = pd.DataFrame({"date": df["date"].unique()})
    days["cluster"] = [cluster_label(pd.Timestamp(d)) for d in days["date"]]
    days["sigma"] = days["cluster"].map(sig_lnD.to_dict()).fillna(0.25)
    days["lnD"] = rng.normal(loc=0.0, scale=days["sigma"].values)
    days["D"] = np.exp(days["lnD"])
    df = df.merge(days[["date","D"]], on="date", how="left")

    # Baseline per hour
    def mu_of(row):
        c = row["cluster"]; h = row["hour"]
        try: return float(mu.loc[c, h])
        except: return 0.0
    df["mu"] = df.apply(mu_of, axis=1)

    # Residuals
    def sig_of(row):
        c = row["cluster"]; h = row["hour"]
        try: return float(sig_res.loc[c, h])
        except: return 0.25
    df["sig"] = df.apply(sig_of, axis=1)
    eps = rng.normal(size=len(df))
    df["kWh"] = (df["D"] * df["mu"] * np.exp(df["sig"]*eps)).clip(lower=0.0)

    if zero_inflated:
        def p0_of(row):
            c = row["cluster"]; h = row["hour"]
            try: return float(p0.loc[c, h])
            except: return 0.0
        df["p0"] = df.apply(p0_of, axis=1)
        mask_zero = rng.uniform(size=len(df)) < df["p0"].values
        df.loc[mask_zero, "kWh"] = 0.0

    return df["kWh"].values

def sample_pv(pv_params, kWp_map, hours_idx, S, rng):
    """Shared-weather PV generation per prosumer (kWh) with robust season fallbacks."""
    Senv = pv_params["S"]              # DataFrame seasons x 24
    ll = pv_params["loglogistic"]      # dict season -> {c, scale}
    mark = pv_params["markov"]         # dict season -> {"P": [[..],[..]], "beta": {alpha,beta}}
    prosumers = sorted(kWp_map.keys())

    # Defaults for missing seasons
    DEF_P = np.array([[0.7,0.3],[0.3,0.7]])
    DEF_BETA = {"alpha": 5.0, "beta": 5.0}
    DEF_LL = {"c": 2.0, "scale": 1.0}

    # Build day-state chain per season
    dates = pd.DatetimeIndex(hours_idx).normalize().unique()
    seasons = [season_of(pd.Timestamp(d)) for d in dates]
    states = []
    prev = {}
    for d, s in zip(dates, seasons):
        P = np.array(mark.get(s, {}).get("P", DEF_P))
        if P.shape != (2,2) or not np.all(np.isfinite(P)):
            P = DEF_P
        if s not in prev:
            # stationary approx for 2-state chain
            pi1 = P[0,1]/(P[0,1]+P[1,0]+1e-9)
            state = int(rng.uniform() < pi1)
        else:
            p01 = P[prev[s], 1]
            state = int(rng.uniform() < p01)
        prev[s] = state
        states.append((d, s, state))
    day_state = pd.DataFrame(states, columns=["date","season","state"])

    # Daily multiplier per day from season-specific Fisk (median ~ 1)
    Ms = []
    for d, s, _ in states:
        pars = ll.get(s, DEF_LL)
        c = float(pars.get("c", 2.0)); sc = float(pars.get("scale", 1.0))
        # rvs handles Generator via random_state=
        M = stats.fisk.rvs(c, loc=0, scale=sc, random_state=rng)
        Ms.append((d, M))
    Mser = pd.Series(dict(Ms))

    # Build hourly per prosumer
    out = {}
    for pid in prosumers:
        kWp = float(kWp_map[pid])
        kwh = []
        for ts in hours_idx:
            s = season_of(ts)
            h = ts.hour
            try:
                S_h = float(Senv.loc[s, h])
                if not np.isfinite(S_h): S_h = 0.0
            except Exception:
                S_h = 0.0
            M = float(Mser.get(ts.normalize(), 1.0))
            beta_ab = mark.get(s, {}).get("beta", DEF_BETA)
            a = float(beta_ab.get("alpha", 5.0)); b = float(beta_ab.get("beta", 5.0))
            if a <= 0 or b <= 0 or not np.isfinite(a) or not np.isfinite(b):
                a, b = 5.0, 5.0
            cclear = rng.beta(a, b)
            eps = rng.normal(scale=0.1)
            base = kWp * S_h * M * (cclear/(1.0+cclear))
            kwh.append(max(0.0, base*np.exp(eps)))
        out[pid] = np.array(kwh, dtype=float)
    return out

# ------------------------
# Water-filling allocation
# ------------------------
def equal_level_fill(supply, demands):
    """
    Equal-level water-filling:
    supply: float
    demands: array of non-negative unmet demands
    returns matched array, residual supply
    """
    n = len(demands)
    if n == 0 or supply <= 0:
        return np.zeros(n), supply
    d = demands.copy().astype(float)
    matched = np.zeros(n)
    active = np.where(d > 0)[0].tolist()
    s = float(supply)
    while s > 1e-12 and len(active) > 0:
        alloc = s / max(len(active), 1)
        used = 0.0
        new_active = []
        for idx in active:
            take = min(alloc, d[idx])
            matched[idx] += take
            d[idx] -= take
            used += take
            if d[idx] > 1e-12:
                new_active.append(idx)
        s -= used
        active = new_active
        if used < 1e-12:
            break
    return matched, s

# ------------------------
# Price layer
# ------------------------
def price_layer(zonal_eur_per_kwh, retail_spread, alpha, phi, hh_gift=False, segment="HH"):
    """
    zonal_eur_per_kwh >= 0 recommended; if <0 apply override outside this function.
    retail = zonal + spread (spread >= 0; constant per year).
    gap = phi * spread
    remainder = (1-phi)*spread
    Ppros = zonal + alpha * remainder
    Pcons = retail - (1-alpha) * remainder
    """
    retail = zonal_eur_per_kwh + retail_spread
    if zonal_eur_per_kwh < 0:
        return 0.0, retail, 0.0, retail  # pros, cons, gap, retail
    spread = retail_spread
    gap = phi * spread
    rem = (1 - phi) * spread
    Ppros = min(retail, zonal_eur_per_kwh + alpha * rem)
    Pcons = min(retail, retail - (1 - alpha) * rem)
    if hh_gift and segment == "HH":
        return 0.0, 0.0, 0.0, retail
    return Ppros, Pcons, gap, retail

# ------------------------
# Monte Carlo runner
# ------------------------
def run_mc(hh_params, shop_params, pv_params, prices, retail_setup, mapping, kWp_map, scenario):
    rng = np.random.default_rng(scenario["seed"])
    # Build hourly index from PUN timestamps (calendar anchor)
    hours = pd.to_datetime(prices["pun"]["timestamp"]).sort_values().unique()
    hours = pd.DatetimeIndex(hours)

    S = scenario["S"]
    prosumers = sorted(kWp_map.keys())
    households = sorted(mapping["household_id"].unique())
    shops = ["K101","K102","K103"]  # demo placeholders

    # Zonal price per hour (€/MWh -> €/kWh)
    zonal = prices["zonal"].copy()
    zonal["eur_kwh"] = zonal["zonal_price (EUR_per_MWh)"] / 1000.0

    matched_hh_all, matched_shop_all = [], []
    import_hh_all, import_shop_all = [], []
    export_all = []
    pros_rev_all, cons_cost_all, platform_margin_all = [], [], []

    for s_idx in range(S):
        hh_series = sample_hh(hh_params, hours, S, rng)
        shop_series = sample_shops(shop_params, hours, S, rng)
        pv_series = sample_pv(pv_params, kWp_map, hours, S, rng)

        m_hh = np.zeros(len(hours))
        m_shop = np.zeros(len(hours))
        imp_hh = np.zeros(len(hours))
        imp_shop = np.zeros(len(hours))
        exp_total = np.zeros(len(hours))
        pros_rev = 0.0
        cons_cost = 0.0
        platform_margin = 0.0

        for t_idx, ts in enumerate(hours):
            rows = zonal[zonal["timestamp"] == ts.strftime("%Y-%m-%d %H:%M:%S%z")]
            if rows.empty:
                continue
            z_eur_kwh = float(rows.iloc[0]["eur_kwh"])
            Ppros_HH, Pcons_HH, gap_HH, retail_HH = price_layer(
                z_eur_kwh, retail_setup["spread_hh"], scenario["alpha_hh"], scenario["phi_hh"],
                hh_gift=scenario["hh_gift"], segment="HH"
            )
            Ppros_SH, Pcons_SH, gap_SH, retail_SH = price_layer(
                z_eur_kwh, retail_setup["spread_shop"], scenario["alpha_shop"], scenario["phi_shop"],
                hh_gift=False, segment="SHOP"
            )
            delta_unm = scenario["delta_unm"]
            loss_factor = scenario["loss_factor"]

            d_hh = np.array([hh_series[h][t_idx] for h in households], dtype=float)
            d_shop = np.array([shop_series[k][t_idx] for k in shops], dtype=float)
            g_pv = np.array([pv_series[p][t_idx] for p in prosumers], dtype=float)

            # STEP 1: designated HH fill (per prosumer)
            for i, pid in enumerate(prosumers):
                hh_list = mapping.loc[mapping["prosumer_id"]==pid, "household_id"].tolist()
                idxs = [households.index(h) for h in hh_list if h in households]
                if not idxs:
                    continue
                need = d_hh[idxs].copy()
                supply = float(g_pv[i])
                matched, resid = equal_level_fill(supply, need)
                m_hh[t_idx] += matched.sum()
                d_hh[idxs] -= matched
                g_pv[i] = resid
                pros_rev += matched.sum() * Ppros_HH
                cons_cost += matched.sum() * Pcons_HH
                platform_margin += (matched.sum() * (1-loss_factor)) * gap_HH

            # STEP 2: pool residual to shops
            pool = g_pv.sum()
            matched_s, resid_pool = equal_level_fill(pool, d_shop.copy())
            m_shop[t_idx] += matched_s.sum()
            d_shop -= matched_s
            g_pv_residual = resid_pool
            pros_rev += matched_s.sum() * Ppros_SH
            cons_cost += matched_s.sum() * Pcons_SH
            platform_margin += (matched_s.sum() * (1-loss_factor)) * gap_SH

            # STEP 3: imports/exports
            imp_hh[t_idx] += d_hh.sum()
            imp_shop[t_idx] += d_shop.sum()
            cons_cost += d_hh.sum() * retail_HH
            cons_cost += d_shop.sum() * retail_SH
            exp_total[t_idx] += g_pv_residual
            pros_rev += g_pv_residual * delta_unm

        # Monthly fees / platform cost (simplified annualized)
        cons_cost += 12.0 * (scenario["fee_hh"] + scenario["fee_shop"])
        platform_margin += 12.0 * (scenario["fee_pros"] + scenario["fee_hh"] + scenario["fee_shop"]) - scenario["platform_monthly"]

        matched_hh_all.append(m_hh.sum()); matched_shop_all.append(m_shop.sum())
        import_hh_all.append(imp_hh.sum()); import_shop_all.append(imp_shop.sum())
        export_all.append(exp_total.sum())
        pros_rev_all.append(pros_rev)
        cons_cost_all.append(cons_cost)
        platform_margin_all.append(platform_margin)

    results = {
        "hours": hours,
        "matched_hh": np.array(matched_hh_all),
        "matched_shop": np.array(matched_shop_all),
        "import_hh": np.array(import_hh_all),
        "import_shop": np.array(import_shop_all),
        "export": np.array(export_all),
        "pros_rev": np.array(pros_rev_all),
        "cons_cost": np.array(cons_cost_all),
        "platform_margin": np.array(platform_margin_all),
    }
    return results

# ------------------------
# Visuals
# ------------------------
def sankey_aggregate(mc):
    labels = ["PV Gen", "HH matched", "SHOP matched", "Export"]
    source = [0, 0, 0]
    target = [1, 2, 3]
    value = [float(mc["matched_hh"].mean()), float(mc["matched_shop"].mean()), float(mc["export"].mean())]
    fig = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels),
        link=dict(source=source, target=target, value=value)
    ))
    fig.update_layout(title="Average energy allocation (across scenarios)")
    return fig

def price_panel_example(mc, scenario):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="α_HH", x=["HH"], y=[scenario["alpha_hh"]]))
    fig.add_trace(go.Bar(name="φ_HH", x=["HH"], y=[scenario["phi_hh"]]))
    fig.add_trace(go.Bar(name="α_SHOP", x=["SHOP"], y=[scenario["alpha_shop"]]))
    fig.add_trace(go.Bar(name="φ_SHOP", x=["SHOP"], y=[scenario["phi_shop"]]))
    fig.update_layout(barmode="group", title="Illustrative α / φ settings")
    return fig
