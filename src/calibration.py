def calibrate_households(raw: pd.DataFrame) -> Dict[str,Any]:
    assert raw is not None and len(raw) > 0, "No HH data"
    df = raw.copy()

    # --- Normalize columns from human-readable headers to canonical names ---
    # timestamp
    if "timestamp" not in df.columns:
        tcols = [c for c in df.columns if "timestamp" in c.lower()]
        if not tcols:
            raise ValueError("Household data has no timestamp column.")
        df = df.rename(columns={tcols[0]: "timestamp"})
    # power_kW
    if "power_kW" not in df.columns:
        pcols = [c for c in df.columns if ("power" in c.lower()) and ("kw" in c.lower())]
        if not pcols:
            raise ValueError("Household data has no power_kW column (kW).")
        df = df.rename(columns={pcols[0]: "power_kW"})
    # meter
    if "meter" not in df.columns:
        df["meter"] = "METER_001"
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # --- 15-min kW -> hourly aggregation (kWh per hour) ---
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

    # Merge hour with E_day; validate the merge shape
    merged = hour.merge(day, on=["meter","date","cluster"], how="left", validate="many_to_one")

    # If, for any reason, E_day is missing (shouldn't happen), recompute and re-merge
    if "E_day" not in merged.columns:
        day2 = (
            hour.groupby(["meter","date","cluster"], as_index=False)["kWh_15"]
            .sum()
            .rename(columns={"kWh_15": "E_day"})
        )
        merged = merged.merge(day2, on=["meter","date","cluster"], how="left", validate="many_to_one")

    # Hourly shares (only on positive E_day)
    pos = merged.loc[merged["E_day"].gt(0)].copy()
    pos["share"] = pos["kWh_15"] / pos["E_day"]

    # Baseline μ_c,h = median share per cluster/hour, normalized to sum to 1
    mu = (
        pos.groupby(["cluster","hour"])["share"].median().unstack("hour").fillna(0.0)
    )
    mu = mu.div(mu.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # Day-scaler ln(D) ~ N(0, σ^2); estimate σ by cluster
    day_m = day.copy()
    med = day_m.groupby("cluster")["E_day"].transform("median").replace(0, np.nan)
    eps = np.log(day_m["E_day"] / med)
    sigma_lnD = (
        eps.groupby(day_m["cluster"]).std(ddof=1)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.25)
    )

    # Residual σ per cluster/hour on positives:
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


