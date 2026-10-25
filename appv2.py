

# ============================================================
# OPTIONAL — LIGHTWEIGHT NEAR-TERM ACTIVITY WATCHLIST
# ============================================================

st.markdown('<div class="section-kicker">Near-term activity watchlist</div>', unsafe_allow_html=True)
with st.expander("Experimental 7-day activity watchlist", expanded=False):
    watchlist_candidates = [
        BASE_DIR / "output" / "model" / "next7d_watchlist.csv",
        BASE_DIR / "data" / "output" / "model" / "next7d_watchlist.csv",
    ]
    metrics_candidates = [
        BASE_DIR / "output" / "model" / "model_metrics.json",
        BASE_DIR / "data" / "output" / "model" / "model_metrics.json",
    ]
    watchlist_path = next((p for p in watchlist_candidates if p.exists()), None)
    metrics_path = next((p for p in metrics_candidates if p.exists()), None)

    st.caption(
        "Optional add-on. It ranks sectors by a lightweight model or persistence baseline, "
        "selected using a chronological validation period. The score is not a probability."
    )
    if watchlist_path is None:
        st.info(
            "No model output found yet. From the project root, run:\n\n"
            "`python fire_harmonizer_light_model.py --science output/harmonized_science_all_years.csv --output-dir output/model --horizon 7`"
        )
    else:
        try:
            watch_df = pd.read_csv(watchlist_path, low_memory=False)
            if watch_df.empty:
                st.warning("The saved watchlist is empty. Re-run the model to create sector scores.")
            else:
                as_of = str(watch_df.iloc[0].get("as_of_date", "unknown"))
                forecast_start = str(watch_df.iloc[0].get("forecast_start", "unknown"))
                forecast_end = str(watch_df.iloc[0].get("forecast_end", "unknown"))
                selected_method = str(watch_df.iloc[0].get("selected_method", "unknown"))
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Data available through", as_of)
                k2.metric("Forecast window begins", forecast_start)
                k3.metric("Forecast window ends", forecast_end)
                k4.metric(
                    "Selected approach",
                    "ML" if selected_method == "hist_gradient_boosting" else (
                        "Persistence" if selected_method == "persistence_baseline" else selected_method
                    ),
                )

                # Normalize the watchlist columns and render a high-contrast table.
                # The chart can work while a themed dataframe grid appears blank, so
                # this table uses explicit white backgrounds and dark text.
                required_watch_columns = {"sector", "model_score"}
                missing_watch_columns = required_watch_columns - set(watch_df.columns)
                if missing_watch_columns:
                    st.error(
                        "The watchlist CSV is missing required columns: "
                        + ", ".join(sorted(missing_watch_columns))
                        + f". File loaded: {watchlist_path}"
                    )
                    st.caption("Available columns: " + ", ".join(map(str, watch_df.columns)))
                else:
                    watch_display = watch_df.copy()
                    if "rank" not in watch_display.columns:
                        watch_display["rank"] = np.arange(1, len(watch_display) + 1)
                    if "watchlist_flag" not in watch_display.columns:
                        watch_display["watchlist_flag"] = np.nan
                    if "selected_method" not in watch_display.columns:
                        watch_display["selected_method"] = selected_method
                    watch_display["rank"] = pd.to_numeric(watch_display["rank"], errors="coerce")
                    watch_display["model_score"] = pd.to_numeric(watch_display["model_score"], errors="coerce")
                    watch_display["watchlist_flag"] = pd.to_numeric(
                        watch_display["watchlist_flag"], errors="coerce"
                    )
                    watch_display = watch_display.sort_values(
                        ["rank", "sector"], ascending=[True, True], na_position="last"
                    )
                    watch_display["rank"] = watch_display["rank"].apply(
                        lambda x: str(int(x)) if pd.notna(x) else "—"
                    )
                    watch_display["model_score"] = watch_display["model_score"].apply(
                        lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                    )
                    watch_display["watchlist_flag"] = watch_display["watchlist_flag"].map(
                        lambda x: ("Above threshold" if x == 1 else "Below threshold" if x == 0 else "—")
                    )
                    watch_display["selected_method"] = watch_display["selected_method"].map(
                        lambda x: (
                            "ML" if str(x) == "hist_gradient_boosting" else
                            "Persistence" if str(x) == "persistence_baseline" else str(x)
                        )
                    )
                    watch_display = watch_display[
                        ["rank", "sector", "model_score", "watchlist_flag", "selected_method"]
                    ].rename(
                        columns={
                            "rank": "Rank",
                            "sector": "Sector",
                            "model_score": "Ranking score (not probability)",
                            "watchlist_flag": "Validation-threshold flag",
                            "selected_method": "Selected approach",
                        }
                    )
                    st.markdown(render_readable_html_table(watch_display), unsafe_allow_html=True)

                    chart_source = watch_df.copy()
                    chart_source["model_score"] = pd.to_numeric(chart_source["model_score"], errors="coerce")
                    chart_source = chart_source.dropna(subset=["model_score"])
                    if chart_source.empty:
                        st.info("The watchlist contains no numeric scores to plot.")
                    else:
                        plot_df = chart_source.sort_values("model_score", ascending=True).set_index("sector")[
                            ["model_score"]
                        ]
                        st.bar_chart(plot_df, use_container_width=True)

                if metrics_path is not None:
                    import json
                    try:
                        model_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                        st.subheader("Chronological back-test metrics")

                        # Parse the schema actually written by fire_harmonizer_light_model.py.
                        # It nests selection metadata under final_method_selection and
                        # aggregate validation results under walk_forward_backtest.
                        selection = model_metrics.get("final_method_selection", {})
                        backtest = model_metrics.get("walk_forward_backtest", {})
                        chosen = selection.get(
                            "selected_method", model_metrics.get("selected_method", selected_method)
                        )
                        reason = selection.get(
                            "reason", model_metrics.get("selection_reason", "Not recorded in metrics file")
                        )
                        chosen_label = (
                            "ML" if chosen == "hist_gradient_boosting" else
                            "Persistence" if chosen == "persistence_baseline" else str(chosen)
                        )
                        st.write("**Validation-selected approach:**", chosen_label)
                        st.write("**Validation-based selection reason:**", reason)

                        if backtest:
                            agg_rows = [{
                                "Held-out rows": backtest.get("rows", "—"),
                                "Positive rate": (
                                    f"{100 * float(backtest['positive_rate']):.1f}%"
                                    if backtest.get("positive_rate") is not None else "—"
                                ),
                                "Precision": (
                                    f"{float(backtest['precision']):.3f}"
                                    if backtest.get("precision") is not None else "—"
                                ),
                                "Recall": (
                                    f"{float(backtest['recall']):.3f}"
                                    if backtest.get("recall") is not None else "—"
                                ),
                                "F1": (
                                    f"{float(backtest['f1']):.3f}"
                                    if backtest.get("f1") is not None else "—"
                                ),
                                "Average precision": (
                                    f"{float(backtest['average_precision']):.3f}"
                                    if backtest.get("average_precision") is not None else "—"
                                ),
                                "Test years": ", ".join(map(str, backtest.get("test_years", []))) or "—",
                            }]
                            st.markdown(
                                render_readable_html_table(pd.DataFrame(agg_rows)),
                                unsafe_allow_html=True,
                            )
                        else:
                            legacy = model_metrics.get("test_metrics_by_method", {})
                            if legacy:
                                st.markdown(
                                    render_readable_html_table(pd.DataFrame(legacy).T.reset_index().rename(
                                        columns={"index": "Method"}
                                    )),
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.warning(
                                    "No walk-forward metrics were found in this JSON. "
                                    "Re-run fire_harmonizer_light_model.py to regenerate the model report."
                                )

                        annual_rows = model_metrics.get("annual_fold_summary", [])
                        if annual_rows:
                            st.write("**Annual walk-forward results**")
                            annual_df = pd.DataFrame(annual_rows)
                            preferred_annual = [
                                "test_year", "validation_year", "selected_method", "train_rows",
                                "validation_rows", "test_rows", "test_precision", "test_recall",
                                "test_f1", "test_average_precision", "threshold",
                            ]
                            annual_df = annual_df[[c for c in preferred_annual if c in annual_df.columns]].copy()
                            annual_df = annual_df.rename(columns={
                                "test_year": "Test year",
                                "validation_year": "Validation year",
                                "selected_method": "Selected method",
                                "train_rows": "Training rows",
                                "validation_rows": "Validation rows",
                                "test_rows": "Test rows",
                                "test_precision": "Precision",
                                "test_recall": "Recall",
                                "test_f1": "F1",
                                "test_average_precision": "Average precision",
                                "threshold": "Threshold",
                            })
                            for col in ["Precision", "Recall", "F1", "Average precision", "Threshold"]:
                                if col in annual_df.columns:
                                    annual_df[col] = annual_df[col].apply(
                                        lambda x: f"{float(x):.3f}" if pd.notna(x) else "—"
                                    )
                            st.markdown(
                                render_readable_html_table(annual_df),
                                unsafe_allow_html=True,
                            )
                    except Exception as exc:
                        st.error(f"Metrics file could not be read: {exc}")
                else:
                    st.info(
                        "No model_metrics.json was found. The watchlist is available, but chronological "
                        "back-test metrics will appear only after running the model script successfully."
                    )
        except Exception as exc:
            st.error(f"Could not read the saved watchlist: {exc}")

    st.warning(
        "Research-only interpretation: the target is whether the existing M7 C score reaches the "
        "C-active threshold in the next seven days for the same sector. It does not predict confirmed fires, "
        "burned area, fire severity, or emergency risk. Because historical M7 baselines may include later dates, "
        "this is not a live early-warning product."
    )


# ============================================================
# STEP 6 — M10 STATIC CHARTS AT THE VERY BOTTOM
# ============================================================

st.markdown('<div class="section-kicker">Scientific plates</div>', unsafe_allow_html=True)

with st.expander("M10 Static Scientific Plates", expanded=False):
    st.caption(
        "These figures are generated offline by generate_charts.py. Streamlit only reads the saved PNG files here."
    )

    chart_directory = find_chart_directory()

    for row_start in range(0, len(M10_CHARTS), 2):
        columns = st.columns(2, gap="large")

        for column, (filename, title, description) in zip(
            columns,
            M10_CHARTS[row_start:row_start + 2],
        ):
            with column:
                st.markdown(
                    f'<div class="event-card"><div class="event-card-title">{title}</div><div class="event-card-copy">{description}</div></div>',
                    unsafe_allow_html=True,
                )

                image_path = (
                    chart_directory / filename
                    if chart_directory is not None
                    else None
                )

                if image_path is not None and image_path.exists():
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.info(f"Pre-generated plate not found: {filename}")

