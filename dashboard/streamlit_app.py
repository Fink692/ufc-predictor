from __future__ import annotations

import pandas as pd

from ufc_predictor.odds import build_fight_recommendations, rank_value_bets


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Install the dashboard extra first: python -m pip install -e .[dashboard]") from exc

    st.set_page_config(page_title="UFC Predictor", layout="wide")
    st.title("UFC Predictor")
    st.caption("Analytics only. This is not betting advice.")

    with st.sidebar:
        st.header("Inputs")
        predictions_file = st.file_uploader("Predictions CSV", type=["csv"])
        odds_file = st.file_uploader("Odds board CSV", type=["csv"])
        bankroll = st.number_input("Bankroll", min_value=1.0, value=1000.0, step=100.0)
        max_confidence_stake = st.number_input("Max confidence stake", min_value=0.0, value=100.0, step=10.0)
        min_edge = st.number_input("Minimum edge", min_value=-1.0, max_value=1.0, value=0.02, step=0.01)

    if predictions_file is None or odds_file is None:
        st.info("Upload a predictions CSV and odds board CSV to view the value board.")
        return

    predictions = pd.read_csv(predictions_file)
    odds_board = pd.read_csv(odds_file)
    value_bets = rank_value_bets(predictions, odds_board, bankroll=bankroll, min_edge=min_edge)
    recommendations = build_fight_recommendations(predictions, odds_board, max_confidence_stake=max_confidence_stake)

    bet_count = int((value_bets["decision"] == "bet").sum()) if not value_bets.empty else 0
    positive_picks = int((recommendations["value_flag"] == "positive_ev").sum()) if not recommendations.empty else 0
    total_confidence_stake = float(recommendations["confidence_stake"].sum()) if not recommendations.empty else 0.0
    total_expected_profit = float(recommendations["expected_profit"].fillna(0).sum()) if not recommendations.empty else 0.0

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Fights", len(predictions))
    col_b.metric("Value Candidates", bet_count)
    col_c.metric("Positive-EV Picks", positive_picks)
    col_d.metric("Expected Profit", f"${total_expected_profit:,.2f}")

    st.subheader("Fight Recommendations")
    st.dataframe(
        recommendations[
            [
                "event_date",
                "fighter_a",
                "fighter_b",
                "predicted_winner",
                "predicted_win_probability",
                "confidence_stake",
                "best_sportsbook",
                "best_american_odds",
                "expected_profit",
                "value_flag",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Value Board")
    st.dataframe(value_bets, use_container_width=True, hide_index=True)

    st.download_button(
        "Download value board",
        value_bets.to_csv(index=False).encode("utf-8"),
        file_name="value_bets.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download fight recommendations",
        recommendations.to_csv(index=False).encode("utf-8"),
        file_name="fight_recommendations.csv",
        mime="text/csv",
    )

    st.caption(f"Confidence-stake exposure: ${total_confidence_stake:,.2f}")


if __name__ == "__main__":
    main()
