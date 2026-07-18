"""channel별 CSAT 평균 vs 재문의율 비교 (단일 y축, 그룹 막대)

CSAT(1~5점)와 재문의율(0~100%)은 척도가 달라 이중 y축(dual-axis)으로 겹치면
두 값의 정렬이 임의적이 되어 실제로 없는 상관관계처럼 보일 수 있다.
따라서 CSAT를 100점 만점으로 환산해 하나의 y축(0~100) 위에서 비교한다.
"""
import os

import pandas as pd
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

satisfaction = pd.read_csv(os.path.join(DATA_DIR, "data_satisfaction.csv"))
consultations = pd.read_csv(os.path.join(DATA_DIR, "data_consultations.csv"))

# consult_id로 연결
merged = satisfaction.merge(
    consultations[["consult_id", "channel", "is_recontact"]],
    on="consult_id",
    how="inner",
)

summary = (
    merged.groupby("channel")
    .agg(
        csat_avg=("csat", "mean"),
        recontact_rate=("is_recontact", lambda s: (s == "Y").mean() * 100),
        n=("consult_id", "count"),
    )
    .reset_index()
)

# CSAT 낮은 순 정렬
summary = summary.sort_values("csat_avg", ascending=True)

# 단일 축(0~100) 비교를 위해 CSAT(1~5)를 100점 만점으로 환산
summary["csat_100"] = summary["csat_avg"] / 5 * 100

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=summary["channel"],
        y=summary["csat_100"],
        name="CSAT (100점 환산)",
        marker_color="#2a78d6",
        customdata=summary[["csat_avg", "recontact_rate"]],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "CSAT 평균: %{customdata[0]:.2f}점 (5점 만점)<br>"
            "재문의율: %{customdata[1]:.1f}%<extra></extra>"
        ),
    )
)

fig.add_trace(
    go.Bar(
        x=summary["channel"],
        y=summary["recontact_rate"],
        name="재문의율 (%)",
        marker_color="#008300",
        customdata=summary[["csat_avg", "recontact_rate"]],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "CSAT 평균: %{customdata[0]:.2f}점 (5점 만점)<br>"
            "재문의율: %{customdata[1]:.1f}%<extra></extra>"
        ),
    )
)

fig.update_layout(
    title="channel별 CSAT(100점 환산) vs 재문의율 비교 (CSAT 낮은 순)",
    xaxis_title="channel",
    yaxis_title="점수 / 비율 (0~100)",
    barmode="group",
    font=dict(family="Malgun Gothic"),
    yaxis_range=[0, 100],
)

if __name__ == "__main__":
    fig.show()
