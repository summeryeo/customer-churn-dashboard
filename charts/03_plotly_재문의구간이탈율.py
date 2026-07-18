"""고객별 재문의(is_recontact=Y) 횟수 구간(0회/1회/2회 이상)별 이탈율 비교"""
import os

import pandas as pd
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

customers = pd.read_csv(os.path.join(DATA_DIR, "data_customers.csv"))
consultations = pd.read_csv(os.path.join(DATA_DIR, "data_consultations.csv"))


def to_segment(count: int) -> str:
    if count == 0:
        return "0회"
    if count == 1:
        return "1회"
    return "2회 이상"


# 고객별 재문의(Y) 횟수 집계
recontact_count = (
    consultations.assign(is_y=(consultations["is_recontact"] == "Y"))
    .groupby("customer_id")["is_y"]
    .sum()
    .rename("recontact_count")
)

merged = customers.merge(recontact_count, on="customer_id", how="left")
merged["recontact_count"] = merged["recontact_count"].fillna(0)
merged["segment"] = merged["recontact_count"].apply(to_segment)

segment_order = ["0회", "1회", "2회 이상"]
summary = (
    merged.groupby("segment")
    .agg(고객수=("customer_id", "count"), 이탈고객수=("churn_yn", lambda s: (s == "Y").sum()))
    .reindex(segment_order)
    .reset_index()
)
summary["이탈율"] = summary["이탈고객수"] / summary["고객수"] * 100

# 전체 평균 이탈율(직접 재계산)
overall_churn_rate = (customers["churn_yn"] == "Y").mean() * 100

color_map = {"0회": "#898781", "1회": "#898781", "2회 이상": "#d03b3b"}

fig = px.bar(
    summary,
    x="segment",
    y="이탈율",
    color="segment",
    color_discrete_map=color_map,
    custom_data=["고객수", "이탈고객수"],
    title="재문의 횟수 구간별 이탈율",
    labels={"segment": "재문의 횟수", "이탈율": "이탈율 (%)"},
    category_orders={"segment": segment_order},
)

fig.update_traces(
    hovertemplate=(
        "<b>%{x}</b><br>"
        "고객 수: %{customdata[0]}명<br>"
        "이탈 고객 수: %{customdata[1]}명<br>"
        "이탈율: %{y:.1f}%<extra></extra>"
    )
)

fig.add_hline(
    y=overall_churn_rate,
    line_dash="dash",
    line_color="#0b0b0b",
    annotation_text=f"전체 평균 이탈율 {overall_churn_rate:.1f}%",
    annotation_position="top right",
)

fig.update_layout(
    font=dict(family="Malgun Gothic"),
    showlegend=False,
)

if __name__ == "__main__":
    fig.show()
