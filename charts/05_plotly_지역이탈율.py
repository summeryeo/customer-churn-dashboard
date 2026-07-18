"""region(지역)별 고객 수·이탈율 비교"""
import os

import pandas as pd
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

customers = pd.read_csv(os.path.join(DATA_DIR, "data_customers.csv"))

summary = (
    customers.groupby("region")
    .agg(고객수=("customer_id", "count"), 이탈고객수=("churn_yn", lambda s: (s == "Y").sum()))
    .reset_index()
)
summary["이탈율"] = summary["이탈고객수"] / summary["고객수"] * 100
summary = summary.sort_values("이탈율", ascending=False)

HIGHLIGHT = {"부산", "대구"}
color_map = {
    region: ("#d03b3b" if region in HIGHLIGHT else "#898781") for region in summary["region"]
}

fig = px.bar(
    summary,
    x="region",
    y="이탈율",
    color="region",
    color_discrete_map=color_map,
    custom_data=["고객수", "이탈고객수"],
    title="지역(region)별 이탈율",
    labels={"region": "지역", "이탈율": "이탈율 (%)"},
)

fig.update_traces(
    hovertemplate=(
        "<b>%{x}</b><br>"
        "고객 수: %{customdata[0]}명<br>"
        "이탈 고객 수: %{customdata[1]}명<br>"
        "이탈율: %{y:.1f}%<extra></extra>"
    )
)

# 인천은 표본은 있으나 이탈 건수가 극히 적어 해석에 주의가 필요함을 캡션으로 명시
incheon = summary[summary["region"] == "인천"].iloc[0]
caption = (
    f"※ 인천은 표본 {int(incheon['고객수'])}건 중 이탈 {int(incheon['이탈고객수'])}건으로 "
    "표본 대비 이탈 건수가 매우 적어 이탈율 수치 해석에 주의가 필요함"
)

fig.update_layout(
    font=dict(family="Malgun Gothic"),
    showlegend=False,
    margin=dict(b=100),
)

fig.add_annotation(
    text=caption,
    xref="paper",
    yref="paper",
    x=0,
    y=-0.25,
    showarrow=False,
    align="left",
    font=dict(size=12, color="#52514e"),
)

if __name__ == "__main__":
    fig.show()
