"""plan(요금제)별 고객 수·이탈율 비교"""
import os

import pandas as pd
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

customers = pd.read_csv(os.path.join(DATA_DIR, "data_customers.csv"))

summary = (
    customers.groupby("plan")
    .agg(고객수=("customer_id", "count"), 이탈고객수=("churn_yn", lambda s: (s == "Y").sum()))
    .reset_index()
)
summary["이탈율"] = summary["이탈고객수"] / summary["고객수"] * 100
summary = summary.sort_values("이탈율", ascending=False)

HIGHLIGHT = "베이직"
color_map = {plan: ("#d03b3b" if plan == HIGHLIGHT else "#898781") for plan in summary["plan"]}

fig = px.bar(
    summary,
    x="plan",
    y="이탈율",
    color="plan",
    color_discrete_map=color_map,
    custom_data=["고객수", "이탈고객수"],
    title="요금제(plan)별 이탈율",
    labels={"plan": "요금제", "이탈율": "이탈율 (%)"},
)

fig.update_traces(
    hovertemplate=(
        "<b>%{x}</b><br>"
        "고객 수: %{customdata[0]}명<br>"
        "이탈 고객 수: %{customdata[1]}명<br>"
        "이탈율: %{y:.1f}%<extra></extra>"
    )
)

fig.update_layout(
    font=dict(family="Malgun Gothic"),
    showlegend=False,
)

if __name__ == "__main__":
    fig.show()
