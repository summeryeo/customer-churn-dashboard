"""가입기간(tenure_months) vs 평균 데이터 사용량 산점도 (color=churn_yn)"""
import os

import pandas as pd
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

REF_DATE = pd.Timestamp("2024-12-31")

customers = pd.read_csv(os.path.join(DATA_DIR, "data_customers.csv"), parse_dates=["join_date"])
usage = pd.read_csv(os.path.join(DATA_DIR, "data_usage_history.csv"))

# join_date ~ 2024-12-31 까지의 개월 수 (일수 기반 환산, 30.44일=1개월)
customers["tenure_months"] = (
    (REF_DATE - customers["join_date"]).dt.days / 30.44
).round(1)

# 고객별 평균 data_gb
avg_usage = usage.groupby("customer_id")["data_gb"].mean().rename("avg_data_gb")

merged = customers.merge(avg_usage, on="customer_id", how="inner")
merged["avg_data_gb"] = merged["avg_data_gb"].round(2)

fig = px.scatter(
    merged,
    x="tenure_months",
    y="avg_data_gb",
    color="churn_yn",
    color_discrete_map={"N": "#2a78d6", "Y": "#d03b3b"},
    custom_data=["customer_id", "tenure_months", "avg_data_gb", "churn_yn"],
    title="가입기간 vs 평균 데이터 사용량 (이탈 여부별)",
    labels={
        "tenure_months": "가입기간 (개월)",
        "avg_data_gb": "평균 데이터 사용량 (GB)",
        "churn_yn": "이탈 여부",
    },
)

fig.update_traces(
    marker=dict(size=8, opacity=0.75),
    hovertemplate=(
        "customer_id: %{customdata[0]}<br>"
        "가입기간: %{customdata[1]:.1f}개월<br>"
        "평균 데이터 사용량: %{customdata[2]:.2f}GB<br>"
        "이탈 여부: %{customdata[3]}<extra></extra>"
    ),
)

fig.update_layout(font=dict(family="Malgun Gothic"))

if __name__ == "__main__":
    fig.show()
