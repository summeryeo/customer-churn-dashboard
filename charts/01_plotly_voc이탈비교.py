"""전체 고객 이탈율 vs 해지관련 부정 VOC 이력 고객 이탈율 비교 (plotly)"""
import os

import pandas as pd
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

customers = pd.read_csv(os.path.join(DATA_DIR, "data_customers.csv"))
voc = pd.read_csv(os.path.join(DATA_DIR, "data_voc.csv"))

# 전체 고객 이탈율
total_customers = len(customers)
total_churn = int((customers["churn_yn"] == "Y").sum())
total_churn_rate = total_churn / total_customers * 100

# 해지관련 + 부정 VOC를 남긴 고객 목록 (중복 제거) → customer_id로 연결
target_ids = voc.loc[
    (voc["category"] == "해지관련") & (voc["sentiment"] == "부정"), "customer_id"
].unique()

target_customers = customers[customers["customer_id"].isin(target_ids)]
target_total = len(target_customers)
target_churn = int((target_customers["churn_yn"] == "Y").sum())
target_churn_rate = target_churn / target_total * 100

df = pd.DataFrame(
    {
        "구분": ["전체 고객", "해지관련 부정 VOC 이력 있음"],
        "이탈율": [total_churn_rate, target_churn_rate],
        "고객수": [total_customers, target_total],
        "이탈고객수": [total_churn, target_churn],
    }
)

fig = px.bar(
    df,
    x="구분",
    y="이탈율",
    color="구분",
    color_discrete_map={
        "전체 고객": "#2a78d6",
        "해지관련 부정 VOC 이력 있음": "#d03b3b",
    },
    custom_data=["고객수", "이탈고객수"],
    title="전체 고객 vs 해지관련 부정 VOC 고객 이탈율 비교",
    labels={"이탈율": "이탈율 (%)"},
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
    yaxis_ticksuffix="%",
)

if __name__ == "__main__":
    fig.show()
