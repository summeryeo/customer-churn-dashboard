"""BigQuery agents·consultations·satisfaction 테이블을 조인해 교육 이수 여부(Y/N)별
CSAT 평균·재문의율 평균을 상담원 단위로 직접 재계산하고 좌우 subplot 막대그래프로 비교한다."""
import os

import pandas as pd
import plotly.graph_objects as go
from google.cloud import bigquery
from plotly.subplots import make_subplots

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-9eedbe22-48b2-44eb-afd")
DATASET = "data_agents"

client = bigquery.Client(project=PROJECT_ID)
query = f"""
    WITH agent_csat AS (
        SELECT a.agent_id, a.training_completed_yn, AVG(s.csat) AS avg_csat
        FROM `{PROJECT_ID}.{DATASET}.agents` a
        JOIN `{PROJECT_ID}.{DATASET}.consultations` c ON a.agent_id = c.agent_id
        JOIN `{PROJECT_ID}.{DATASET}.satisfaction` s ON c.consult_id = s.consult_id
        GROUP BY a.agent_id, a.training_completed_yn
    ),
    agent_recontact AS (
        SELECT a.agent_id, AVG(CAST(c.is_recontact AS INT64)) AS recontact_rate
        FROM `{PROJECT_ID}.{DATASET}.agents` a
        JOIN `{PROJECT_ID}.{DATASET}.consultations` c ON a.agent_id = c.agent_id
        GROUP BY a.agent_id
    )
    SELECT ac.agent_id, ac.training_completed_yn, ac.avg_csat, ar.recontact_rate
    FROM agent_csat ac
    JOIN agent_recontact ar ON ac.agent_id = ar.agent_id
"""
agent_df = client.query(query).to_dataframe()

# 상담원 단위 지표를 교육 이수 여부로 그룹 평균 (그룹 내 상담원을 동일 가중치로 취급)
group = (
    agent_df.groupby("training_completed_yn")[["avg_csat", "recontact_rate"]]
    .mean()
    .reindex([True, False])  # Y(이수) 먼저, N(미이수) 다음
)
group["recontact_pct"] = group["recontact_rate"] * 100

labels = ["이수(Y)", "미이수(N)"]
HIGHLIGHT = "#2a78d6"  # Y: 강조색
GRAY = "#898781"       # N: 회색 계열
colors = [HIGHLIGHT, GRAY]

fig = make_subplots(rows=1, cols=2, subplot_titles=["CSAT 평균", "재문의율 평균(%)"])

fig.add_trace(
    go.Bar(
        x=labels,
        y=group["avg_csat"],
        marker_color=colors,
        text=[f"{v:.2f}" for v in group["avg_csat"]],
        textposition="outside",
        showlegend=False,
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Bar(
        x=labels,
        y=group["recontact_pct"],
        marker_color=colors,
        text=[f"{v:.1f}%" for v in group["recontact_pct"]],
        textposition="outside",
        showlegend=False,
    ),
    row=1,
    col=2,
)

fig.update_yaxes(title_text="CSAT 평균", range=[0, group["avg_csat"].max() * 1.25], row=1, col=1)
fig.update_yaxes(title_text="재문의율(%)", range=[0, group["recontact_pct"].max() * 1.25], row=1, col=2)

fig.update_layout(
    font=dict(family="Malgun Gothic"),
    title=dict(text="교육 이수 여부(Y/N)에 따른 CSAT·재문의율 비교", x=0.02),
    plot_bgcolor="white",
    margin=dict(t=90, b=60, l=60, r=40),
)

if __name__ == "__main__":
    fig.show()
