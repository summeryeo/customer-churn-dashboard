"""BigQuery agents·consultations·satisfaction 테이블을 조인해 상담원별 CSAT 평균을
직접 재계산하고, 초과근무 시간(번아웃 지표)과의 관계를 산점도 + OLS 추세선으로 본다."""
import os

import numpy as np
import plotly.express as px
from google.cloud import bigquery

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-9eedbe22-48b2-44eb-afd")
DATASET = "data_agents"

client = bigquery.Client(project=PROJECT_ID)
query = f"""
    SELECT
        a.agent_id,
        a.overtime_hours_avg,
        AVG(s.csat) AS avg_csat
    FROM `{PROJECT_ID}.{DATASET}.agents` a
    JOIN `{PROJECT_ID}.{DATASET}.consultations` c ON a.agent_id = c.agent_id
    JOIN `{PROJECT_ID}.{DATASET}.satisfaction` s ON c.consult_id = s.consult_id
    GROUP BY a.agent_id, a.overtime_hours_avg
"""
agent_csat = client.query(query).to_dataframe()

r = np.corrcoef(agent_csat["overtime_hours_avg"], agent_csat["avg_csat"])[0, 1]

fig = px.scatter(
    agent_csat,
    x="overtime_hours_avg",
    y="avg_csat",
    hover_name="agent_id",
    trendline="ols",
    trendline_color_override="#0b0b0b",
    title="번아웃(초과근무 시간)과 CSAT 평균의 관계",
    labels={"overtime_hours_avg": "월평균 초과근무 시간(h)", "avg_csat": "상담원별 CSAT 평균"},
)

# 산점(마커) 트레이스에만 커스텀 스타일·툴팁 적용, OLS 추세선 트레이스는 그대로 둠
fig.update_traces(
    marker=dict(size=10, color="#2a78d6", line=dict(width=1, color="white")),
    hovertemplate=(
        "<b>%{hovertext}</b><br>"
        "초과근무 시간: %{x}시간<br>"
        "CSAT 평균: %{y:.2f}<extra></extra>"
    ),
    selector=dict(mode="markers"),
)

fig.update_layout(
    font=dict(family="Malgun Gothic"),
    plot_bgcolor="white",
    margin=dict(t=90, b=60, l=60, r=40),
)

fig.add_annotation(
    text=f"r = {r:.2f}",
    xref="paper",
    yref="paper",
    x=0.98,
    y=0.98,
    showarrow=False,
    align="right",
    font=dict(size=16, color="#0b0b0b"),
    bgcolor="rgba(255,255,255,0.7)",
)

if __name__ == "__main__":
    fig.show()
