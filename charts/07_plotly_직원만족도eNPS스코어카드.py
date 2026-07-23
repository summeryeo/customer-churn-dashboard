"""BigQuery agents 테이블을 직접 조회해 eNPS(직원 순추천지수)를 재계산하고,
전체 eNPS 게이지 + 팀별 eNPS 숫자 카드로 구성된 스코어카드를 그린다."""
import os

import pandas as pd
import plotly.graph_objects as go
from google.cloud import bigquery

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-9eedbe22-48b2-44eb-afd")
DATASET = "data_agents"
TABLE = "agents"

# agent_satisfaction: 0~10점 설문 점수 (eNPS 원점수)
# 프로모터(9~10점) 비율 - 디트랙터(0~6점) 비율 = eNPS (-100 ~ 100)
client = bigquery.Client(project=PROJECT_ID)
query = f"""
    SELECT team, agent_satisfaction
    FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
    WHERE agent_satisfaction IS NOT NULL
"""
agents = client.query(query).to_dataframe()


def calc_enps(scores: pd.Series) -> float:
    promoter = (scores >= 9).sum()
    detractor = (scores <= 6).sum()
    n = len(scores)
    return (promoter / n - detractor / n) * 100


enps_all = calc_enps(agents["agent_satisfaction"])

team_order = ["1팀", "2팀", "3팀"]
enps_by_team = {
    team: calc_enps(agents.loc[agents["team"] == team, "agent_satisfaction"])
    for team in team_order
}

# dataviz 팔레트: diverging blue<->red, status good/critical
RED = "#e34948"
RED_LIGHT = "#f6c9c8"
NEUTRAL = "#f0efec"
BLUE_LIGHT = "#cde2fb"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
INK = "#0b0b0b"
MUTED = "#898781"
BORDER = "#c3c2b7"

fig = go.Figure()

# 큰 게이지: 전체 eNPS (-100~100), 마이너스 구간은 빨간 계열 배경
fig.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=enps_all,
        number={"font": {"size": 44, "color": INK}},
        title={"text": "전체 직원 eNPS", "font": {"size": 20, "color": INK}},
        domain={"x": [0, 0.52], "y": [0, 1]},
        gauge={
            "axis": {"range": [-100, 100], "tickwidth": 1, "tickcolor": MUTED},
            "bar": {"color": INK, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": BORDER,
            "steps": [
                {"range": [-100, -50], "color": RED},
                {"range": [-50, 0], "color": RED_LIGHT},
                {"range": [0, 50], "color": NEUTRAL},
                {"range": [50, 100], "color": BLUE_LIGHT},
            ],
            "threshold": {
                "line": {"color": INK, "width": 3},
                "thickness": 0.75,
                "value": 0,
            },
        },
    )
)

# 작은 숫자 카드 3개: 팀별 eNPS, 나란히 배치
card_domains = [[0.58, 0.71], [0.735, 0.865], [0.89, 1.0]]
for team, x_range in zip(team_order, card_domains):
    value = enps_by_team[team]
    color = STATUS_GOOD if value >= 0 else STATUS_CRITICAL
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=value,
            number={"suffix": "", "valueformat": ".1f", "font": {"size": 32, "color": color}},
            title={"text": f"{team} eNPS", "font": {"size": 15, "color": INK}},
            domain={"x": x_range, "y": [0.2, 0.8]},
        )
    )

fig.update_layout(
    font=dict(family="Malgun Gothic"),
    title=dict(text="직원 만족도 eNPS 스코어카드", x=0.02, font=dict(size=22, color=INK)),
    paper_bgcolor="white",
    margin=dict(t=90, b=40, l=30, r=30),
    height=380,
)

if __name__ == "__main__":
    fig.show()
