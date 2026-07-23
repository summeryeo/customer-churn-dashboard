"""08번 산점도(번아웃×CSAT)에서 이상치로 보이는 AG02·AG03(월평균 초과근무 25시간 이상)을
포함했을 때/제외했을 때의 상관계수·추세선 기울기를 나란히 비교한다."""
import os

import numpy as np
import plotly.graph_objects as go
from google.cloud import bigquery
from plotly.subplots import make_subplots

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-9eedbe22-48b2-44eb-afd")
DATASET = "data_agents"
OUTLIER_IDS = ["AG02", "AG03"]  # 초과근무 25시간 이상

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

is_outlier = agent_csat["agent_id"].isin(OUTLIER_IDS)
df_incl = agent_csat
df_excl = agent_csat[~is_outlier]


def fit_ols(df):
    x, y = df["overtime_hours_avg"].to_numpy(), df["avg_csat"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return slope, intercept, r


slope_incl, intercept_incl, r_incl = fit_ols(df_incl)
slope_excl, intercept_excl, r_excl = fit_ols(df_excl)

# 두 패널을 동일 축 범위로 비교해야 기울기 변화가 왜곡 없이 보임
X_RANGE = [0, agent_csat["overtime_hours_avg"].max() + 3]
Y_RANGE = [agent_csat["avg_csat"].min() - 0.1, agent_csat["avg_csat"].max() + 0.1]

BLUE = "#2a78d6"
RED = "#e34948"
INK = "#0b0b0b"
MUTED = "#52514e"

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[f"이상치 포함 (n={len(df_incl)})", f"이상치 제외 (n={len(df_excl)})"],
    horizontal_spacing=0.08,
)

# --- 왼쪽: 이상치 포함, AG02·AG03을 빨간색으로 구분 표시 ---
fig.add_trace(
    go.Scatter(
        x=df_incl.loc[~is_outlier, "overtime_hours_avg"],
        y=df_incl.loc[~is_outlier, "avg_csat"],
        mode="markers",
        name="일반",
        marker=dict(size=10, color=BLUE, line=dict(width=1, color="white")),
        hovertext=df_incl.loc[~is_outlier, "agent_id"],
        hovertemplate="<b>%{hovertext}</b><br>초과근무: %{x}시간<br>CSAT 평균: %{y:.2f}<extra></extra>",
        legendgroup="normal",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_incl.loc[is_outlier, "overtime_hours_avg"],
        y=df_incl.loc[is_outlier, "avg_csat"],
        mode="markers",
        name="이상치(AG02·AG03)",
        marker=dict(size=12, color=RED, symbol="diamond", line=dict(width=1, color="white")),
        hovertext=df_incl.loc[is_outlier, "agent_id"],
        hovertemplate="<b>%{hovertext}</b><br>초과근무: %{x}시간<br>CSAT 평균: %{y:.2f}<extra></extra>",
        legendgroup="outlier",
    ),
    row=1,
    col=1,
)
trend_x_incl = np.array(X_RANGE)
fig.add_trace(
    go.Scatter(
        x=trend_x_incl,
        y=slope_incl * trend_x_incl + intercept_incl,
        mode="lines",
        name="OLS 추세선",
        line=dict(color=INK, width=2),
        showlegend=False,
    ),
    row=1,
    col=1,
)

# --- 오른쪽: 이상치 제외 ---
fig.add_trace(
    go.Scatter(
        x=df_excl["overtime_hours_avg"],
        y=df_excl["avg_csat"],
        mode="markers",
        name="일반",
        marker=dict(size=10, color=BLUE, line=dict(width=1, color="white")),
        hovertext=df_excl["agent_id"],
        hovertemplate="<b>%{hovertext}</b><br>초과근무: %{x}시간<br>CSAT 평균: %{y:.2f}<extra></extra>",
        legendgroup="normal",
        showlegend=False,
    ),
    row=1,
    col=2,
)
trend_x_excl = np.array(X_RANGE)
fig.add_trace(
    go.Scatter(
        x=trend_x_excl,
        y=slope_excl * trend_x_excl + intercept_excl,
        mode="lines",
        name="OLS 추세선",
        line=dict(color=INK, width=2),
        showlegend=False,
    ),
    row=1,
    col=2,
)

for col, r, slope in [(1, r_incl, slope_incl), (2, r_excl, slope_excl)]:
    fig.add_annotation(
        text=f"r = {r:.2f}<br>기울기 = {slope:.3f}",
        xref="x domain",
        yref="y domain",
        x=0.98,
        y=0.98,
        showarrow=False,
        align="right",
        font=dict(size=14, color=INK),
        bgcolor="rgba(255,255,255,0.75)",
        row=1,
        col=col,
    )

fig.update_xaxes(range=X_RANGE, title_text="월평균 초과근무 시간(h)")
fig.update_yaxes(range=Y_RANGE, title_text="상담원별 CSAT 평균", col=1)

fig.update_layout(
    font=dict(family="Malgun Gothic"),
    title=dict(text="번아웃×CSAT 상관관계: 이상치(AG02·AG03) 포함 vs 제외 비교", x=0.02),
    plot_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
    margin=dict(t=140, b=100, l=60, r=40),
)

caption = (
    f"※ 이상치 제외 시 상관계수는 {r_incl:.2f} → {r_excl:.2f}로 다소 완만해지지만, "
    f"추세선 기울기는 {slope_incl:.3f} → {slope_excl:.3f}로 오히려 더 가팔라짐 "
    "(10시간당 CSAT 변화 "
    f"{slope_incl*10:.2f} → {slope_excl*10:.2f}). "
    "즉 AG02·AG03이 상관관계의 '강도'를 다소 부풀렸을 뿐, 번아웃-CSAT의 음(-)의 방향성 자체를 만든 것은 아님."
)
fig.add_annotation(
    text=caption,
    xref="paper",
    yref="paper",
    x=0,
    y=-0.22,
    showarrow=False,
    align="left",
    font=dict(size=12, color=MUTED),
)

if __name__ == "__main__":
    fig.show()
