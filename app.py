"""고객은 왜 이탈하는가 — 이탈 원인 진단 대시보드"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google.cloud import bigquery
from plotly.subplots import make_subplots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

REF_DATE = pd.Timestamp("2024-12-31")

BQ_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-9eedbe22-48b2-44eb-afd")
BQ_DATASET = "data_agents"


@st.cache_data
def load_customers() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "data_customers.csv"), parse_dates=["join_date"])


@st.cache_data
def load_voc() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "data_voc.csv"))


@st.cache_data
def load_consultations() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "data_consultations.csv"))


@st.cache_data
def load_satisfaction() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "data_satisfaction.csv"))


@st.cache_data
def load_usage() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "data_usage_history.csv"))


@st.cache_resource
def get_bigquery_client() -> bigquery.Client:
    return bigquery.Client(project=BQ_PROJECT_ID)


@st.cache_data(ttl=3600)
def load_agent_metrics() -> pd.DataFrame:
    """agents·consultations·satisfaction을 조인해 상담원 단위 지표를 직접 재계산."""
    client = get_bigquery_client()
    query = f"""
        WITH agent_csat AS (
            SELECT
                a.agent_id, a.team, a.overtime_hours_avg,
                a.agent_satisfaction, a.training_completed_yn,
                AVG(s.csat) AS avg_csat
            FROM `{BQ_PROJECT_ID}.{BQ_DATASET}.agents` a
            JOIN `{BQ_PROJECT_ID}.{BQ_DATASET}.consultations` c ON a.agent_id = c.agent_id
            JOIN `{BQ_PROJECT_ID}.{BQ_DATASET}.satisfaction` s ON c.consult_id = s.consult_id
            GROUP BY a.agent_id, a.team, a.overtime_hours_avg, a.agent_satisfaction, a.training_completed_yn
        ),
        agent_recontact AS (
            SELECT a.agent_id, AVG(CAST(c.is_recontact AS INT64)) AS recontact_rate
            FROM `{BQ_PROJECT_ID}.{BQ_DATASET}.agents` a
            JOIN `{BQ_PROJECT_ID}.{BQ_DATASET}.consultations` c ON a.agent_id = c.agent_id
            GROUP BY a.agent_id
        )
        SELECT ac.*, ar.recontact_rate
        FROM agent_csat ac
        JOIN agent_recontact ar ON ac.agent_id = ar.agent_id
    """
    df = client.query(query).to_dataframe()
    # BigQuery INT64 컬럼은 pandas nullable Int64로 오는데, 이 타입의 to_numpy()는
    # object dtype 배열이 되어 np.corrcoef/np.polyfit이 깨진다 → 표준 float64로 변환
    df["overtime_hours_avg"] = df["overtime_hours_avg"].astype("float64")
    df["agent_satisfaction"] = df["agent_satisfaction"].astype("float64")
    return df


def calc_enps(scores: pd.Series) -> float:
    n = len(scores)
    if n == 0:
        return float("nan")
    promoter = (scores >= 9).sum()
    detractor = (scores <= 6).sum()
    return (promoter / n - detractor / n) * 100


def build_voc_chart(customers: pd.DataFrame, voc: pd.DataFrame) -> go.Figure:
    total_customers = len(customers)
    total_churn = int((customers["churn_yn"] == "Y").sum())
    total_churn_rate = total_churn / total_customers * 100

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
    fig.update_layout(font=dict(family="Malgun Gothic"), showlegend=False, yaxis_ticksuffix="%")
    return fig


def build_channel_chart(satisfaction: pd.DataFrame, consultations: pd.DataFrame) -> go.Figure:
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
    summary = summary.sort_values("csat_avg", ascending=True)
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
    return fig


def build_recontact_segment_chart(customers: pd.DataFrame, consultations: pd.DataFrame) -> go.Figure:
    def to_segment(count: int) -> str:
        if count == 0:
            return "0회"
        if count == 1:
            return "1회"
        return "2회 이상"

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
    fig.update_layout(font=dict(family="Malgun Gothic"), showlegend=False)
    return fig


def build_plan_chart(customers: pd.DataFrame) -> go.Figure:
    summary = (
        customers.groupby("plan")
        .agg(고객수=("customer_id", "count"), 이탈고객수=("churn_yn", lambda s: (s == "Y").sum()))
        .reset_index()
    )
    summary["이탈율"] = summary["이탈고객수"] / summary["고객수"] * 100
    summary = summary.sort_values("이탈율", ascending=False)

    highlight = "베이직"
    color_map = {plan: ("#d03b3b" if plan == highlight else "#898781") for plan in summary["plan"]}

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
    fig.update_layout(font=dict(family="Malgun Gothic"), showlegend=False)
    return fig


def build_region_chart(customers: pd.DataFrame) -> go.Figure:
    summary = (
        customers.groupby("region")
        .agg(고객수=("customer_id", "count"), 이탈고객수=("churn_yn", lambda s: (s == "Y").sum()))
        .reset_index()
    )
    summary["이탈율"] = summary["이탈고객수"] / summary["고객수"] * 100
    summary = summary.sort_values("이탈율", ascending=False)

    highlight = {"부산", "대구"}
    color_map = {
        region: ("#d03b3b" if region in highlight else "#898781") for region in summary["region"]
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

    incheon = summary[summary["region"] == "인천"].iloc[0]
    caption = (
        f"※ 인천은 표본 {int(incheon['고객수'])}건 중 이탈 {int(incheon['이탈고객수'])}건으로 "
        "표본 대비 이탈 건수가 매우 적어 이탈율 수치 해석에 주의가 필요함"
    )
    fig.update_layout(font=dict(family="Malgun Gothic"), showlegend=False, margin=dict(b=100))
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
    return fig


def build_tenure_usage_scatter(customers: pd.DataFrame, usage: pd.DataFrame) -> go.Figure:
    customers = customers.copy()
    customers["tenure_months"] = ((REF_DATE - customers["join_date"]).dt.days / 30.44).round(1)

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
    return fig


st.set_page_config(page_title="고객은 왜 이탈하는가", layout="wide")
st.title("고객은 왜 이탈하는가 — 이탈 원인 진단 대시보드")

customers = load_customers()
voc = load_voc()
consultations = load_consultations()
satisfaction = load_satisfaction()
usage = load_usage()

total_customers = len(customers)
total_churn = int((customers["churn_yn"] == "Y").sum())
total_churn_rate = total_churn / total_customers * 100

col1, col2, col3 = st.columns(3)
col1.metric("전체 고객 수", f"{total_customers}명")
col2.metric("이탈 고객 수", f"{total_churn}명")
col3.metric("전체 이탈율", f"{total_churn_rate:.1f}%")

st.subheader("① VOC로 본 이탈")
st.plotly_chart(build_voc_chart(customers, voc), use_container_width=True)

st.subheader("② 채널·만족도로 본 이탈")
st.plotly_chart(build_channel_chart(satisfaction, consultations), use_container_width=True)

st.subheader("③ 재문의 반복으로 본 이탈")
st.plotly_chart(build_recontact_segment_chart(customers, consultations), use_container_width=True)

st.subheader("④ 요금제로 본 이탈")
st.plotly_chart(build_plan_chart(customers), use_container_width=True)

st.subheader("⑤ 지역으로 본 이탈")
st.plotly_chart(build_region_chart(customers), use_container_width=True)

st.subheader("⑥ 가입기간·이용량으로 본 이탈")
st.plotly_chart(build_tenure_usage_scatter(customers, usage), use_container_width=True)

st.subheader("상담원 관점: 직원만족도와 고객 경험")

try:
    agent_metrics = load_agent_metrics()
except Exception as exc:  # BigQuery 인증/연결 실패 시 나머지 대시보드는 계속 동작
    st.warning(f"BigQuery 연결에 실패해 이 섹션을 표시할 수 없습니다: {exc}")
else:
    team_options = ["전체", "1팀", "2팀", "3팀"]
    selected_team = st.selectbox("팀 선택", team_options, key="agent_team_select")

    # selectbox 값이 바뀌면 app.py가 처음부터 재실행되고, 선택된 팀으로 아래가 새로 계산·렌더링됨
    if selected_team == "전체":
        filtered = agent_metrics
    else:
        filtered = agent_metrics[agent_metrics["team"] == selected_team]

    GAUGE_RED, GAUGE_RED_LIGHT = "#e34948", "#f6c9c8"
    GAUGE_NEUTRAL, GAUGE_BLUE_LIGHT = "#f0efec", "#cde2fb"
    STATUS_GOOD, STATUS_CRITICAL = "#0ca30c", "#d03b3b"
    INK, MUTED, BORDER, BLUE = "#0b0b0b", "#898781", "#c3c2b7", "#2a78d6"

    # --- 07_plotly_직원만족도eNPS스코어카드 로직: 큰 게이지 + 팀별 카드 ---
    enps_selected = calc_enps(filtered["agent_satisfaction"])
    fig_enps = go.Figure()
    fig_enps.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=enps_selected,
            number={"font": {"size": 40, "color": INK}},
            title={"text": f"{selected_team} eNPS", "font": {"size": 18, "color": INK}},
            domain={"x": [0, 0.52], "y": [0, 1]},
            gauge={
                "axis": {"range": [-100, 100], "tickwidth": 1, "tickcolor": MUTED},
                "bar": {"color": INK, "thickness": 0.3},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": BORDER,
                "steps": [
                    {"range": [-100, -50], "color": GAUGE_RED},
                    {"range": [-50, 0], "color": GAUGE_RED_LIGHT},
                    {"range": [0, 50], "color": GAUGE_NEUTRAL},
                    {"range": [50, 100], "color": GAUGE_BLUE_LIGHT},
                ],
                "threshold": {"line": {"color": INK, "width": 3}, "thickness": 0.75, "value": 0},
            },
        )
    )
    card_domains = [[0.58, 0.71], [0.735, 0.865], [0.89, 1.0]]
    for team_name, x_range in zip(["1팀", "2팀", "3팀"], card_domains):
        team_enps = calc_enps(agent_metrics.loc[agent_metrics["team"] == team_name, "agent_satisfaction"])
        color = STATUS_GOOD if team_enps >= 0 else STATUS_CRITICAL
        label = f"{team_name} eNPS" + (" ◀ 선택됨" if team_name == selected_team else "")
        fig_enps.add_trace(
            go.Indicator(
                mode="number",
                value=team_enps,
                number={"valueformat": ".1f", "font": {"size": 28, "color": color}},
                title={"text": label, "font": {"size": 13, "color": INK}},
                domain={"x": x_range, "y": [0.2, 0.8]},
            )
        )
    fig_enps.update_layout(font=dict(family="Malgun Gothic"), height=320, margin=dict(t=60, b=10, l=10, r=10))
    st.plotly_chart(fig_enps, use_container_width=True)

    # --- 08_plotly_번아웃CSAT산점도 로직: OLS 추세선 + r 주석 ---
    if len(filtered) >= 2 and filtered["overtime_hours_avg"].nunique() >= 2:
        r_selected = np.corrcoef(filtered["overtime_hours_avg"].to_numpy(), filtered["avg_csat"].to_numpy())[0, 1]
        fig_scatter = px.scatter(
            filtered,
            x="overtime_hours_avg",
            y="avg_csat",
            hover_name="agent_id",
            trendline="ols",
            trendline_color_override=INK,
            title=f"번아웃(초과근무) × CSAT — {selected_team}",
            labels={"overtime_hours_avg": "월평균 초과근무 시간(h)", "avg_csat": "CSAT 평균"},
        )
        fig_scatter.update_traces(
            marker=dict(size=10, color=BLUE, line=dict(width=1, color="white")),
            hovertemplate="<b>%{hovertext}</b><br>초과근무: %{x}시간<br>CSAT 평균: %{y:.2f}<extra></extra>",
            selector=dict(mode="markers"),
        )
        fig_scatter.add_annotation(
            text=f"r = {r_selected:.2f}",
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            showarrow=False,
            font=dict(size=15, color=INK),
            bgcolor="rgba(255,255,255,0.7)",
        )
        fig_scatter.update_layout(font=dict(family="Malgun Gothic"), plot_bgcolor="white")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.caption(f"{selected_team}은 표본이 너무 적어 추세선을 계산할 수 없습니다.")

    # --- 09_plotly_번아웃CSAT이상치비교 로직: 이상치(초과근무 25h+) 포함 vs 제외 ---
    OUTLIER_THRESHOLD = 25
    is_outlier = filtered["overtime_hours_avg"] >= OUTLIER_THRESHOLD
    df_incl, df_excl = filtered, filtered[~is_outlier]

    def fit_ols(df):
        if len(df) < 2 or df["overtime_hours_avg"].nunique() < 2:
            return None
        x, y = df["overtime_hours_avg"].to_numpy(), df["avg_csat"].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]
        return slope, intercept, r

    fit_incl, fit_excl = fit_ols(df_incl), fit_ols(df_excl)

    if fit_incl and fit_excl:
        x_range = [0, agent_metrics["overtime_hours_avg"].max() + 3]
        y_range = [agent_metrics["avg_csat"].min() - 0.1, agent_metrics["avg_csat"].max() + 0.1]

        fig_cmp = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[f"이상치 포함 (n={len(df_incl)})", f"이상치 제외 (n={len(df_excl)})"],
            horizontal_spacing=0.08,
        )
        fig_cmp.add_trace(
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
        if is_outlier.any():
            fig_cmp.add_trace(
                go.Scatter(
                    x=df_incl.loc[is_outlier, "overtime_hours_avg"],
                    y=df_incl.loc[is_outlier, "avg_csat"],
                    mode="markers",
                    name=f"이상치(≥{OUTLIER_THRESHOLD}h)",
                    marker=dict(size=12, color=GAUGE_RED, symbol="diamond", line=dict(width=1, color="white")),
                    hovertext=df_incl.loc[is_outlier, "agent_id"],
                    hovertemplate="<b>%{hovertext}</b><br>초과근무: %{x}시간<br>CSAT 평균: %{y:.2f}<extra></extra>",
                    legendgroup="outlier",
                ),
                row=1,
                col=1,
            )
        trend_x = np.array(x_range)
        fig_cmp.add_trace(
            go.Scatter(
                x=trend_x,
                y=fit_incl[0] * trend_x + fit_incl[1],
                mode="lines",
                line=dict(color=INK, width=2),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig_cmp.add_trace(
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
        fig_cmp.add_trace(
            go.Scatter(
                x=trend_x,
                y=fit_excl[0] * trend_x + fit_excl[1],
                mode="lines",
                line=dict(color=INK, width=2),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        for col, fit in [(1, fit_incl), (2, fit_excl)]:
            fig_cmp.add_annotation(
                text=f"r = {fit[2]:.2f}<br>기울기 = {fit[0]:.3f}",
                xref="x domain",
                yref="y domain",
                x=0.98,
                y=0.98,
                showarrow=False,
                align="right",
                font=dict(size=13, color=INK),
                bgcolor="rgba(255,255,255,0.75)",
                row=1,
                col=col,
            )
        fig_cmp.update_xaxes(range=x_range, title_text="월평균 초과근무 시간(h)")
        fig_cmp.update_yaxes(range=y_range, title_text="CSAT 평균", col=1)
        fig_cmp.update_layout(
            font=dict(family="Malgun Gothic"),
            title=dict(text=f"이상치(초과근무 {OUTLIER_THRESHOLD}h+) 포함 vs 제외 — {selected_team}", x=0.02),
            plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="left", x=0),
            margin=dict(t=110, b=50, l=50, r=30),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
        if not is_outlier.any():
            st.caption(f"{selected_team}에는 초과근무 {OUTLIER_THRESHOLD}시간 이상 상담원이 없어 두 패널의 결과가 동일합니다.")
    else:
        st.caption(f"{selected_team}은 표본이 너무 적어 이상치 비교를 계산할 수 없습니다.")
