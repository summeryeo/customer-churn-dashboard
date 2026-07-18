"""고객은 왜 이탈하는가 — 이탈 원인 진단 대시보드"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

REF_DATE = pd.Timestamp("2024-12-31")


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
