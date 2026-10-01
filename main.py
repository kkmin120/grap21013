import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------------
# 기본 설정
# -----------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write("서울의 연도별 평균기온을 바탕으로 기온 변화를 분석하고 미래 기온을 예측합니다.")

# -----------------------------------
# 데이터 불러오기
# -----------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 숫자형 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 연도 만들기
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()

# -----------------------------------
# 연도별 평균기온 계산
# -----------------------------------
yearly = (
    df.dropna(subset=["연도", "평균기온"])
      .groupby("연도")
      .agg(
          연평균기온=("평균기온", "mean"),
          관측일수=("평균기온", "count")
      )
      .reset_index()
)

# -----------------------------------
# 분석에 사용할 기간 필터
# 조건:
# 1. 2025년까지
# 2. 관측일 300일 이상
# -----------------------------------
analysis = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()

analysis = analysis.sort_values("연도")

# -----------------------------------
# 회귀분석
# 독립변수 = 1908년부터 지난 연수
# x = 연도 - 1908
# y = 연평균기온
# -----------------------------------
x = analysis["연도"].values - 1908
y = analysis["연평균기온"].values

# 1차 회귀식
slope, intercept = np.polyfit(x, y, 1)

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# 회귀선 예측값
analysis["회귀예측기온"] = intercept + slope * x

# -----------------------------------
# 회귀선 전체 범위
# 1900 ~ 2100
# -----------------------------------
prediction_years = np.arange(1900, 2101)
prediction_x = prediction_years - 1908
prediction_temperatures = intercept + slope * prediction_x

# -----------------------------------
# 주요 정보
# -----------------------------------
start_year = int(analysis["연도"].min())
end_year = int(analysis["연도"].max())
year_count = len(analysis)

# -----------------------------------
# 분석 결과 표시
# -----------------------------------
st.subheader("📊 회귀분석 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("회귀에 사용한 연도 수", f"{year_count}년")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")

with col4:
    st.metric("상관계수", f"{correlation:.3f}")

st.write(
    f"회귀선은 **{start_year}년부터 {end_year}년까지**, "
    f"관측일수가 300일 이상인 연도의 연평균기온을 사용해 만들었습니다."
)

st.write(
    f"회귀식에서 독립변수는 **1908년부터 지난 연수**를 사용했습니다. "
    f"연수가 1년 증가할 때마다 회귀선의 예상 연평균기온은 "
    f"약 **{slope:.4f}℃**씩 변하도록 계산됩니다."
)

# -----------------------------------
# 산점도 + 회귀선
# -----------------------------------
st.subheader("📈 서울 연평균기온과 회귀선")

fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=analysis["연도"],
        y=analysis["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=analysis["관측일수"],
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        )
    )
)

# 회귀선
fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temperatures,
        mode="lines",
        name="회귀선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 예상기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    height=600
)

# 가로축에 연도를 그대로 표시
fig.update_xaxes(
    tickmode="linear",
    dtick=10,
    tickformat="d"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# 연도 슬라이더
# -----------------------------------
st.subheader("🌡️ 연도를 골라 기온 예측하기")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

selected_x = selected_year - 1908
predicted_temperature = intercept + slope * selected_x

st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temperature:.2f} ℃"
)

# -----------------------------------
# 선택한 연도를 그래프에 표시
# -----------------------------------
selected_actual = analysis[
    analysis["연도"] == selected_year
]

fig2 = go.Figure()

# 회귀선
fig2.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temperatures,
        mode="lines",
        name="회귀선",
        line=dict(width=3)
    )
)

# 선택한 연도의 예측값
fig2.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temperature],
        mode="markers",
        name=f"{selected_year}년 예측",
        marker=dict(size=14),
        hovertemplate=(
            f"{selected_year}년<br>"
            f"예상 연평균기온: {predicted_temperature:.2f}℃"
            "<extra></extra>"
        )
    )
)

# 실제 관측값이 존재하는 연도라면 함께 표시
if not selected_actual.empty:
    actual_temperature = selected_actual.iloc[0]["연평균기온"]

    fig2.add_trace(
        go.Scatter(
            x=[selected_year],
            y=[actual_temperature],
            mode="markers",
            name=f"{selected_year}년 실제값",
            marker=dict(size=12, symbol="diamond"),
            hovertemplate=(
                f"{selected_year}년<br>"
                f"실제 연평균기온: {actual_temperature:.2f}℃"
                "<extra></extra>"
            )
        )
    )

fig2.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=500,
    hovermode="closest"
)

fig2.update_xaxes(
    tickmode="linear",
    dtick=10,
    tickformat="d",
    range=[1900, 2100]
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------
# 안내
# -----------------------------------
st.caption(
    "※ 예측값은 과거 관측자료의 연평균기온에 1차 선형회귀를 적용한 값입니다. "
    "실제 미래의 기온을 보장하는 값은 아닙니다."
)
