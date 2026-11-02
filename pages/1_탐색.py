import streamlit as st
import pandas as pd
import plotly.express as px

# 1. 브라우저 탭 제목과 아이콘 설정
st.set_page_config(
    page_title="탐색 - 뇌졸중 예측 실습실",
    page_icon="🔍",
    layout="wide"
)

# 2. 데이터 불러오기 (첫 화면과 동일한 주소, UTF-8 인코딩)
DATA_URL = "https://raw.githubusercontent.com/cofls969/data/main/stroke.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    return df

df = load_data()

# 보기 쉬운 그래프 라벨을 위한 파생 열 생성
df["뇌졸중_여부"] = df["stroke"].map({0: "뇌졸중 없음 (0)", 1: "뇌졸중 겪음 (1)"})
df["고혈압_여부"] = df["hypertension"].map({0: "고혈압 없음 (0)", 1: "고혈압 있음 (1)"})
df["심장병_여부"] = df["heart_disease"].map({0: "심장병 없음 (0)", 1: "심장병 있음 (1)"})

# 화면 제목
st.title("🔍 데이터 탐색")
st.markdown("나이와 혈당 분포, 질병 유무에 따른 뇌졸중 비율, 결측치 특징 등을 차트와 표로 살펴봅니다.")
st.divider()

# ---------------------------------------------------------
# 1) 나이와 평균 혈당의 분포 (히스토그램 두 개 나란히)
# ---------------------------------------------------------
st.subheader("1. 나이와 평균 혈당 분포")
col1, col2 = st.columns(2)

with col1:
    fig_age_hist = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이(age) 분포",
        labels={"age": "나이", "count": "사람 수"},
        color_discrete_sequence=["#4C78A8"]
    )
    fig_age_hist.update_layout(yaxis_title="사람 수")
    st.plotly_chart(fig_age_hist, use_container_width=True)

with col2:
    fig_gluc_hist = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당(avg_glucose_level) 분포",
        labels={"avg_glucose_level": "평균 혈당", "count": "사람 수"},
        color_discrete_sequence=["#F58518"]
    )
    fig_gluc_hist.update_layout(yaxis_title="사람 수")
    st.plotly_chart(fig_gluc_hist, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# 2) 뇌졸중 여부에 따른 나이와 평균 혈당 비교 (상자그림 2개 + 평균값 표)
# ---------------------------------------------------------
st.subheader("2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교")
col3, col4 = st.columns(2)

with col3:
    fig_age_box = px.box(
        df,
        x="뇌졸중_여부",
        y="age",
        color="뇌졸중_여부",
        title="뇌졸중 여부별 나이(age) 상자그림",
        labels={"뇌졸중_여부": "뇌졸중 여부", "age": "나이"},
        color_discrete_sequence=["#72B7B2", "#E45756"]
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with col4:
    fig_gluc_box = px.box(
        df,
        x="뇌졸중_여부",
        y="avg_glucose_level",
        color="뇌졸중_여부",
        title="뇌졸중 여부별 평균 혈당(avg_glucose_level) 상자그림",
        labels={"뇌졸중_여부": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"},
        color_discrete_sequence=["#72B7B2", "#E45756"]
    )
    st.plotly_chart(fig_gluc_box, use_container_width=True)

# 두 그룹의 평균값 표
st.markdown("**📌 뇌졸중 여부별 나이 및 평균 혈당의 평균값 비교**")
mean_df = (
    df.groupby("뇌졸중_여부", as_index=False)[["age", "avg_glucose_level"]]
    .mean()
    .rename(columns={
        "뇌졸중_여부": "그룹",
        "age": "평균 나이 (세)",
        "avg_glucose_level": "평균 혈당 수치"
    })
)
mean_df["평균 나이 (세)"] = mean_df["평균 나이 (세)"].round(2)
mean_df["평균 혈당 수치"] = mean_df["평균 혈당 수치"].round(2)
st.dataframe(mean_df, hide_index=True, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# 3) 고혈압 및 심장병 유무에 따른 뇌졸중 비율 (막대그래프 2개)
# ---------------------------------------------------------
st.subheader("3. 고혈압 및 심장병 유무에 따른 뇌졸중 비율")
col5, col6 = st.columns(2)

with col5:
    hyp_ratio = (
        df.groupby("고혈압_여부", as_index=False)["stroke"]
        .mean()
    )
    hyp_ratio["뇌졸중 비율 (%)"] = (hyp_ratio["stroke"] * 100).round(2)

    fig_hyp = px.bar(
        hyp_ratio,
        x="고혈압_여부",
        y="뇌졸중 비율 (%)",
        color="고혈압_여부",
        text="뇌졸중 비율 (%)",
        title="고혈압 유무별 뇌졸중 비율 (%)",
        labels={"고혈압_여부": "고혈압 유무"},
        color_discrete_sequence=["#54A24B", "#E45756"]
    )
    fig_hyp.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_hyp.update_layout(yaxis_range=[0, max(hyp_ratio["뇌졸중 비율 (%)"]) * 1.25])
    st.plotly_chart(fig_hyp, use_container_width=True)

with col6:
    heart_ratio = (
        df.groupby("심장병_여부", as_index=False)["stroke"]
        .mean()
    )
    heart_ratio["뇌졸중 비율 (%)"] = (heart_ratio["stroke"] * 100).round(2)

    fig_heart = px.bar(
        heart_ratio,
        x="심장병_여부",
        y="뇌졸중 비율 (%)",
        color="심장병_여부",
        text="뇌졸중 비율 (%)",
        title="심장병 유무별 뇌졸중 비율 (%)",
        labels={"심장병_여부": "심장병 유무"},
        color_discrete_sequence=["#54A24B", "#E45756"]
    )
    fig_heart.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_heart.update_layout(yaxis_range=[0, max(heart_ratio["뇌졸중 비율 (%)"]) * 1.25])
    st.plotly_chart(fig_heart, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# 4) 체질량지수(bmi) 빈 값 그룹과 전체의 뇌졸중 비율 나란히 비교 표
# ---------------------------------------------------------
st.subheader("4. 체질량지수(bmi) 결측치(빈 값) 집단과 전체 집단의 뇌졸중 비율 비교")

bmi_missing_df = df[df["bmi"].isna()]

total_cnt = len(df)
total_stroke_cnt = int((df["stroke"] == 1).sum())
total_stroke_rate = round((total_stroke_cnt / total_cnt) * 100, 2)

missing_cnt = len(bmi_missing_df)
missing_stroke_cnt = int((bmi_missing_df["stroke"] == 1).sum())
missing_stroke_rate = round((missing_stroke_cnt / missing_cnt) * 100, 2) if missing_cnt > 0 else 0.0

bmi_compare_df = pd.DataFrame({
    "구분": ["사람 수 (명)", "뇌졸중 발생 수 (명)", "뇌졸중 비율 (%)"],
    "전체 사람들": [f"{total_cnt:,}명", f"{total_stroke_cnt:,}명", f"{total_stroke_rate:.2f}%"],
    "bmi가 비어 있는 사람들": [f"{missing_cnt:,}명", f"{missing_stroke_cnt:,}명", f"{missing_stroke_rate:.2f}%"]
})

st.dataframe(bmi_compare_df, hide_index=True, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# 5) 흡연 상태(smoking_status) 값마다 사람 수 표
# ---------------------------------------------------------
st.subheader("5. 흡연 상태(smoking_status)별 사람 수")

smoking_counts = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .reset_index()
)
smoking_counts.columns = ["흡연 상태 (smoking_status)", "사람 수 (명)"]
smoking_counts["비율 (%)"] = ((smoking_counts["사람 수 (명)"] / len(df)) * 100).round(2)

st.dataframe(smoking_counts, hide_index=True, use_container_width=True)
