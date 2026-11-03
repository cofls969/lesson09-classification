import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

# 1. 브라우저 탭 제목과 아이콘 설정
st.set_page_config(
    page_title="채점표 - 뇌졸중 예측 실습실",
    page_icon="📊",
    layout="wide"
)

# 2. 데이터 불러오기
DATA_URL = "https://raw.githubusercontent.com/cofls969/data/main/stroke.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    return df

df = load_data()

# 우리말 이름과 열 이름 매핑
KOR_TO_COL = {
    "나이": "age",
    "평균 혈당": "avg_glucose_level",
    "체질량지수": "bmi",
    "고혈압": "hypertension",
    "심장병": "heart_disease"
}

st.title("📊 모델 채점표 (혼동행렬과 평가 지표)")
st.markdown("테스트 데이터를 바탕으로 **로지스틱 회귀(확률로 답하는 모델)**와 **의사결정트리(질문으로 답하는 모델)**의 예측 결과를 혼동행렬과 여러 평가 지표로 비교해 봅니다.")
st.divider()

# ---------------------------------------------------------
# 3. 입력 속성 선택 ('분류 모델' 페이지와 동일)
# ---------------------------------------------------------
st.subheader("1. 입력으로 사용할 속성 고르기")

all_kor_features = ["나이", "평균 혈당", "체질량지수", "고혈압", "심장병"]
default_kor_features = ["나이", "평균 혈당", "고혈압", "심장병"]

selected_kor = st.multiselect(
    label="모델이 학습할 입력 속성을 골라 보세요 (우리말 이름)",
    options=all_kor_features,
    default=default_kor_features
)

if len(selected_kor) < 2:
    st.warning("⚠️ 입력 속성을 두 개 이상 골라 주세요. 두 개보다 적게 고르면 진행을 멈춥니다.")
    st.stop()

selected_cols = [KOR_TO_COL[k] for k in selected_kor]

# ---------------------------------------------------------
# 4. 데이터 정렬, 분할 (10명씩 묶어 앞 3명 테스트용 고정) 및 결측치·크기 맞추기
# ---------------------------------------------------------
df_sorted = df.sort_values("id").reset_index(drop=True)

test_mask = (df_sorted.index % 10) < 3
train_df = df_sorted[~test_mask].copy()
test_df = df_sorted[test_mask].copy()

X_train = train_df[selected_cols].copy()
y_train = train_df["stroke"].copy()
X_test = test_df[selected_cols].copy()
y_test = test_df["stroke"].copy()

if "bmi" in selected_cols:
    train_bmi_median = X_train["bmi"].median()
    X_train["bmi"] = X_train["bmi"].fillna(train_bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(train_bmi_median)
    st.caption(f"💡 **체질량지수**가 선택되어 빈 값을 훈련용 데이터의 중앙값(**{train_bmi_median:.2f}**)으로 채웠습니다.")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 5. 모델 학습 및 예측 ('분류 모델' 페이지와 동일)
# ---------------------------------------------------------
# 1) 로지스틱 회귀(확률로 답하는 모델)
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

# 2) 의사결정트리(질문으로 답하는 모델)
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

# 3) 훈련용에서 많은 쪽으로만 답하는 모델
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train_scaled, y_train)
dummy_preds = dummy_model.predict(X_test_scaled)

st.divider()

# ---------------------------------------------------------
# 6. 혼동행렬 계산 및 2x2 표(합계 포함) 표시
# ---------------------------------------------------------
st.subheader("2. 두 모델의 혼동행렬 (행: 실제, 열: 예측)")

def get_confusion_counts(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fn, fp, tn

def make_cm_table(tp, fn, fp, tn):
    row_pos_sum = tp + fn
    row_neg_sum = fp + tn
    col_pos_sum = tp + fp
    col_neg_sum = fn + tn
    total_sum = row_pos_sum + row_neg_sum

    return pd.DataFrame({
        "실제 \\ 예측": ["실제 뇌졸중 (1, 양성)", "실제 아님 (0, 음성)", "열 합계"],
        "예측 뇌졸중 (1, 양성)": [
            f"TP: {tp:,}명",
            f"FP: {fp:,}명",
            f"{col_pos_sum:,}명"
        ],
        "예측 아님 (0, 음성)": [
            f"FN: {fn:,}명",
            f"TN: {tn:,}명",
            f"{col_neg_sum:,}명"
        ],
        "행 합계": [
            f"{row_pos_sum:,}명",
            f"{row_neg_sum:,}명",
            f"{total_sum:,}명"
        ]
    })

def make_cm_plotly(tp, fn, fp, tn, title_text):
    z_vals = [[tn, fp], [fn, tp]]
    text_vals = [
        [f"FP (헛짚음)<br>{fp:,}명", f"TN (정답)<br>{tn:,}명"],
        [f"TP (정답)<br>{tp:,}명", f"FN (놓침)<br>{fn:,}명"]
    ]
    # 위에서부터 실제 뇌졸중(1), 실제 아님(0) 순서가 되도록 구성
    fig = go.Figure(
        data=go.Heatmap(
            z=[[fp, tn], [tp, fn]],
            x=["예측 뇌졸중 (1)", "예측 아님 (0)"],
            y=["실제 아님 (0)", "실제 뇌졸중 (1)"],
            text=text_vals,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False
        )
    )
    fig.update_layout(
        title=title_text,
        xaxis_title="예측",
        yaxis_title="실제",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

lr_tp, lr_fn, lr_fp, lr_tn = get_confusion_counts(y_test, lr_preds)
dt_tp, dt_fn, dt_fp, dt_tn = get_confusion_counts(y_test, dt_preds)
dm_tp, dm_fn, dm_fp, dm_tn = get_confusion_counts(y_test, dummy_preds)

cm_col1, cm_col2 = st.columns(2)

with cm_col1:
    st.markdown("#### 📌 로지스틱 회귀(확률로 답하는 모델)")
    st.dataframe(make_cm_table(lr_tp, lr_fn, lr_fp, lr_tn), hide_index=True, use_container_width=True)
    st.plotly_chart(
        make_cm_plotly(lr_tp, lr_fn, lr_fp, lr_tn, "로지스틱 회귀(확률로 답하는 모델) 2×2 칸 그림"),
        use_container_width=True
    )

with cm_col2:
    st.markdown("#### 📌 의사결정트리(질문으로 답하는 모델)")
    st.dataframe(make_cm_table(dt_tp, dt_fn, dt_fp, dt_tn), hide_index=True, use_container_width=True)
    st.plotly_chart(
        make_cm_plotly(dt_tp, dt_fn, dt_fp, dt_tn, "의사결정트리(질문으로 답하는 모델) 2×2 칸 그림"),
        use_container_width=True
    )

st.divider()

# ---------------------------------------------------------
# 7. 정확도·재현율·정밀도·F1 비교 표
# ---------------------------------------------------------
st.subheader("3. 모델별 평가 지표 비교 (정확도 · 재현율 · 정밀도 · F1)")

total_test_cnt = len(y_test)
actual_pos_cnt = int((y_test == 1).sum())
st.write(f"테스트용 사람 수는 모두 **{total_test_cnt:,}명**이고, 그중 실제 뇌졸중인 사람 수는 **{actual_pos_cnt:,}명**입니다.")

def calc_metrics_row(model_name, tp, fn, fp, tn):
    total = tp + fn + fp + tn
    acc = (tp + tn) / total if total > 0 else 0.0

    # 재현율 분모: 실제 뇌졸중인 사람 수 (tp + fn)
    actual_pos = tp + fn
    recall = tp / actual_pos if actual_pos > 0 else 0.0

    # 정밀도 분모: 뇌졸중이라 예측한 사람 수 (tp + fp)
    pred_pos = tp + fp
    if pred_pos == 0:
        precision_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"
    else:
        precision = tp / pred_pos
        precision_str = f"{precision:.4f} ({precision * 100:.2f}%)"
        if (precision + recall) == 0:
            f1_str = "0.0000"
        else:
            f1 = (2 * precision * recall) / (precision + recall)
            f1_str = f"{f1:.4f}"

    return {
        "모델": model_name,
        "정확도": f"{acc:.4f} ({acc * 100:.2f}%)",
        "재현율": f"{recall:.4f} ({recall * 100:.2f}%)",
        "정밀도": precision_str,
        "F1": f1_str
    }

metrics_rows = [
    calc_metrics_row("로지스틱 회귀(확률로 답하는 모델)", lr_tp, lr_fn, lr_fp, lr_tn),
    calc_metrics_row("의사결정트리(질문으로 답하는 모델)", dt_tp, dt_fn, dt_fp, dt_tn),
    calc_metrics_row("훈련용에서 많은 쪽으로만 답하는 모델", dm_tp, dm_fn, dm_fp, dm_tn)
]

metrics_df = pd.DataFrame(metrics_rows)
st.dataframe(metrics_df, hide_index=True, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# 8. 놓친 환자(FN)와 헛짚은 사람(FP) 목록 나란히 보기
# ---------------------------------------------------------
st.subheader("4. 놓친 환자(FN)와 헛짚은 사람(FP) 명단 살펴보기")

model_choice = st.radio(
    label="어느 모델의 목록을 볼지 골라 보세요",
    options=[
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)"
    ],
    horizontal=True
)

if model_choice == "로지스틱 회귀(확률로 답하는 모델)":
    chosen_preds = lr_preds
    chosen_fn_cnt = lr_fn
    chosen_fp_cnt = lr_fp
else:
    chosen_preds = dt_preds
    chosen_fn_cnt = dt_fn
    chosen_fp_cnt = dt_fp

# 테스트 데이터에 예측 결과 결합 후 우리말 표시용 표 구성
inspect_df = test_df.copy()
inspect_df["pred"] = chosen_preds

def format_person_list(sub_df):
    sorted_sub = sub_df.sort_values(by="age", ascending=False).copy()
    display_df = pd.DataFrame({
        "번호": sorted_sub["id"],
        "나이": sorted_sub["age"],
        "평균 혈당": sorted_sub["avg_glucose_level"].round(2),
        "고혈압 여부": sorted_sub["hypertension"].map({0: "없음 (0)", 1: "있음 (1)"}),
        "심장병 여부": sorted_sub["heart_disease"].map({0: "없음 (0)", 1: "있음 (1)"})
    })
    return display_df

# 왼쪽: 실제 뇌졸중(1)인데 아니라고(0) 예측한 사람 (놓친 환자, FN)
fn_raw_df = inspect_df[(inspect_df["stroke"] == 1) & (inspect_df["pred"] == 0)]
fn_display_df = format_person_list(fn_raw_df)
fn_total_len = len(fn_display_df)

# 오른쪽: 실제 아닌데(0) 뇌졸중이라고(1) 예측한 사람 (헛짚은 사람, FP)
fp_raw_df = inspect_df[(inspect_df["stroke"] == 0) & (inspect_df["pred"] == 1)]
fp_display_df = format_person_list(fp_raw_df)
fp_total_len = len(fp_display_df)

list_col1, list_col2 = st.columns(2)

with list_col1:
    if fn_total_len > 20:
        st.markdown(f"#### 🚨 실제 뇌졸중인데 아니라고 예측한 사람 (놓친 환자) - 전체 {fn_total_len:,}명 (앞 20줄 표시)")
    else:
        st.markdown(f"#### 🚨 실제 뇌졸중인데 아니라고 예측한 사람 (놓친 환자) - 전체 {fn_total_len:,}명")
    st.dataframe(fn_display_df.head(20), hide_index=True, use_container_width=True)

with list_col2:
    if fp_total_len > 20:
        st.markdown(f"#### ⚠️ 아닌데 뇌졸중이라고 예측한 사람 (헛짚은 사람) - 전체 {fp_total_len:,}명 (앞 20줄 표시)")
    else:
        st.markdown(f"#### ⚠️ 아닌데 뇌졸중이라고 예측한 사람 (헛짚은 사람) - 전체 {fp_total_len:,}명")
    st.dataframe(fp_display_df.head(20), hide_index=True, use_container_width=True)

st.info(
    f"💡 왼쪽 '놓친 환자' 목록의 전체 인원수(**{fn_total_len:,}명**)는 위 혼동행렬의 **FN 칸({chosen_fn_cnt:,}명)**과 같고, "
    f"오른쪽 '헛짚은 사람' 목록의 전체 인원수(**{fp_total_len:,}명**)는 혼동행렬의 **FP 칸({chosen_fp_cnt:,}명)**과 같습니다."
)
