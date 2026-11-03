import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 1. 브라우저 탭 제목과 아이콘 설정
st.set_page_config(
    page_title="성능 높이기 - 뇌졸중 예측 실습실",
    page_icon="🚀",
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
COL_TO_KOR = {v: k for k, v in KOR_TO_COL.items()}

st.title("🚀 가중치 조절로 성능 높이기")
st.markdown("뇌졸중을 겪은 사람이 아주 드물어 생기는 불균형 문제를 해결하기 위해, **양쪽의 가중치를 맞추기 전과 맞춘 뒤(`class_weight='balanced'`)**의 결과를 한 화면에서 나란히 비교해 봅니다.")
st.divider()

# ---------------------------------------------------------
# 3. 입력 속성 선택 (처음에는 나이, 평균 혈당, 고혈압, 심장병 선택)
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
    st.warning("⚠️ 입력 속성을 두 개 이상 골라 주세요. 두 개보다 적게 고르면 비교 그림을 그릴 수 없어 진행을 멈춥니다.")
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

# 체질량지수(bmi)가 비어 있으면 훈련용 중앙값으로 채우기
if "bmi" in selected_cols:
    train_bmi_median = X_train["bmi"].median()
    X_train["bmi"] = X_train["bmi"].fillna(train_bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(train_bmi_median)
    st.caption(f"💡 **체질량지수**가 선택되어 빈 값을 훈련용 데이터의 중앙값(**{train_bmi_median:.2f}**)으로 채웠습니다. (테스트용: **{len(test_df):,}명**, 훈련용: **{len(train_df):,}명**)")
else:
    st.caption(f"💡 번호(id) 순으로 정렬해 10명씩 묶은 뒤 앞 3명씩 뽑아 테스트용 **{len(test_df):,}명**, 나머지 7명씩 뽑아 훈련용 **{len(train_df):,}명**으로 나누었습니다.")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 5. 가중치 맞추기 전·후 모델 학습 및 예측
# ---------------------------------------------------------
# 1) 로지스틱 회귀(확률로 답하는 모델) - 가중치 맞추기 전 / 후
lr_before = LogisticRegression(random_state=42)
lr_before.fit(X_train_scaled, y_train)
lr_pred_before = lr_before.predict(X_test_scaled)

lr_after = LogisticRegression(class_weight="balanced", random_state=42)
lr_after.fit(X_train_scaled, y_train)
lr_pred_after = lr_after.predict(X_test_scaled)

# 2) 의사결정트리(질문으로 답하는 모델) - 가중치 맞추기 전 / 후
dt_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_before.fit(X_train, y_train)
dt_pred_before = dt_before.predict(X_test)

dt_after = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=42)
dt_after.fit(X_train, y_train)
dt_pred_after = dt_after.predict(X_test)

st.divider()

# ---------------------------------------------------------
# 6. 두 모델마다 네 칸(TP·FN·FP·TN)과 네 지표를 전·후로 나란히 놓은 표
# ---------------------------------------------------------
st.subheader("2. 가중치 맞추기 전·후 결과 나란히 비교")

def calc_stats(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    total = tp + fn + fp + tn
    acc = (tp + tn) / total if total > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    pred_pos = tp + fp
    if pred_pos == 0:
        prec_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"
    else:
        prec = tp / pred_pos
        prec_str = f"{prec:.4f} ({prec * 100:.2f}%)"
        if (prec + recall) == 0:
            f1_str = "0.0000"
        else:
            f1 = (2 * prec * recall) / (prec + recall)
            f1_str = f"{f1:.4f}"

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "acc_str": f"{acc:.4f} ({acc * 100:.2f}%)",
        "recall_str": f"{recall:.4f} ({recall * 100:.2f}%)",
        "prec_str": prec_str,
        "f1_str": f1_str
    }

def make_before_after_df(st_before, st_after):
    return pd.DataFrame({
        "항목 (네 칸 · 네 지표)": [
            "TP (찾아낸 환자)",
            "FN (놓친 환자)",
            "FP (헛짚은 사람)",
            "TN (잘 가려낸 아님)",
            "정확도 (Accuracy)",
            "재현율 (Recall)",
            "정밀도 (Precision)",
            "F1 점수"
        ],
        "가중치 맞추기 전": [
            f"{st_before['tp']:,}명",
            f"{st_before['fn']:,}명",
            f"{st_before['fp']:,}명",
            f"{st_before['tn']:,}명",
            st_before["acc_str"],
            st_before["recall_str"],
            st_before["prec_str"],
            st_before["f1_str"]
        ],
        "가중치 맞춘 뒤": [
            f"{st_after['tp']:,}명",
            f"{st_after['fn']:,}명",
            f"{st_after['fp']:,}명",
            f"{st_after['tn']:,}명",
            st_after["acc_str"],
            st_after["recall_str"],
            st_after["prec_str"],
            st_after["f1_str"]
        ]
    })

lr_st_before = calc_stats(y_test, lr_pred_before)
lr_st_after = calc_stats(y_test, lr_pred_after)

dt_st_before = calc_stats(y_test, dt_pred_before)
dt_st_after = calc_stats(y_test, dt_pred_after)

col_lr, col_dt = st.columns(2)

with col_lr:
