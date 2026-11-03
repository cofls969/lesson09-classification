import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# 1. 브라우저 탭 제목과 아이콘 설정
st.set_page_config(
    page_title="분류 모델 - 뇌졸중 예측 실습실",
    page_icon="🤖",
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

st.title("🤖 분류 모델 학습 및 비교")
st.markdown("환자의 건강 속성을 바탕으로 **로지스틱 회귀(확률로 답하는 모델)**와 **의사결정트리(질문으로 답하는 모델)**를 학습시키고 비교해 봅니다.")
st.divider()

# ---------------------------------------------------------
# 3. 입력 속성 선택 (처음에는 체질량지수를 뺀 넷 선택)
# ---------------------------------------------------------
st.subheader("1. 입력으로 사용할 속성 고르기")

all_kor_features = ["나이", "평균 혈당", "체질량지수", "고혈압", "심장병"]
default_kor_features = ["나이", "평균 혈당", "고혈압", "심장병"]

selected_kor = st.multiselect(
    label="모델이 학습할 입력 속성을 골라 보세요 (우리말 이름)",
    options=all_kor_features,
    default=default_kor_features
)

# 두 개보다 적게 고르면 안내문을 표시하고 멈춤
if len(selected_kor) < 2:
    st.warning("⚠️ 입력 속성을 두 개 이상 골라 주세요. 두 개보다 적게 고르면 비교 그림을 그릴 수 없어 진행을 멈춥니다.")
    st.stop()

selected_cols = [KOR_TO_COL[k] for k in selected_kor]

# ---------------------------------------------------------
# 4. 데이터 정렬, 분할 (10명씩 묶어 앞 3명 테스트용 고정) 및 결측치·크기 맞추기
# ---------------------------------------------------------
# 번호(id) 순으로 정렬 후 인덱스 재설정
df_sorted = df.sort_values("id").reset_index(drop=True)

# 10명씩 묶어 각 묶음의 앞 3명(나머지가 0, 1, 2)을 테스트용(1,533명), 나머지 7명을 훈련용(3,577명)으로 고정
test_mask = (df_sorted.index % 10) < 3
train_df = df_sorted[~test_mask].copy()
test_df = df_sorted[test_mask].copy()

X_train = train_df[selected_cols].copy()
y_train = train_df["stroke"].copy()
X_test = test_df[selected_cols].copy()
y_test = test_df["stroke"].copy()

# 체질량지수(bmi)가 골라진 경우에만 훈련용의 중앙값으로 빈 값 채우기
if "bmi" in selected_cols:
    train_bmi_median = X_train["bmi"].median()
    X_train["bmi"] = X_train["bmi"].fillna(train_bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(train_bmi_median)
    st.caption(f"💡 **체질량지수**가 선택되어 빈 값을 훈련용 데이터의 중앙값(**{train_bmi_median:.2f}**)으로 채웠습니다. (테스트용 데이터: **{len(test_df):,}명**, 훈련용 데이터: **{len(train_df):,}명**)")
else:
    st.caption(f"💡 번호(id) 순으로 정렬해 10명씩 묶은 뒤 앞 3명씩 뽑아 테스트용 데이터 **{len(test_df):,}명**, 나머지 7명씩 뽑아 훈련용 데이터 **{len(train_df):,}명**으로 나누었습니다.")

# 훈련용 데이터로만 크기 맞추기(StandardScaler) 학습
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 5. 모델 학습 및 정확도 계산
# ---------------------------------------------------------
# 1) 입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train_scaled, y_train)
dummy_train_acc = accuracy_score(y_train, dummy_model.predict(X_train_scaled)) * 100
dummy_test_acc = accuracy_score(y_test, dummy_model.predict(X_test_scaled)) * 100

# 2) 로지스틱 회귀(확률로 답하는 모델)
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_train_acc = accuracy_score(y_train, lr_model.predict(X_train_scaled)) * 100
lr_test_acc = accuracy_score(y_test, lr_model.predict(X_test_scaled)) * 100

# 3) 의사결정트리(질문으로 답하는 모델) - 질문 3번까지, 마지막 마디 최소 5명, 난수 고정
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_train_acc = accuracy_score(y_train, dt_model.predict(X_train)) * 100
dt_test_acc = accuracy_score(y_test, dt_model.predict(X_test)) * 100

st.divider()

# ---------------------------------------------------------
# 6. 세 모델의 정확도 큰 숫자 카드 표시
# ---------------------------------------------------------
st.subheader("2. 모델별 정확도 비교")

card1, card2, card3 = st.columns(3)

with card1:
    st.metric(
        label="로지스틱 회귀(확률로 답하는 모델)",
        value=f"{lr_test_acc:.2f}%"
    )
    st.caption(f"훈련 정확도: {lr_train_acc:.2f}%  |  테스트 정확도: {lr_test_acc:.2f}%")

with card2:
    st.metric(
        label="의사결정트리(질문으로 답하는 모델)",
        value=f"{dt_test_acc:.2f}%"
    )
    st.caption(f"훈련 정확도: {dt_train_acc:.2f}%  |  테스트 정확도: {dt_test_acc:.2f}%")

with card3:
    st.metric(
        label="입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델",
        value=f"{dummy_test_acc:.2f}%"
    )
    st.caption(f"훈련 정확도: {dummy_train_acc:.2f}%  |  테스트 정확도: {dummy_test_acc:.2f}%")

st.divider()

# ---------------------------------------------------------
# 7. 두 축 선택 및 모델 경계(0.5 확률 선 + 의사결정트리 칸)와 산점도 그리기
# ---------------------------------------------------------
st.subheader("3. 두 속성으로 본 테스트 데이터와 모델의 판단 경계")

axis_col1, axis_col2 = st.columns(2)
with axis_col1:
    x_kor = st.selectbox("가로축으로 삼을 속성", options=selected_kor, index=0)
with axis_col2:
    remaining_kor = [k for k in selected_kor if k != x_kor]
    y_kor = st.selectbox("세로축으로 삼을 속성", options=remaining_kor, index=0)

x_col = KOR_TO_COL[x_kor]
y_col = KOR_TO_COL[y_kor]

# 두 축이 아닌 속성은 테스트 데이터의 중앙값으로 고정
other_cols = [c for c in selected_cols if c not in (x_col, y_col)]
fixed_medians = {}
fixed_text_parts = []
for c in other_cols:
    med_val = float(X_test[c].median())
    fixed_medians[c] = med_val
    if med_val.is_integer():
        fixed_text_parts.append(f"**{COL_TO_KOR[c]}** = {int(med_val)}")
    else:
        fixed_text_parts.append(f"**{COL_TO_KOR[c]}** = {med_val:.2f}")

if fixed_text_parts:
    st.info("📌 가로축·세로축이 아닌 속성은 테스트 데이터의 중앙값으로 고정해 계산했습니다: " + ", ".join(fixed_text_parts))
else:
    st.info("📌 선택된 속성이 두 개뿐이므로 고정된 다른 속성이 없습니다.")

# 그림 범위 설정
x_min, x_max = float(X_test[x_col].min()), float(X_test[x_col].max())
y_min, y_max = float(X_test[y_col].min()), float(X_test[y_col].max())
x_pad = (x_max - x_min) * 0.08 if x_max > x_min else 1.0
y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 1.0
plot_x_min, plot_x_max = x_min - x_pad, x_max + x_pad
plot_y_min, plot_y_max = y_min - y_pad, y_max + y_pad

# 의사결정트리 모델이 나눈 칸을 옅은 색으로 칠하기 위한 격자 생성
grid_size = 120
gx_step = (plot_x_max - plot_x_min) / (grid_size - 1)
gy_step = (plot_y_max - plot_y_min) / (grid_size - 1)
gx_vals = [plot_x_min + i * gx_step for i in range(grid_size)]
gy_vals = [plot_y_min + j * gy_step for j in range(grid_size)]

grid_rows = []
for y_v in gy_vals:
    for x_v in gx_vals:
        row = {}
        for c in selected_cols:
            if c == x_col:
                row[c] = x_v
            elif c == y_col:
                row[c] = y_v
            else:
                row[c] = fixed_medians[c]
        grid_rows.append(row)

grid_df = pd.DataFrame(grid_rows)[selected_cols]
dt_grid_preds = dt_model.predict(grid_df)

# 격자 예측값을 2차원 리스트로 변환
z_grid = []
for j in range(grid_size):
    z_grid.append(dt_grid_preds[j * grid_size : (j + 1) * grid_size].tolist())

fig = go.Figure()

# 1) 의사결정트리가 나눈 칸 (옅은 배경색: 0=아님은 옅은 파랑, 1=뇌졸중은 옅은 빨강)
fig.add_trace(
    go.Heatmap(
        x=gx_vals,
        y=gy_vals,
        z=z_grid,
        zmin=0,
        zmax=1,
        colorscale=[
            [0.0, "rgba(114, 183, 178, 0.22)"],
            [0.5, "rgba(114, 183, 178, 0.22)"],
            [0.5, "rgba(228, 87, 86, 0.25)"],
            [1.0, "rgba(228, 87, 86, 0.25)"]
        ],
        showscale=False,
        hoverinfo="skip",
        name="의사결정트리 영역"
    )
)

# 의사결정트리 분할 경계선도 점선으로 함께 표시 (구분된 칸이 명확히 보이도록)
tree_obj = dt_model.tree_
for node_idx in range(tree_obj.node_count):
    feat_idx = tree_obj.feature[node_idx]
    if feat_idx != _tree.TREE_UNDEFINED:
        feat_col = selected_cols[feat_idx]
        thresh = tree_obj.threshold[node_idx]
        if feat_col == x_col and plot_x_min <= thresh <= plot_x_max:
            fig.add_vline(x=thresh, line_dash="dot", line_color="rgba(80, 80, 80, 0.45)", line_width=1)
        elif feat_col == y_col and plot_y_min <= thresh <= plot_y_max:
            fig.add_hline(y=thresh, line_dash="dot", line_color="rgba(80, 80, 80, 0.45)", line_width=1)

# 2) 테스트 데이터 점 찍기 (실제 뇌졸중 여부로 색 분리 및 범례 표시)
test_neg_mask = (y_test == 0)
test_pos_mask = (y_test == 1)

fig.add_trace(
    go.Scatter(
        x=X_test.loc[test_neg_mask, x_col],
        y=X_test.loc[test_neg_mask, y_col],
        mode="markers",
        name="아님 (stroke=0)",
        marker=dict(color="#4C78A8", size=7, opacity=0.65, line=dict(width=0.5, color="white"))
    )
)

fig.add_trace(
    go.Scatter(
        x=X_test.loc[test_pos_mask, x_col],
        y=X_test.loc[test_pos_mask, y_col],
        mode="markers",
        name="뇌졸중 (stroke=1, 양성)",
        marker=dict(color="#E45756", size=9, opacity=0.9, symbol="circle", line=dict(width=1, color="black"))
    )
)

# 3) 로지스틱 회귀(확률로 답하는 모델)가 0.5 확률(결정함수 값 = 0)로 가르는 선 계산
# w_x * ((x - mu_x)/s_x) + w_y * ((y - mu_y)/s_y) + sum_other + b = 0
coefs = lr_model.coef_[0]
intercept = lr_model.intercept_[0]
means = scaler.mean_
scales = scaler.scale_

idx_x = selected_cols.index(x_col)
idx_y = selected_cols.index(y_col)

a_x = coefs[idx_x] / scales[idx_x]
a_y = coefs[idx_y] / scales[idx_y]

const_term = intercept - (coefs[idx_x] * means[idx_x] / scales[idx_x]) - (coefs[idx_y] * means[idx_y] / scales[idx_y])
for i, c in enumerate(selected_cols):
    if c not in (x_col, y_col):
        const_term += coefs[i] * ((fixed_medians[c] - means[i]) / scales[i])

# 그림 영역 안을 지나는 선분 끝점 구하기
line_points = []
if abs(a_y) > 1e-12:
    for xv in [plot_x_min, plot_x_max]:
        yv = -(a_x * xv + const_term) / a_y
        if plot_y_min <= yv <= plot_y_max:
            line_points.append((xv, yv))
if abs(a_x) > 1e-12:
    for yv in [plot_y_min, plot_y_max]:
        xv = -(a_y * yv + const_term) / a_x
        if plot_x_min <= xv <= plot_x_max:
            line_points.append((xv, yv))

# 중복 점 제거
unique_pts = []
for pt in line_points:
    if not any(abs(pt[0] - upt[0]) < 1e-7 and abs(pt[1] - upt[1]) < 1e-7 for upt in unique_pts):
        unique_pts.append(pt)

line_inside = len(unique_pts) >= 2

if line_inside:
    unique_pts.sort(key=lambda item: item[0])
    lx = [unique_pts[0][0], unique_pts[-1][0]]
    ly = [unique_pts[0][1], unique_pts[-1][1]]
    fig.add_trace(
        go.Scatter(
            x=lx,
            y=ly,
            mode="lines",
            name="로지스틱 회귀 0.5 경계선",
            line=dict(color="#111111", width=3, dash="dash")
        )
    )

fig.update_layout(
    title=f"{x_kor} · {y_kor} 산점도 및 모델 경계 (배경 칸: 의사결정트리 영역)",
    xaxis_title=x_kor,
    yaxis_title=y_kor,
    xaxis=dict(range=[plot_x_min, plot_x_max]),
    yaxis=dict(range=[plot_y_min, plot_y_max]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=560
)

st.plotly_chart(fig, use_container_width=True)

if not line_inside:
    st.warning("ℹ️ **안내:** 로지스틱 회귀(확률로 답하는 모델)가 확률 **0.5**로 가르는 선은 현재 그림 범위 밖에 있어 화면에 나타나지 않습니다. (뇌졸중 양성 비율이 낮아 0.5가 되는 경계가 데이터 범위를 크게 벗어나 있기 때문입니다.)")
else:
    st.success("ℹ️ 로지스틱 회귀(확률로 답하는 모델)가 확률 **0.5**로 가르는 경계선이 검은색 파선으로 표시되었습니다.")

st.divider()

# ---------------------------------------------------------
# 8. 의사결정트리 모델의 가지 그림 (st.graphviz_chart + DOT 문자열)
# ---------------------------------------------------------
st.subheader("4. 의사결정트리(질문으로 답하는 모델)의 가지 그림")
st.caption("마디마다 훈련용 사람 수, 실제 뇌졸중 사람 수와 비율을 적었습니다. 왼쪽 가지는 **예(조건 만족)**, 오른쪽 가지는 **아니요(조건 불만족)**이며, 마지막 답을 내는 마디는 예측 결과에 따라 색을 달리했습니다.")

def build_tree_dot(dt_clf, feature_cols):
    tree_ = dt_clf.tree_
    lines = [
        "digraph DecisionTree {",
        '    graph [rankdir=TB, fontname="Malgun Gothic, Apple SD Gothic Neo, sans-serif"];',
        '    node [shape=box, style="rounded,filled", fontname="Malgun Gothic, Apple SD Gothic Neo, sans-serif", fontsize=11, margin="0.2,0.12"];',
        '    edge [fontname="Malgun Gothic, Apple SD Gothic Neo, sans-serif", fontsize=11];'
    ]

    leaf_total = 0
    leaf_neg_count = 0
    asked_kor_features = []

    for i in range(tree_.node_count):
        left_child = tree_.children_left[i]
        right_child = tree_.children_right[i]
        is_leaf = (left_child == right_child)

        # tree_.value[i][0]에는 각 클래스별 샘플 수(가중치가 없으므로 실제 훈련용 사람 수)가 들어 있음
        counts = tree_.value[i][0]
        neg_cnt = int(round(counts[0]))
        pos_cnt = int(round(counts[1])) if len(counts) > 1 else 0
        total_cnt = neg_cnt + pos_cnt
        pos_ratio = (pos_cnt / total_cnt * 100.0) if total_cnt > 0 else 0.0

        stats_text = f"훈련용 사람 수: {total_cnt:,}명\\n실제 뇌졸중: {pos_cnt:,}명 ({pos_ratio:.1f}%)"

        if is_leaf:
            leaf_total += 1
            pred_class = 1 if pos_cnt > neg_cnt else 0
            if pred_class == 0:
                leaf_neg_count += 1
                ans_label = "답: 아님 (0)"
                fill_color = "#D9EDF7"   # 아님: 하늘색 계열
                border_color = "#31708F"
            else:
                ans_label = "답: 뇌졸중 (1)"
                fill_color = "#F2DEDE"   # 뇌졸중: 연분홍/빨강 계열
                border_color = "#A94442"

            node_label = f"{ans_label}\\n({stats_text})"
            lines.append(f'    node_{i} [label="{node_label}", fillcolor="{fill_color}", color="{border_color}", penwidth=1.8];')
        else:
            feat_idx = tree_.feature[i]
            col_name = feature_cols[feat_idx]
            kor_name = COL_TO_KOR[col_name]
            if kor_name not in asked_kor_features:
                asked_kor_features.append(kor_name)

            thresh = tree_.threshold[i]
            if col_name in ("hypertension", "heart_disease"):
                question_text = f"{kor_name} <= {thresh:.1f} (없음인가요?)"
            else:
                question_text = f"{kor_name} <= {thresh:.2f} 인가요?"

            node_label = f"질문: {question_text}\\n({stats_text})"
            lines.append(f'    node_{i} [label="{node_label}", fillcolor="#FFF9DB", color="#B58900"];')
            lines.append(f'    node_{i} -> node_{left_child} [label=" 예", color="#2B8A3E", fontcolor="#2B8A3E"];')
            lines.append(f'    node_{i} -> node_{right_child} [label=" 아니요", color="#C92A2A", fontcolor="#C92A2A"];')

    lines.append("}")
    return "\n".join(lines), leaf_total, leaf_neg_count, asked_kor_features

dot_code, total_leaves, neg_leaves, used_features_kor = build_tree_dot(dt_model, selected_cols)

st.graphviz_chart(dot_code, use_container_width=True)

# 그 아래 한 줄씩 요약 출력
used_features_str = ", ".join(used_features_kor) if used_features_kor else "없음"
st.write(f"- 더 묻지 않고 답을 내는 마지막 마디는 모두 **{total_leaves}칸**이고, 그중 **{neg_leaves}칸**이 **'아님'**이라고 답합니다.")
st.write(f"- 고른 속성 가운데 이 나무가 실제로 물은 속성은 **{used_features_str}**입니다.")
