import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="성능 높이기 - 뇌졸중 예측 실습실",
    page_icon="🚀",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/cofls969/data/main/stroke.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    return df


df = load_data()

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
    st.caption(f"💡 **체질량지수**가 선택되어 빈 값을 훈련용 데이터의 중앙값(**{train_bmi_median:.2f}**)으로 채웠습니다. (테스트용: **{len(test_df):,}명**, 훈련용: **{len(train_df):,}명**)")
else:
    st.caption(f"💡 번호(id) 순으로 정렬해 10명씩 묶은 뒤 앞 3명씩 뽑아 테스트용 **{len(test_df):,}명**, 나머지 7명씩 뽑아 훈련용 **{len(train_df):,}명**으로 나누었습니다.")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_before = LogisticRegression(random_state=42)
lr_before.fit(X_train_scaled, y_train)
lr_pred_before = lr_before.predict(X_test_scaled)

lr_after = LogisticRegression(class_weight="balanced", random_state=42)
lr_after.fit(X_train_scaled, y_train)
lr_pred_after = lr_after.predict(X_test_scaled)

dt_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_before.fit(X_train, y_train)
dt_pred_before = dt_before.predict(X_test)

dt_after = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=42)
dt_after.fit(X_train, y_train)
dt_pred_after = dt_after.predict(X_test)

st.divider()

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
    st.markdown("#### 📌 로지스틱 회귀(확률로 답하는 모델)")
    st.dataframe(make_before_after_df(lr_st_before, lr_st_after), hide_index=True, use_container_width=True)
    st.write(
        f"👉 **변화 요약:** 찾아낸 환자 수는 **{lr_st_before['tp']:,}명에서 {lr_st_after['tp']:,}명**으로, "
        f"놓친 환자 수는 **{lr_st_before['fn']:,}명에서 {lr_st_after['fn']:,}명**으로, "
        f"헛짚은 사람 수는 **{lr_st_before['fp']:,}명에서 {lr_st_after['fp']:,}명**으로 바뀌었습니다."
    )

with col_dt:
    st.markdown("#### 📌 의사결정트리(질문으로 답하는 모델)")
    st.dataframe(make_before_after_df(dt_st_before, dt_st_after), hide_index=True, use_container_width=True)
    st.write(
        f"👉 **변화 요약:** 찾아낸 환자 수는 **{dt_st_before['tp']:,}명에서 {dt_st_after['tp']:,}명**으로, "
        f"놓친 환자 수는 **{dt_st_before['fn']:,}명에서 {dt_st_after['fn']:,}명**으로, "
        f"헛짚은 사람 수는 **{dt_st_before['fp']:,}명에서 {dt_st_after['fp']:,}명**으로 바뀌었습니다."
    )

st.divider()

st.subheader("3. 두 속성 산점도와 가중치 전·후 0.5 경계선 비교")

ax_col1, ax_col2 = st.columns(2)
with ax_col1:
    x_kor_2d = st.selectbox("가로축으로 삼을 속성 (2차원)", options=selected_kor, index=0)
with ax_col2:
    rem_kor_2d = [k for k in selected_kor if k != x_kor_2d]
    y_kor_2d = st.selectbox("세로축으로 삼을 속성 (2차원)", options=rem_kor_2d, index=0)

x_col_2d = KOR_TO_COL[x_kor_2d]
y_col_2d = KOR_TO_COL[y_kor_2d]

other_cols_2d = [c for c in selected_cols if c not in (x_col_2d, y_col_2d)]
fixed_medians_2d = {}
fixed_text_2d = []
for c in other_cols_2d:
    med_v = float(X_test[c].median())
    fixed_medians_2d[c] = med_v
    if med_v.is_integer():
        fixed_text_2d.append(f"**{COL_TO_KOR[c]}** = {int(med_v)}")
    else:
        fixed_text_2d.append(f"**{COL_TO_KOR[c]}** = {med_v:.2f}")

if fixed_text_2d:
    st.caption("📌 가로축·세로축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했습니다: " + ", ".join(fixed_text_2d))

x_min, x_max = float(X_test[x_col_2d].min()), float(X_test[x_col_2d].max())
y_min, y_max = float(X_test[y_col_2d].min()), float(X_test[y_col_2d].max())
x_pad = (x_max - x_min) * 0.08 if x_max > x_min else 1.0
y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 1.0
plot_x_min, plot_x_max = x_min - x_pad, x_max + x_pad
plot_y_min, plot_y_max = y_min - y_pad, y_max + y_pad


def get_2d_boundary_points(lr_clf, x_c, y_c, fixed_dict, px_min, px_max, py_min, py_max):
    coefs = lr_clf.coef_[0]
    intercept = lr_clf.intercept_[0]
    means = scaler.mean_
    scales = scaler.scale_

    idx_x = selected_cols.index(x_c)
    idx_y = selected_cols.index(y_c)

    a_x = coefs[idx_x] / scales[idx_x]
    a_y = coefs[idx_y] / scales[idx_y]

    const_term = intercept - (coefs[idx_x] * means[idx_x] / scales[idx_x]) - (coefs[idx_y] * means[idx_y] / scales[idx_y])
    for i, c in enumerate(selected_cols):
        if c not in (x_c, y_c):
            const_term += coefs[i] * ((fixed_dict[c] - means[i]) / scales[i])

    pts = []
    if abs(a_y) > 1e-12:
        for xv in [px_min, px_max]:
            yv = -(a_x * xv + const_term) / a_y
            if py_min <= yv <= py_max:
                pts.append((xv, yv))
    if abs(a_x) > 1e-12:
        for yv in [py_min, py_max]:
            xv = -(a_y * yv + const_term) / a_x
            if px_min <= xv <= px_max:
                pts.append((xv, yv))

    uniq = []
    for pt in pts:
        if not any(abs(pt[0] - u[0]) < 1e-7 and abs(pt[1] - u[1]) < 1e-7 for u in uniq):
            uniq.append(pt)
    if len(uniq) >= 2:
        uniq.sort(key=lambda item: item[0])
        return [uniq[0][0], uniq[-1][0]], [uniq[0][1], uniq[-1][1]], True
    return [], [], False


fig_2d = go.Figure()

neg_mask = (y_test == 0)
pos_mask = (y_test == 1)

fig_2d.add_trace(
    go.Scatter(
        x=X_test.loc[neg_mask, x_col_2d],
        y=X_test.loc[neg_mask, y_col_2d],
        mode="markers",
        name="실제 아님 (stroke=0)",
        marker=dict(color="#4C78A8", size=7, opacity=0.6, line=dict(width=0.4, color="white"))
    )
)

fig_2d.add_trace(
    go.Scatter(
        x=X_test.loc[pos_mask, x_col_2d],
        y=X_test.loc[pos_mask, y_col_2d],
        mode="markers",
        name="실제 뇌졸중 (stroke=1, 양성)",
        marker=dict(color="#E45756", size=9, opacity=0.9, line=dict(width=1, color="black"))
    )
)

lx_bef, ly_bef, inside_bef = get_2d_boundary_points(
    lr_before, x_col_2d, y_col_2d, fixed_medians_2d, plot_x_min, plot_x_max, plot_y_min, plot_y_max
)
lx_aft, ly_aft, inside_aft = get_2d_boundary_points(
    lr_after, x_col_2d, y_col_2d, fixed_medians_2d, plot_x_min, plot_x_max, plot_y_min, plot_y_max
)

if inside_bef:
    fig_2d.add_trace(
        go.Scatter(
            x=lx_bef,
            y=ly_bef,
            mode="lines",
            name="가중치 맞추기 전 0.5 경계선",
            line=dict(color="#7F3C8D", width=3, dash="dot")
        )
    )

if inside_aft:
    fig_2d.add_trace(
        go.Scatter(
            x=lx_aft,
            y=ly_aft,
            mode="lines",
            name="가중치 맞춘 뒤 0.5 경계선",
            line=dict(color="#11A579", width=3.5, dash="solid")
        )
    )

fig_2d.update_layout(
    title=f"{x_kor_2d} · {y_kor_2d} 산점도와 가중치 전·후 0.5 경계선",
    xaxis_title=x_kor_2d,
    yaxis_title=y_kor_2d,
    xaxis=dict(range=[plot_x_min, plot_x_max]),
    yaxis=dict(range=[plot_y_min, plot_y_max]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=540
)

st.plotly_chart(fig_2d, use_container_width=True)

if not inside_bef and inside_aft:
    st.warning("ℹ️ **경계선 안내:** **가중치를 맞추기 전의 0.5 경계선**은 그림 범위 밖에 있어 보이지 않고, **가중치를 맞춘 뒤의 0.5 경계선**(초록색 실선)은 그림 안으로 들어와 표시되었습니다.")
elif inside_bef and not inside_aft:
    st.warning("ℹ️ **경계선 안내:** **가중치를 맞추기 전의 0.5 경계선**(보라색 점선)은 그림 안에 표시되었으나, **가중치를 맞춘 뒤의 0.5 경계선**은 그림 범위 밖에 있습니다.")
elif not inside_bef and not inside_aft:
    st.warning("ℹ️ **경계선 안내:** 가중치를 맞추기 전의 0.5 경계선과 맞춘 뒤의 0.5 경계선이 **모두 현재 그림 범위 밖에** 있습니다.")
else:
    st.success("ℹ️ **경계선 안내:** 가중치를 맞추기 전의 0.5 경계선(보라색 점선)과 맞춘 뒤의 0.5 경계선(초록색 실선)이 모두 그림 안에 표시되었습니다.")

st.divider()

st.subheader("4. 세 속성으로 본 3차원 산점도와 가중치 전·후 0.5 평면")

if len(selected_kor) < 3:
    st.info("ℹ️ 속성을 **셋 이상** 고르면 3차원 산점도와 전·후 0.5 평면을 마우스로 돌려 볼 수 있습니다. 위에서 속성을 3개 이상 골라 보세요.")
else:
    st.caption("💡 마우스로 그래프를 드래그하면 이리저리 돌려 볼 수 있고, 휠을 굴리면 확대·축소할 수 있습니다.")
    c3_1, c3_2, c3_3 = st.columns(3)
    with c3_1:
        x_kor_3d = st.selectbox("X축 속성 (3차원)", options=selected_kor, index=0)
    with c3_2:
        rem_y_3d = [k for k in selected_kor if k != x_kor_3d]
        y_kor_3d = st.selectbox("Y축 속성 (3차원)", options=rem_y_3d, index=0)
    with c3_3:
        rem_z_3d = [k for k in selected_kor if k not in (x_kor_3d, y_kor_3d)]
        z_kor_3d = st.selectbox("Z축 속성 (3차원)", options=rem_z_3d, index=0)

    x_c3 = KOR_TO_COL[x_kor_3d]
    y_c3 = KOR_TO_COL[y_kor_3d]
    z_c3 = KOR_TO_COL[z_kor_3d]

    other_cols_3d = [c for c in selected_cols if c not in (x_c3, y_c3, z_c3)]
    fixed_medians_3d = {}
    fixed_text_3d = []
    for c in other_cols_3d:
        med_v = float(X_test[c].median())
        fixed_medians_3d[c] = med_v
        if med_v.is_integer():
            fixed_text_3d.append(f"**{COL_TO_KOR[c]}** = {int(med_v)}")
        else:
            fixed_text_3d.append(f"**{COL_TO_KOR[c]}** = {med_v:.2f}")

    if fixed_text_3d:
        st.caption("📌 세 축이 아닌 나머지 속성은 테스트 데이터의 중앙값에 세워 두고 계산했습니다: " + ", ".join(fixed_text_3d))

    x3_min, x3_max = float(X_test[x_c3].min()), float(X_test[x_c3].max())
    y3_min, y3_max = float(X_test[y_c3].min()), float(X_test[y_c3].max())
    z3_min, z3_max = float(X_test[z_c3].min()), float(X_test[z_c3].max())
    z3_span = (z3_max - z3_min) if z3_max > z3_min else 1.0
    z3_plot_min = z3_min - z3_span * 0.25
    z3_plot_max = z3_max + z3_span * 0.25

    def build_3d_plane_surface(lr_clf, xc, yc, zc, fixed_dict, color_hex, name_str):
        coefs = lr_clf.coef_[0]
        intercept = lr_clf.intercept_[0]
        means = scaler.mean_
        scales = scaler.scale_

        ix = selected_cols.index(xc)
        iy = selected_cols.index(yc)
        iz = selected_cols.index(zc)

        ax = coefs[ix] / scales[ix]
        ay = coefs[iy] / scales[iy]
        az = coefs[iz] / scales[iz]

        const_t = (
            intercept
            - (coefs[ix] * means[ix] / scales[ix])
            - (coefs[iy] * means[iy] / scales[iy])
            - (coefs[iz] * means[iz] / scales[iz])
        )
        for i, c in enumerate(selected_cols):
            if c not in (xc, yc, zc):
                const_t += coefs[i] * ((fixed_dict[c] - means[i]) / scales[i])

        grid_n = 25
        xs = [x3_min + (x3_max - x3_min) * i / (grid_n - 1) for i in range(grid_n)]
        ys = [y3_min + (y3_max - y3_min) * j / (grid_n - 1) for j in range(grid_n)]

        if abs(az) > 1e-12:
            z_mat = []
            in_bounds_cnt = 0
            for yv in ys:
                row_z = []
                for xv in xs:
                    zv = -(ax * xv + ay * yv + const_t) / az
                    if z3_plot_min <= zv <= z3_plot_max:
                        in_bounds_cnt += 1
                    row_z.append(zv)
                z_mat.append(row_z)

            surface = go.Surface(
                x=xs,
                y=ys,
                z=z_mat,
                name=name_str,
                colorscale=[[0, color_hex], [1, color_hex]],
                opacity=0.45,
                showscale=False,
                showlegend=True
            )
            return surface, (in_bounds_cnt > 0)
        return None, False

    fig_3d = go.Figure()

    fig_3d.add_trace(
        go.Scatter3d(
            x=X_test.loc[neg_mask, x_c3],
            y=X_test.loc[neg_mask, y_c3],
            z=X_test.loc[neg_mask, z_c3],
            mode="markers",
            name="실제 아님 (stroke=0)",
            marker=dict(color="#4C78A8", size=3.5, opacity=0.55)
        )
    )

    fig_3d.add_trace(
        go.Scatter3d(
            x=X_test.loc[pos_mask, x_c3],
            y=X_test.loc[pos_mask, y_c3],
            z=X_test.loc[pos_mask, z_c3],
            mode="markers",
            name="실제 뇌졸중 (stroke=1, 양성)",
            marker=dict(color="#E45756", size=5.5, opacity=0.95, line=dict(width=1, color="black"))
        )
    )

    surf_bef, plane_in_bef = build_3d_plane_surface(
        lr_before, x_c3, y_c3, z_c3, fixed_medians_3d, "#8856A7", "가중치 맞추기 전 0.5 평면 (보라색)"
    )
    surf_aft, plane_in_aft = build_3d_plane_surface(
        lr_after, x_c3, y_c3, z_c3, fixed_medians_3d, "#2CA25F", "가중치 맞춘 뒤 0.5 평면 (초록색)"
    )

    if surf_bef is not None:
        fig_3d.add_trace(surf_bef)
    if surf_aft is not None:
        fig_3d.add_trace(surf_aft)

    fig_3d.update_layout(
        title=f"{x_kor_3d} · {y_kor_3d} · {z_kor_3d} 3차원 산점도와 가중치 전(보라색)·후(초록색) 0.5 평면",
        scene=dict(
            xaxis_title=x_kor_3d,
            yaxis_title=y_kor_3d,
            zaxis_title=z_kor_3d
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=680,
        margin=dict(l=10, r=10, t=50, b=10)
    )

    st.plotly_chart(fig_3d, use_container_width=True)

    if not plane_in_bef and plane_in_aft:
        st.caption("ℹ️ 3차원 공간에서 **가중치 맞추기 전 0.5 평면(보라색)**은 데이터 분포 영역에서 멀리 떨어져 있고, **가중치 맞춘 뒤 0.5 평면(초록색)**은 데이터 점들 사이를 가로지르고 있습니다.")
