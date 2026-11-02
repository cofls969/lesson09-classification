import streamlit as st
import pandas as pd
import os
import json

# 1. 브라우저 탭 제목과 아이콘 설정
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# 2. 기록 저장용 폴더 및 JSON 파일 경로 설정
RECORD_DIR = "내_기록"
RECORD_FILE = os.path.join(RECORD_DIR, "user_records.json")
os.makedirs(RECORD_DIR, exist_ok=True)

# JSON 불러오기 함수
def load_records(columns):
    if os.path.exists(RECORD_FILE):
        try:
            with open(RECORD_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 누락된 열이 있을 경우 빈 문자열로 보완
                if "meanings" not in data:
                    data["meanings"] = {col: "" for col in columns}
                else:
                    for col in columns:
                        if col not in data["meanings"]:
                            data["meanings"][col] = ""
                if "source" not in data:
                    data["source"] = ""
                return data
        except Exception:
            pass
    return {
        "meanings": {col: "" for col in columns},
        "source": ""
    }

# JSON 저장하기 함수
def save_records(data):
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 3. 데이터 불러오기 (캐싱 처리로 속도 향상)
DATA_URL = "https://raw.githubusercontent.com/cofls969/data/main/stroke.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    # 혹시 bmi 열에 'N/A' 같은 문자열이 섞여 있을 경우를 대비해 숫자로 변환 (빈 값은 NaN 처리)
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    return df

df = load_data()
columns = list(df.columns)
records = load_records(columns)

# 4. 화면 맨 위 앱 제목 표시
st.title("🧠 뇌졸중 예측 실습실")
st.markdown("환자의 건강 및 생활 습관 데이터를 살펴보고, 각 열의 의미와 출처를 직접 정리해 보는 데이터 소개 화면입니다.")
st.divider()

# 5. 큰 숫자 카드 네 개 (전체 사람 수, 열 개수, stroke가 1인 사람 수, 그 비율)
total_people = len(df)
total_cols = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())
stroke_ratio = (stroke_count / total_people) * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="전체 사람 수", value=f"{total_people:,}명")
with col2:
    st.metric(label="열 개수", value=f"{total_cols}개")
with col3:
    st.metric(label="뇌졸중(stroke=1) 사람 수", value=f"{stroke_count:,}명")
with col4:
    st.metric(label="뇌졸중 비율", value=f"{stroke_ratio:.2f}%")

st.divider()

# 6. 열 정보 요약 및 '우리말 뜻' 직접 입력 표
st.subheader("📋 데이터 열 정보 및 우리말 뜻 정리")
st.caption("💡 표의 **'우리말 뜻'** 칸을 더블 클릭하여 교재를 보고 내용을 직접 입력해 보세요. 입력 후 엔터(Enter)를 누르면 `내_기록` 폴더에 자동 저장됩니다.")

# 각 열의 값의 종류와 빈 값 개수 요약 생성
col_info_list = []
for col in columns:
    missing_cnt = int(df[col].isna().sum())
    nunique_cnt = df[col].nunique(dropna=True)
    
    # 값의 종류를 이해하기 쉽게 예시와 함께 표시
    if pd.api.types.is_numeric_dtype(df[col]):
        if nunique_cnt <= 5:
            unique_vals = ", ".join(map(str, sorted(df[col].dropna().unique())))
            val_type = f"숫자형 ({unique_vals})"
        else:
            val_type = f"연속형 숫자 ({df[col].min()} ~ {df[col].max()})"
    else:
        unique_vals = ", ".join(map(str, df[col].dropna().unique()[:5]))
        val_type = f"범주형 ({unique_vals})"
        
    col_info_list.append({
        "열 이름": col,
        "우리말 뜻": records["meanings"].get(col, ""),
        "값의 종류": val_type,
        "빈 값 개수": missing_cnt
    })

info_df = pd.DataFrame(col_info_list)

# 우리말 뜻 칸만 수정 가능하도록 설정된 데이터 에디터
edited_df = st.data_editor(
    info_df,
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름", disabled=True),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻 (직접 입력)",
            help="교재를 참고하여 각 열의 우리말 뜻을 적어주세요.",
            default=""
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류", disabled=True),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", disabled=True),
    },
    hide_index=True,
    use_container_width=True
)

# 표에서 수정된 '우리말 뜻'이 있으면 즉시 JSON 파일에 업데이트
updated_meanings = dict(zip(edited_df["열 이름"], edited_df["우리말 뜻"]))
if updated_meanings != records["meanings"]:
    records["meanings"] = updated_meanings
    save_records(records)
    st.toast("우리말 뜻이 자동 저장되었습니다!", icon="✅")

st.divider()

# 7. 데이터 처음 다섯 줄 그대로 보여주기
st.subheader("🔍 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# 8. 맨 아래 데이터 출처 적는 자리
st.subheader("📚 데이터 출처")
source_input = st.text_area(
    label="교재에 나와 있는 데이터 출처를 아래 칸에 직접 적어 보세요. (작성 후 칸 바깥을 클릭하거나 Ctrl+Enter를 누르면 자동 저장됩니다.)",
    value=records.get("source", ""),
    placeholder="여기에 데이터 출처를 입력하세요...",
    height=100
)

# 출처 내용이 변경되었으면 즉시 JSON 파일에 업데이트
if source_input != records.get("source", ""):
    records["source"] = source_input
    save_records(records)
    st.toast("데이터 출처가 자동 저장되었습니다!", icon="✅")
