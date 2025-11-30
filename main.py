import pandas as pd
import numpy as np
import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from db import get_connection

# FastAPI 초기화 + CORS 설정
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://3.37.90.119:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
model = joblib.load("exercise_recommender.pkl")
encoder = joblib.load("label_encoder.pkl")

# 영상 매핑 로드
video_df = pd.read_csv("exercise_video_mapping.csv")

video_map = {
    row["운동명"]: {
        "url": None if pd.isna(row.get("영상URL")) else row.get("영상URL"),
        "title": row.get("운동명"),
        "thumb": "/images/icon/no_video.png" if pd.isna(row.get("썸네일")) else row.get("썸네일"),
    }
    for _, row in video_df.iterrows()
}

# -------------------------------------------------------------
# 헬퍼 함수
# -------------------------------------------------------------
def safe_float(v):
    """None/NaN → 0 변환"""
    if v is None:
        return 0
    if isinstance(v, float) and np.isnan(v):
        return 0
    return v


def get_video(ex):
    """운동명 → 영상 매핑"""
    info = video_map.get(ex)
    if not info:
        return {
            "url": None,
            "title": ex,
            "thumb": "/images/icon/no_video.png"
        }
    return info


ROUTINE = {
    "하체근력": {
        "준비": ["전신 스트레칭", "무릎 돌리기", "고관절 스트레칭"],
        "본": ["스쿼트", "런지", "힙 브릿지"],
        "정리": ["하체 스트레칭", "이완 호흡"],
    },
    "상체근력": {
        "준비": ["팔 스트레칭", "어깨 스트레칭", "목 스트레칭"],
        "본": ["푸시업", "덤벨 로우", "밴드 로우"],
        "정리": ["상체 스트레칭", "전신 스트레칭"],
    },
    "코어": {
        "준비": ["브릿지", "고양이-소", "전신 스트레칭"],
        "본": ["플랭크", "크런치", "사이드 플랭크"],
        "정리": ["이완 호흡", "명상"],
    },
    "전신근력": {
        "준비": ["전신 스트레칭", "무릎 돌리기"],
        "본": ["버피", "스쿼트", "푸시업", "점핑잭"],
        "정리": ["상체 스트레칭", "하체 스트레칭"],
    },
    "유산소-걷기": {
        "준비": ["전신 스트레칭", "무릎 돌리기"],
        "본": ["빠르게 걷기"],
        "정리": ["하체 스트레칭", "이완 호흡"],
    },
    "유산소-달리기": {
        "준비": ["전신 스트레칭", "고관절 스트레칭"],
        "본": ["조깅", "인터벌 러닝"],
        "정리": ["하체 스트레칭", "이완 호흡"],
    },
    "유산소-자전거": {
        "준비": ["무릎 돌리기", "고관절 스트레칭"],
        "본": ["실내 자전거", "에르고 미터"],
        "정리": ["하체 스트레칭", "전신 스트레칭"],
    },
    "전신유산소": {
        "준비": ["전신 스트레칭", "팔 스트레칭"],
        "본": ["버피", "점핑잭"],
        "정리": ["이완 호흡", "명상"],
    },
    "요가필라테스": {
        "준비": ["전신 스트레칭", "전굴"],
        "본": ["요가 루틴", "전굴", "고양이-소"],
        "정리": ["비둘기 자세", "이완 호흡", "명상"],
    },
}



# feature 생성 함수
def build_features(age, gender, height, weight, bmi):
    age = safe_float(age)
    height = safe_float(height)
    weight = safe_float(weight)
    bmi = safe_float(bmi)

    group = (int(age) // 10) * 10
    F = 1 if gender == "F" else 0
    M = 1 if gender == "M" else 0

    return np.array([[group, age, F, M, height, weight, bmi, 20, 22, 10, 30]])


# API: 운동 추천
@app.get("/recommend/{user_id}")
def recommend_live(user_id: int):

    # DB에서 유저 정보 가져오기
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT AGE, GENDER, HEIGHT, CURRENT_WEIGHT, BMI
        FROM USERS
        WHERE USER_ID=%s
    """, (user_id,))

    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if not user:
        return {"error": "User not found"}

    # Feature 생성
    X = build_features(
        user["AGE"], user["GENDER"], user["HEIGHT"],
        user["CURRENT_WEIGHT"], user["BMI"],
    )

    # 모델 예측
    pred = model.predict(X)[0]
    category = encoder.inverse_transform([pred])[0]

    # "전신근력(난이도)" → "전신근력"
    base_category = category.split("(")[0]

    routine = ROUTINE.get(base_category, ROUTINE["전신근력"])

    # 랜덤 운동 선택
    prep = random.choice(routine["준비"])
    main = random.choice(routine["본"])
    cool = random.choice(routine["정리"])

    # 최종 반환(JSON)
    return {
        "user_id": user_id,
        "predicted_category": category,
        "routine": {
            "준비운동": {"name": prep, **get_video(prep)},
            "본운동": {"name": main, **get_video(main)},
            "정리운동": {"name": cool, **get_video(cool)},
        }
    }
