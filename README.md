<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=240&text=Exercise%20AI%20Recommend%20API&fontAlign=50&fontAlignY=40&color=gradient&customColorList=0,2,5,10&fontSize=45&fontColor=ffffff&desc=Machine%20Learning%20Based%20Exercise%20Recommendation&descAlignY=60&descAlign=50" />
</p>

<p align="center">
  머신러닝 기반 운동 카테고리 분류 + 유튜브 영상 매핑 자동 추천 API  
  <br>(최종 모델: RandomOversampler + RandomForestClassifier)
</p>

---

# 📌 Table of Contents
- [1. 프로젝트 개요](#1-프로젝트-개요)
- [2. 사용 데이터](#2-사용-데이터)
- [3. 데이터 전처리 & 라벨링](#3-데이터-전처리--라벨링)
- [4. 데이터 불균형 문제 해결](#4-데이터-불균형-문제-해결)
- [5. 모델 구조](#5-모델-구조)
- [6. 최종 성능](#6-최종-성능)
- [7. FastAPI 연동 구조](#7-fastapi-연동-구조)
- [8. API 예시 응답](#8-api-예시-응답)
- [9. 기술 스택](#9-기술-스택)
- [10. 결론](#10-결론)

---

# 1. 프로젝트 개요

본 프로젝트는 공공데이터 기반 체력측정·운동처방 데이터를 이용하여  
**사용자에게 가장 적합한 운동 카테고리(총 9종)를 예측하는 AI 모델**을 구축하고  
FastAPI 기반 REST API로 서비스화한 프로젝트입니다.

예측된 카테고리를 기반으로  
**준비 → 본운동 → 정리운동 루틴을 자동 생성**하고,  
**유튜브 운동 영상 매핑**까지 제공하여  
사용자가 즉시 따라할 수 있는 맞춤 운동 루틴을 생성합니다.

---

# 2. 사용 데이터

### 📌 데이터 출처  
📍 문화빅데이터 플랫폼 — *체력측정 및 운동처방 데이터*   
https://www.bigdata-culture.kr/

### 포함 항목  
- 연령, 성별, 키, 체중, BMI  
- 악력(좌/우), 윗몸말아올리기, VO₂max 등 주요 체력 지표  
- 운동처방 텍스트(MVM_PRSCRPTN_CN)

총 **약 50만 건 이상**의 방대한 데이터셋을 사용.

---

# 3. 데이터 전처리 & 라벨링

### 9개 운동 카테고리로 단순화
텍스트 기반 라벨링 함수(map_label)를 통해 다음 9개 카테고리로 변환:

- 스트레칭  
- 유산소-걷기  
- 유산소-달리기  
- 유산소-자전거  
- 전신유산소  
- 하체근력  
- 상체근력  
- 코어  
- 요가필라테스  

### Feature 구성 (11개)
- 연령 / 연령대 그룹  
- 성별(F/M)  
- 키 / 체중 / BMI  
- 악력(좌/우)  
- 윗몸말아올리기  
- VO₂max  

---

# 4. 데이터 불균형 문제 해결

## 📌 문제점
원본 데이터는 **극단적인 불균형**을 가짐:

| 라벨 | 개수 |
|------|------|
| 스트레칭 | 17,000+ |
| 요가필라테스 | 20 |
| 하체근력 | 10 |
| 상체근력 | 100~300 |
| 전신유산소 | 400~800 |

→ 학습 시 모델이 무조건 **다수 클래스(스트레칭)** 만 예측하는 문제 발생.

---

## ✔ 최종 해결 전략
### 🔹 RandomOversampler (Oversampling)
- **Train 데이터에만 적용**
- 소수 클래스 데이터를 단순 복제하여 **클래스별 개수를 동일하게 맞춤**
- SMOTE는 사용하지 않음  
  → 이유: 체력측정 데이터는 실제 신체검사 기반 수치라  
    인공 데이터 생성(SMOTE)이 오히려 성능을 악화시켰기 때문

➡️ 이 방식이 가장 **안정적인 Accuracy & F1 조합**을 달성함

---

# 5. 모델 구조

### 🔸 알고리즘: RandomForestClassifier
- n_estimators=350  
- max_depth=22  
- min_samples_split=10  
- n_jobs=-1  

### 🔸 Label Encoding
scikit-learn LabelEncoder 사용

---

# 6. 최종 성능

| 지표 | 값 |
|------|------|
| **Accuracy** | **0.734** |
| Macro F1 | 0.205 |
| Micro F1 | 0.734 |

➡ Accuracy 73%는 **9개 다중 분류 + 극단적 불균형 데이터** 상황에서 매우 준수한 성능  
➡ Macro F1이 낮은 이유는 실제 데이터 분포 자체가 불균형하기 때문  

---

# 7. FastAPI 연동 구조

1. Spring Boot → `/exercise/recommend/{userId}` 호출  
2. FastAPI 서버에서 USERS DB 조회  
3. Feature Vector 자동 생성  
4. RandomForest 모델 예측  
5. 운동 루틴 템플릿 매칭(준비/본/정리)  
6. 유튜브 영상 URL + 썸네일 매핑  
7. JSON으로 반환  

---

# 8. API 예시 응답

```json
{
  "user_id": 17,
  "predicted_category": "코어",
  "routine": {
    "준비운동": {"name": "브릿지", "url": "...", "thumb": "..."},
    "본운동": {"name": "플랭크", "url": "...", "thumb": "..."},
    "정리운동": {"name": "명상", "url": "...", "thumb": "..."}
  }
}
