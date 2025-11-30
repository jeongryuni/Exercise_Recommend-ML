# ============================================================
# Accuracy: 0.7344827586206897
# Macro F1: 0.20553577107650758
# Micro F1: 0.7344827586206897
# ============================================================

import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

print(" 모델 학습 시작...")

# 1. CSV 로드

files = sorted(glob.glob("data/KS_NFA_FTNESS_MESURE_MVN_PRSCRPTN_GNRLZ_INFO_*.csv"))
print("로드 파일:", files)

df_list = []
for file in files:
    print("로드 중:", file)
    temp = pd.read_csv(file, encoding='utf-8-sig')
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)
print("원본 shape:", df.shape)

# 2. 수치형 컬럼 변환

numeric_cols = [
    "MESURE_AGE_CO",
    "MESURE_IEM_001_VALUE", "MESURE_IEM_002_VALUE",
    "MESURE_IEM_007_VALUE", "MESURE_IEM_008_VALUE",
    "MESURE_IEM_009_VALUE", "MESURE_IEM_018_VALUE",
    "MESURE_IEM_030_VALUE"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df["MESURE_AGE_CO"].between(10, 100)]
df["group"] = (df["MESURE_AGE_CO"] // 10) * 10

# 3. 성별 인코딩

df["F"] = (df["SEXDSTN_FLAG_CD"] == "F").astype(int)
df["M"] = (df["SEXDSTN_FLAG_CD"] == "M").astype(int)

# 4. 운동 라벨링 함수 (9개 카테고리)

def map_label(text):
    if pd.isna(text):
        return None

    t = text.lower()

    if "스트레칭" in t:
        return "스트레칭"
    if "걷기" in t or "워킹" in t:
        return "유산소-걷기"
    if "달리기" in t or "조깅" in t or "러닝" in t:
        return "유산소-달리기"
    if "자전거" in t or "사이클" in t:
        return "유산소-자전거"
    if "버피" in t or "점프" in t or "전신" in t:
        return "전신유산소"
    if "스쿼트" in t or "런지" in t or "하체" in t:
        return "하체근력"
    if "푸시업" in t or "팔굽" in t or "가슴" in t or "어깨" in t:
        return "상체근력"
    if "플랭크" in t or "복근" in t or "코어" in t:
        return "코어"
    if "요가" in t or "필라테스" in t:
        return "요가필라테스"

    return None

df["label"] = df["MVM_PRSCRPTN_CN"].apply(map_label)
df = df.dropna(subset=["label"])

print(" 라벨 분포:")
print(df["label"].value_counts())

# 5. Feature 구성

feature_cols = [
    "group",
    "MESURE_AGE_CO",
    "F", "M",
    "MESURE_IEM_001_VALUE",
    "MESURE_IEM_002_VALUE",
    "MESURE_IEM_018_VALUE",
    "MESURE_IEM_007_VALUE",
    "MESURE_IEM_008_VALUE",
    "MESURE_IEM_009_VALUE",
    "MESURE_IEM_030_VALUE",
]

df = df.dropna(subset=feature_cols)

X = df[feature_cols]
y = df["label"]

# 6. Label Encoding

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 7. Train/Test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 8. Oversampling(Train 데이터 ONLY)

ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

print("\n Oversampling 후 분포:")
print(pd.Series(y_train_res).value_counts())

# 9. 모델 학습

model = RandomForestClassifier(
    n_estimators=350,
    max_depth=22,
    min_samples_split=10,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_res, y_train_res)
pred = model.predict(X_test)

# 10. 평가

print("\n==============================")
print(" 최종 모델 성능")
print("==============================")
print("Accuracy:", accuracy_score(y_test, pred))
print("Macro F1:", f1_score(y_test, pred, average="macro"))
print("Micro F1:", f1_score(y_test, pred, average="micro"))
print("\nClassification Report:\n")
print(classification_report(y_test, pred, target_names=encoder.classes_))

# ============================================================
# 11. 모델 저장
# ============================================================

joblib.dump(model, "exercise_recommender.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("\n 모델 저장 완료!")

