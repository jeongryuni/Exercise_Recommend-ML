import pandas as pd
import re
import chardet

SRC = "서울올림픽기념국민체육진흥공단_국민체력100 운동처방 동영상주소 정보_20210727.csv"

# 인코딩 자동 감지
with open(SRC, 'rb') as f:
    raw = f.read()
    encoding = chardet.detect(raw)['encoding']
df = pd.read_csv(SRC, encoding=encoding)

print("원본 로우:", df.shape)

# 운동 > 카테고리 매핑
CATEGORY_MAP = {
    "스쿼트": "근력",
    "런지": "근력",
    "힙 브릿지": "근력",
    "푸시업": "근력",
    "밴드 로우": "근력",

    "플랭크": "코어",
    "사이드 플랭크": "코어",
    "크런치": "코어",

    "목 스트레칭": "스트레칭",  # ROUTINE에 맞게 추가
    "팔 스트레칭": "스트레칭",  # ROUTINE에 맞게 추가
    "어깨 스트레칭": "스트레칭",  # ROUTINE에 맞게 추가
    "전신 스트레칭": "스트레칭",
    "상체 스트레칭": "스트레칭",
    "하체 스트레칭": "스트레칭",
    "고관절 스트레칭": "스트레칭",
    "이완 호흡": "스트레칭",
    "명상": "스트레칭",

    "요가 루틴": "요가",
    "비둘기 자세": "요가",
    "전굴": "요가",
    "고양이-소": "요가",
    "브릿지": "요가",

    "빠르게 걷기": "유산소",
    "조깅": "유산소",
    "인터벌 러닝": "유산소",
    "버피": "유산소",
    "점핑잭": "유산소",
    "실내 자전거": "유산소",
    "에르고 미터": "유산소",
    "무릎 돌리기": "유산소",
}

ROUTINE_EXERCISES = list(CATEGORY_MAP.keys())

video_list = []


DEFAULT_THUMB_PATH = "/images/icon/no_video.png"


def extract_youtube_id(url):
    if not url:
        return None

    # 'youtu.be/' 또는 'watch?v=' 뒤의 ID 추출
    match = re.search(r'(?:youtu\.be\/|v=)([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)
    return None


for ex in ROUTINE_EXERCISES:
    category = CATEGORY_MAP[ex]

    # '소분류' 또는 '제목'에 정확한 운동명으로 검색
    matched = df[
        df["소분류"].str.contains(ex, na=False, case=False) |
        df["제목"].str.contains(ex, na=False, case=False)
        ]

    url = None
    thumb = None

    if len(matched) > 0:
        # 가장 관련성 높은 첫 번째 영상 선택
        video_row = matched.iloc[0]
        url = video_row["동영상주소"]

        # 1. 원본 CSV에 썸네일 정보가 있는지 확인
        if "썸네일" in video_row.index and pd.notna(video_row["썸네일"]):
            thumb = video_row["썸네일"]

        if not thumb and url:
            video_id = extract_youtube_id(url)
            if video_id:
                # 고화질(hqdefault) 썸네일 URL 생성
                thumb = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

    video_list.append({
        "운동명": ex,
        "카테고리": category,
        "영상URL": url,
        "썸네일": thumb  # 썸네일 컬럼 추가
    })

video_df = pd.DataFrame(video_list)

video_df["썸네일"] = video_df["썸네일"].fillna(DEFAULT_THUMB_PATH)

# 영상 URL이 없는 경우는 빈 문자열로 유지
video_df["영상URL"] = video_df["영상URL"].fillna("")

video_df.to_csv("exercise_video_mapping.csv", index=False, encoding="utf-8-sig")

print("[완료] exercise_video_mapping.csv 생성됨")
print(video_df)