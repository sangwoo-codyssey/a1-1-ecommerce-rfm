# A1-1 - 쇼핑몰 단골 찾기 (ecommerce-rfm)

> 🚧 **미착수** — 디렉터리·레포 준비만 완료

이커머스 데이터(수치·텍스트·범주·날짜·이미지 배열)를 NumPy 벡터 연산으로 처리하는
`DataAnalyzer` 파이프라인을 만들고, EDA·IQR 이상치 탐지·그룹별 결측 대치·6종 시각화를 거쳐
RFM 으로 고객을 4개 이상 세그먼트로 나눠 데이터 근거가 있는 전략을 제안한다.

- 과제 원문: 로컬 `../a1-1-ecommerce-rfm.md`
- AI 심화 학습 · 데이터 분석 · 60시간 · 필수 · 개인
- Python 3.12 이상 (과제 조건은 3.8 이상 — 고정한 `numpy==2.5.4` 가 3.12 를 요구한다)

## 설치 및 실행

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt   # uv 가 없으면 python -m venv .venv 후 pip install -r requirements.txt
```

`requirements.txt` 는 버전을 `==` 로 고정했다. `jupyterlab`·`ipykernel` 은 노트북 실행용이고 분석 코드에서는 쓰지 않는다.

진입점(노트북·스크립트)은 착수 때 정한다.

## 데이터 (과제 §4-1·§7)

| 조건 | 기준 |
|---|---|
| 규모 | 1,000건 이상 · 8개 이상 컬럼 |
| 유형 | 수치형·범주형·날짜형·이미지(배열) 중 3가지 이상 |
| RFM 거래 데이터 | `customer_id`, `order_date`, `amount`(또는 price*qty) |
| 출처 | Kaggle·AI Hub 등 공개 라이선스 우선, **출처 명시 필수** |

### 사용 데이터셋

**H&M Personalized Fashion Recommendations** (Kaggle 대회 데이터)
— https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data

| 파일 | 크기 | 행 수 | 열 수 | 주요 열 |
|---|---|---|---|---|
| `transactions_train.csv` | 3.49GB | 31,788,324 | 5 | `t_dat`, `customer_id`, `article_id`, `price`, `sales_channel_id` |
| `articles.csv` | 36.1MB | 105,542 | 25 | `article_id`, `prod_name`, `product_type_name`, `colour_group_name`, `detail_desc` 등 |
| `customers.csv` | 207MB | 1,371,980 | 7 | `customer_id`, `age`, `club_member_status`, `fashion_news_frequency` 등 |
| `images/` | — | — | — | 상품 사진(`article_id` 별 JPG) — 표본 기준을 정한 뒤 일부만 받는다 |

- **라이선스**: Kaggle 대회 규칙(Competition Rules)을 따른다. 재배포가 금지돼 있어 원본·가공본·사진 모두
  레포에 올리지 않는다(`.gitignore` 의 `data/`).
- **받는 방법**: Kaggle 에 로그인해 대회 규칙에 동의한 뒤, Data 탭에서 위 CSV 3개를 파일별로 내려받아
  `data/raw/` 에 압축을 푼다.

## 라이브러리 제약 (과제 §6·§7)

| 구분 | 라이브러리 |
|---|---|
| 허용 | NumPy, Pandas, Matplotlib, Seaborn |
| 예외 | Plotly — 보너스 3 한정 |
| 금지 | OpenCV·Pillow 등 이미지 처리 전용, Scikit-learn·NLTK, 자동 EDA(pandas-profiling·sweetviz) |

이미지는 `imread()` 로 읽은 NumPy 배열을 직접 조작하고, 리사이즈는 슬라이싱(`arr[::n, ::n]`)으로 한다.

## 제출물 체크리스트

- [ ] `src/pipeline.py` — `DataAnalyzer` 클래스 (`load_data`·`handle_missing_values`·`detect_outliers`·`calculate_rfm`, 주요 파라미터는 인자로)
- [ ] 이미지 평균·표준편차 컬럼(반복문 없이) + 상품명 단어 수/길이 파생 변수
- [ ] 그룹별(GroupBy) 결측 대치 1가지 이상 + IQR 이상치 함수 직접 구현, 처리 전후 분포 비교
- [ ] 통계(평균·중앙값·표준편차·사분위수) + 상관계수 2쌍 이상 — 해석마다 수치 근거 1개 + 시사점 1문장
- [ ] 시각화 6종: 히스토그램·박스플롯(전후)·막대·히트맵·산점도·라인차트 — 제목·축 레이블 포함
- [ ] `notebooks/analysis_report.ipynb` — 단계마다 마크다운 설명
- [ ] README — 개요·데이터 설명·실행 방법·비즈니스 인사이트 3개 이상 (각각 근거/실행/검증)

## 보너스

- [ ] 이미지 고급 피처 (히스토그램 분포·엣지 특성, NumPy)
- [ ] 코호트 분석 (가입 월별 재구매율 히트맵)
- [ ] Plotly 인터랙티브 RFM 시각화
