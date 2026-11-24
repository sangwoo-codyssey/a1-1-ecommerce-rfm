"""A1-1 쇼핑몰 단골 찾기 — 분석 파이프라인.

H&M Personalized Fashion Recommendations(Kaggle) 데이터를 읽어
전처리·피처 엔지니어링·RFM 계산까지 하는 `DataAnalyzer` 클래스.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# articles.csv 에서 가져올 열 — 같은 뜻의 숫자 코드(…_no·…_code·…_id)는 빼고 이름 열만 남긴다.
ARTICLE_COLS = [
    "article_id", "prod_name", "product_type_name", "product_group_name",
    "graphical_appearance_name", "colour_group_name",
    "perceived_colour_value_name", "perceived_colour_master_name",
    "department_name", "index_name", "index_group_name",
    "section_name", "garment_group_name", "detail_desc",
]

# customers.csv 에서 가져올 열 — postal_code(64자 암호화 문자열)는 뺀다.
CUSTOMER_COLS = [
    "customer_id", "FN", "Active", "club_member_status",
    "fashion_news_frequency", "age",
]


def iqr_bounds(series: pd.Series, threshold: float = 1.5) -> tuple[float, float]:
    """IQR(Interquartile Range, 사분위 범위) 방식의 이상치 경계 (lower, upper) 를 돌려준다.

    threshold 는 IQR 에 곱하는 배수다(관례 1.5, 클수록 이상치를 덜 잡는다).
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - threshold * iqr), float(q3 + threshold * iqr)


class DataAnalyzer:
    """거래 줄 단위 표 하나(`self.df`)를 들고 분석 단계를 차례로 적용한다."""

    def __init__(self) -> None:
        self.df: pd.DataFrame | None = None
        self.load_summary: dict = {}
        self.missing_report: pd.DataFrame | None = None
        self.outlier_bounds: dict[str, tuple[float, float]] = {}

    def load_data(
        self,
        transactions_path: str | Path = RAW_DIR / "transactions_train.csv",
        articles_path: str | Path = RAW_DIR / "articles.csv",
        customers_path: str | Path = RAW_DIR / "customers.csv",
        start_date: str = "2019-09-23",
        end_date: str = "2020-09-22",
        n_customers: int | None = 10_000,
        random_state: int = 42,
        chunksize: int = 2_000_000,
    ) -> pd.DataFrame:
        """원본 CSV 3개를 읽어 거래 줄 단위 표를 만든다.

        1. 거래를 덩어리로 읽으며 [start_date, end_date] 밖의 줄을 버린다.
        2. 기간 안에 한 번이라도 산 고객 중 n_customers 명을 무작위로 뽑는다(None 이면 전원).
        3. 뽑힌 고객의 기간 내 거래 줄은 **전부** 남긴다 — 줄을 뽑으면 고객마다 F·M 이 줄어든다.
        4. 거래 줄마다 상품·고객 정보를 왼쪽 결합한다.

        중복 줄은 지우지 않는다. 이 데이터에는 수량 열이 없고 같은 상품을 여러 개 사면
        같은 줄이 반복된다(Kaggle 데이터 설명).
        """
        started = time.perf_counter()

        # 1) 거래 — t_dat 이 'YYYY-MM-DD' 문자열이라 날짜 변환 전에 문자열 비교로 거른다.
        reader = pd.read_csv(
            transactions_path,
            dtype={
                "t_dat": "str",
                "customer_id": "str",
                "article_id": "int64",
                "price": "float64",
                "sales_channel_id": "int8",
            },
            chunksize=chunksize,
        )
        rows_read = 0
        parts = []
        for chunk in reader:
            rows_read += len(chunk)
            parts.append(chunk[chunk["t_dat"].between(start_date, end_date)])
        tx = pd.concat(parts, ignore_index=True)
        window_rows = len(tx)

        # 2) 고객 표본 — 정렬해 두고 뽑아야 파일 줄 순서와 무관하게 같은 사람이 뽑힌다.
        customers_in_window = pd.Series(tx["customer_id"].unique()).sort_values(ignore_index=True)
        if n_customers is not None:
            if n_customers > len(customers_in_window):
                raise ValueError(
                    f"n_customers={n_customers} 가 기간 내 고객 수 {len(customers_in_window)} 보다 많다"
                )
            chosen = customers_in_window.sample(n=n_customers, random_state=random_state)
            tx = tx[tx["customer_id"].isin(chosen)].reset_index(drop=True)

        # 3) 남은 줄에만 날짜 변환
        tx["t_dat"] = pd.to_datetime(tx["t_dat"], format="%Y-%m-%d")

        # 4) 상품·고객 정보 결합 (거래 줄 수가 바뀌면 안 되므로 many_to_one 검증)
        articles = pd.read_csv(articles_path, usecols=ARTICLE_COLS, dtype={"article_id": "int64"})
        customers = pd.read_csv(customers_path, usecols=CUSTOMER_COLS, dtype={"customer_id": "str"})
        customers = customers[customers["customer_id"].isin(tx["customer_id"].unique())]

        df = tx.merge(articles, on="article_id", how="left", validate="many_to_one")
        df = df.merge(customers, on="customer_id", how="left", validate="many_to_one")

        self.df = df
        self.load_summary = {
            "start_date": start_date,
            "end_date": end_date,
            "random_state": random_state,
            "rows_read": rows_read,
            "window_rows": window_rows,
            "window_customers": len(customers_in_window),
            "sampled_customers": df["customer_id"].nunique(),
            "rows": len(df),
            "articles_unmatched": int((~df["article_id"].isin(articles["article_id"])).sum()),
            "customers_unmatched": int((~df["customer_id"].isin(customers["customer_id"])).sum()),
            "seconds": round(time.perf_counter() - started, 1),
        }
        return self.df

    def _require_df(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("load_data() 를 먼저 실행해야 한다")
        return self.df

    def add_text_features(self, col: str = "prod_name") -> pd.DataFrame:
        """상품명 텍스트에서 단어 수·글자 수 열을 만든다.

        `.str` 접근자는 열 전체에 한 번에 적용되므로 줄마다 도는 반복문이 없다.
        단어는 공백 기준으로 나눈다("Emery(1)" 은 1단어).
        """
        df = self._require_df()
        df[f"{col}_word_count"] = df[col].str.split().str.len()
        df[f"{col}_length"] = df[col].str.len()
        return df

    def handle_missing_values(
        self,
        strategy: str = "group_median",
        target_col: str = "age",
        group_cols: list[str] | None = None,
        unit_col: str | None = "customer_id",
        flag_cols: tuple[str, ...] = ("FN", "Active"),
        category_cols: tuple[str, ...] = ("club_member_status", "fashion_news_frequency", "detail_desc"),
        category_fill: str = "UNKNOWN",
    ) -> pd.DataFrame:
        """결측을 세 가지 규칙으로 처리하고 처리 전후 결측 수를 `self.missing_report` 에 남긴다.

        1. flag_cols — 값이 1 하나뿐인 표시 열. 빈칸 = '아니오' 로 보고 0 으로 채운다(인코딩 복원).
        2. category_cols — 범주·텍스트 열. category_fill 로 채운다. 그룹 기준 열의 빈칸도
           여기서 먼저 채워야 빈칸 고객도 한 그룹으로 묶여 3번 대치를 받는다.
        3. target_col — strategy 로 대치한다.
           - "group_median"/"group_mean": group_cols 그룹의 중앙값/평균 (그룹 값이 없으면 전체 값)
           - "median"/"mean": 전체 중앙값/평균
           unit_col 을 주면 통계를 거래 줄이 아니라 unit_col 1개당 1표로 계산한다
           (자주 사는 고객이 통계를 끌고 가지 않게).
        """
        df = self._require_df()
        if group_cols is None:
            group_cols = ["club_member_status", "fashion_news_frequency"]
        func = {"group_median": "median", "median": "median",
                "group_mean": "mean", "mean": "mean"}.get(strategy)
        if func is None:
            raise ValueError(f"알 수 없는 strategy: {strategy}")

        touched = [*flag_cols, *category_cols, target_col]
        before = df[touched].isna().sum()

        df[list(flag_cols)] = df[list(flag_cols)].fillna(0)
        df[list(category_cols)] = df[list(category_cols)].fillna(category_fill)

        units = df.drop_duplicates(unit_col) if unit_col else df
        overall = units[target_col].agg(func)
        if strategy.startswith("group_"):
            group_stat = units.groupby(group_cols)[target_col].agg(func).rename("_fill")
            fill = df[group_cols].join(group_stat, on=group_cols)["_fill"].fillna(overall)
        else:
            fill = overall
        df[target_col] = df[target_col].fillna(fill)

        self.missing_report = pd.DataFrame({"before": before, "after": df[touched].isna().sum()})
        if strategy.startswith("group_"):
            self.missing_report.attrs["group_stat"] = group_stat
        return df

    def detect_outliers(self, column: str = "price", threshold: float = 1.5) -> pd.DataFrame:
        """IQR 경계 밖의 줄을 돌려준다. 경계는 `self.outlier_bounds[column]` 에 남긴다."""
        df = self._require_df()
        lower, upper = iqr_bounds(df[column], threshold)
        self.outlier_bounds[column] = (lower, upper)
        return df[(df[column] < lower) | (df[column] > upper)]

    def clip_outliers(
        self, column: str = "price", threshold: float = 1.5, new_col: str | None = None
    ) -> pd.DataFrame:
        """IQR 경계 밖 값을 경계값으로 자른 새 열을 만든다. 원본 열은 그대로 둔다.

        비싼 상품은 입력 오류가 아니라 실제 구매라서 지우지 않는다. RFM 의 M 은 원본 열로 계산한다.
        """
        df = self._require_df()
        lower, upper = iqr_bounds(df[column], threshold)
        self.outlier_bounds[column] = (lower, upper)
        df[new_col or f"{column}_clipped"] = df[column].clip(lower, upper)
        return df
