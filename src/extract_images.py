"""H&M 전체 zip 에서 표본 고객이 산 상품의 사진만 골라 data/images/ 에 푼다.

    python -m src.extract_images                 # 기본: 1,000장, random_state=42

표본은 DataAnalyzer.load_data() 기본값과 같은 기준으로 만든다. 사진이 없는 상품은
후보에서 빼고 뽑으므로 정확히 n_images 장이 나온다. random_state 가 같으면 같은 사진이 뽑힌다.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from src.pipeline import IMAGE_DIR, RAW_DIR, DataAnalyzer, image_relpath

ZIP_PATH = RAW_DIR / "h-and-m-personalized-fashion-recommendations.zip"


def extract_sample_images(
    zip_path: str | Path = ZIP_PATH,
    out_dir: str | Path = IMAGE_DIR,
    n_images: int = 1_000,
    random_state: int = 42,
) -> pd.Series:
    """뽑은 article_id 를 돌려준다."""
    df = DataAnalyzer().load_data()
    articles = pd.Series(sorted(df["article_id"].unique()))

    with zipfile.ZipFile(zip_path) as zf:
        members = set(zf.namelist())
        member_of = articles.map(lambda a: f"images/{image_relpath(a)}")
        with_image = articles[member_of.isin(members)]
        chosen = with_image.sample(n=n_images, random_state=random_state).sort_values()

        out_dir = Path(out_dir)
        for article_id in chosen:
            target = out_dir / image_relpath(article_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(f"images/{image_relpath(article_id)}"))

    print(f"표본 상품 {len(articles):,}개 중 사진 있음 {len(with_image):,}개 → {len(chosen):,}장 추출 → {out_dir}")
    return chosen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-images", type=int, default=1_000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    extract_sample_images(n_images=args.n_images, random_state=args.random_state)
