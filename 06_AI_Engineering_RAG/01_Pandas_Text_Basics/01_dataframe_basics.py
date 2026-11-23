"""DataFrame 생성, 기본 탐색, 정렬, 조건 필터링 및 체이닝 스니펫."""

import numpy as np
import pandas as pd


def main() -> None:
    # 행 단위 생성 (딕셔너리의 리스트)
    rows_data = [
        {"name": "Bella", "breed": "Labrador", "color": "Black", "height_cm": 56.0, "weight_kg": 28.5, "date_of_birth": "2018-04-12"},
        {"name": "Max", "breed": "Poodle", "color": "Brown", "height_cm": 42.0, "weight_kg": 11.2, "date_of_birth": "2020-09-03"},
        {"name": "Charlie", "breed": "Labrador", "color": "Brown", "height_cm": 60.0, "weight_kg": 31.0, "date_of_birth": "2015-01-20"},
        {"name": "Lucy", "breed": "Beagle", "color": "White", "height_cm": 38.0, "weight_kg": np.nan, "date_of_birth": "2014-07-19"},
        {"name": "Bella", "breed": "Labrador", "color": "Black", "height_cm": 56.0, "weight_kg": 28.5, "date_of_birth": "2018-04-12"},
    ]
    df = pd.DataFrame(rows_data)

    # 열 단위 생성 (리스트의 딕셔너리)
    df_cols = pd.DataFrame({
        "product": ["A", "B", "C"],
        "price": [1000, 2000, 3000],
    })

    print("=== 탐색 및 핵심 구성 요소 ===")
    print(df.head(2))
    print("Shape:", df.shape)
    print("Columns:", df.columns)
    print("Index:", df.index)
    print("2D NumPy 배열 형태:\n", df[["height_cm", "weight_kg"]].to_numpy())
    print(df.describe())
    print("\n열 단위 생성 DataFrame:\n", df_cols)

    # 파생 변수 추가 및 다중 기준 정렬
    df["height_m"] = df["height_cm"] / 100
    df["weight_per_m"] = df["weight_kg"] / df["height_m"]
    _sorted_df = df.sort_values(["weight_kg", "height_cm"], ascending=[False, True])

    # 조건 필터링 + .isin() + 정렬 + 열 선택 체이닝
    filtered_df = (
        df[
            (df["height_cm"] >= 40)
            & (df["date_of_birth"] >= "2015-01-01")
            & (df["color"].isin(["Black", "Brown"]))
        ]
        .sort_values("height_cm", ascending=False)[["name", "breed", "color", "height_cm"]]
    )
    print("\n=== 필터링 + 정렬 + 열 선택 결과 ===")
    print(filtered_df)


if __name__ == "__main__":
    main()
