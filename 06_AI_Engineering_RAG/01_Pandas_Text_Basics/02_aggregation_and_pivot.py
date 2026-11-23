"""맞춤형 집계(.agg), 누적합(.cumsum), 중복 제거, 그룹화 및 피벗 테이블 스니펫."""

import numpy as np
import pandas as pd


def iqr(series: pd.Series) -> float:
    return float(series.quantile(0.75) - series.quantile(0.25))


def main() -> None:
    df = pd.DataFrame([
        {"name": "Bella", "breed": "Labrador", "color": "Black", "height_cm": 56.0, "weight_kg": 28.5},
        {"name": "Max", "breed": "Poodle", "color": "Brown", "height_cm": 42.0, "weight_kg": 11.2},
        {"name": "Charlie", "breed": "Labrador", "color": "Brown", "height_cm": 60.0, "weight_kg": 31.0},
        {"name": "Lucy", "breed": "Beagle", "color": "White", "height_cm": 38.0, "weight_kg": np.nan},
        {"name": "Bella", "breed": "Labrador", "color": "Black", "height_cm": 56.0, "weight_kg": 28.5},
    ])

    print("=== 맞춤 집계 (.agg) & 누적합 (.cumsum) ===")
    print(df["weight_kg"].agg([iqr, "median", "mean"]))
    df["cum_height_cm"] = df["height_cm"].cumsum()
    print(df[["name", "height_cm", "cum_height_cm"]])

    # 중복 제거 후 그룹별 집계 및 빈도 확인
    unique_dogs = df.drop_duplicates(subset=["name", "breed"])
    print("\n=== 견종 빈도 (.value_counts) ===")
    print(unique_dogs["breed"].value_counts(sort=True))

    grouped_stats = unique_dogs.groupby(["breed", "color"])["weight_kg"].agg(["min", "max", "mean"])
    print("\n=== 그룹별 요약 통계 ===")
    print(grouped_stats)

    pivot = unique_dogs.pivot_table(
        values="weight_kg",
        index="breed",
        columns="color",
        aggfunc="mean",
        fill_value=0.0,
        margins=True,
    )
    print("\n=== 피벗 테이블 ===")
    print(pivot)


if __name__ == "__main__":
    main()
