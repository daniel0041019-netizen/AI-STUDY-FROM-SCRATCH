"""명시적 인덱스, 멀티인덱스 및 .loc / .iloc 슬라이싱 비교 스니펫."""

import numpy as np
import pandas as pd


def main() -> None:
    df = pd.DataFrame([
        {"name": "Bella", "breed": "Labrador", "color": "Black", "height_cm": 56.0, "weight_kg": 28.5},
        {"name": "Max", "breed": "Poodle", "color": "Brown", "height_cm": 42.0, "weight_kg": 11.2},
        {"name": "Charlie", "breed": "Labrador", "color": "Brown", "height_cm": 60.0, "weight_kg": 31.0},
        {"name": "Lucy", "breed": "Beagle", "color": "White", "height_cm": 38.0, "weight_kg": np.nan},
    ])

    unique_dogs = df.drop_duplicates(subset=["name", "breed"])
    indexed_df = unique_dogs.set_index(["breed", "color"]).sort_index()

    print("=== 멀티인덱스 .loc 슬라이싱 (끝 경계 포함) ===")
    print(indexed_df.loc[("Beagle", "White"):("Labrador", "Brown"), ["name", "height_cm"]])

    print("\n=== 위치 기반 .iloc 슬라이싱 (끝 경계 미포함) ===")
    print(df.iloc[0:2, 0:4])

    _reset_df = indexed_df.reset_index()


if __name__ == "__main__":
    main()
