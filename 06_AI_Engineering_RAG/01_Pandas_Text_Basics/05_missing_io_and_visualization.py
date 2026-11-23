"""결측치 탐지/처리, CSV 입출력 및 Matplotlib 시각화 스니펫."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    df = pd.DataFrame([
        {"name": "Bella", "breed": "Labrador", "color": "Black", "height_cm": 56.0, "weight_kg": 28.5, "review": "활발함"},
        {"name": "Max", "breed": "Poodle", "color": "Brown", "height_cm": 42.0, "weight_kg": 11.2, "review": "똑똑함"},
        {"name": "Charlie", "breed": "Labrador", "color": "Brown", "height_cm": 60.0, "weight_kg": 31.0, "review": "산책 최고"},
        {"name": "Lucy", "breed": "Beagle", "color": "White", "height_cm": 38.0, "weight_kg": np.nan, "review": None},
    ])

    print("=== 결측치 확인 ===")
    print(df.isna().any())
    print(df.isna().sum())

    # 결측치 대체(.fillna) 및 특정 열 기준 행 제거(.dropna)
    df_filled = df.copy()
    df_filled["weight_kg"] = df_filled["weight_kg"].fillna(df_filled["weight_kg"].mean())
    _df_dropped = df.dropna(subset=["review"])

    # CSV 저장 및 불러오기
    csv_path = Path("dogs_summary.csv")
    df_filled.to_csv(csv_path, index=False)
    loaded_df = pd.read_csv(csv_path)
    print("\n=== CSV 로드 확인 ===")
    print(loaded_df.head(2))

    # Matplotlib 시각화
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    df_filled["height_cm"].hist(bins=8, alpha=0.7, ax=axes[0])
    axes[0].set_title("Height Distribution (cm)")
    axes[0].set_xlabel("height_cm")

    df_filled.plot(x="height_cm", y="weight_kg", kind="scatter", ax=axes[1], title="Height vs Weight")

    plt.tight_layout()
    # plt.show()  # 로컬/노트북 환경에서 창 출력 시 주석 해제


if __name__ == "__main__":
    main()
