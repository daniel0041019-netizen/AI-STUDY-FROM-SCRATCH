"""문자열 전용 접근자(.str) 기반 벡터화 전처리 스니펫."""

import pandas as pd


def main() -> None:
    str_df = pd.DataFrame({
        "review": [
            "  너무 좋아요!! 👍  ",
            "PYTHON error 발생했습니다. ",
            None,
            "배송이 빠르고 만족합니다.",
        ]
    })

    str_df["cleaned"] = (
        str_df["review"]
        .str.strip()
        .str.replace("너무", "정말", regex=False)
        .str.replace(r"[^가-힣a-zA-Z0-9 ]", "", regex=True)
        .str.lower()
    )
    str_df["char_len"] = str_df["cleaned"].str.len()
    str_df["tokens"] = str_df["cleaned"].str.split()
    str_df["first_token"] = str_df["tokens"].str[0]

    python_mentions = str_df[str_df["review"].str.contains("python", case=False, na=False)]
    print("=== .str 문자열 전처리 결과 ===")
    print(str_df)
    print("\n'python' 포함 행:")
    print(python_mentions)


if __name__ == "__main__":
    main()
