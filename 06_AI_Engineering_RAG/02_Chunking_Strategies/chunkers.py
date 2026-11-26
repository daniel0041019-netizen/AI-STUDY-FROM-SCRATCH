from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import pandas as pd


@dataclass
class ChunkConfig:
    """청킹 하이퍼파라미터 설정"""

    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: Sequence[str] = ("\n\n", "\n", ". ", " ", "")

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size는 0보다 커야 합니다.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap은 0 이상이어야 합니다.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap은 chunk_size보다 작아야 합니다.")


class TextChunker:
    """
    Fixed-size 및 Recursive Character Chunking을 지원하고
    Pandas DataFrame을 청크 단위 DataFrame으로 변환하는 클래스
    """

    def __init__(self, config: Optional[ChunkConfig] = None) -> None:
        self.config = config or ChunkConfig()

    @property
    def chunk_size(self) -> int:
        return self.config.chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self.config.chunk_overlap

    @property
    def separators(self) -> List[str]:
        return list(self.config.separators)

    def fixed_chunk(self, text: str) -> List[str]:
        """고정 길이(chunk_size)와 중첩(chunk_overlap)을 적용한 분할"""
        if not isinstance(text, str):
            return []
        cleaned = text.strip()
        if not cleaned:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: List[str] = []

        for start in range(0, len(cleaned), step):
            end = start + self.chunk_size
            piece = cleaned[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(cleaned):
                break

        return chunks

    def recursive_chunk(self, text: str) -> List[str]:
        """구분자 우선순위에 따른 Recursive Character Chunking"""
        if not isinstance(text, str):
            return []
        cleaned = text.strip()
        if not cleaned:
            return []

        return self._split_recursive(cleaned, self.separators)

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """현재 텍스트에 적합한 구분자를 찾아 분할하고, 초과 조각은 재귀 처리"""
        chosen_sep = separators[-1]
        remaining_seps: List[str] = []

        for idx, sep in enumerate(separators):
            if sep == "":
                chosen_sep = ""
                remaining_seps = []
                break
            if sep in text:
                chosen_sep = sep
                remaining_seps = separators[idx + 1 :]
                break

        raw_splits = text.split(chosen_sep) if chosen_sep else list(text)

        good_splits: List[str] = []
        final_chunks: List[str] = []

        for part in raw_splits:
            if not part:
                continue
            if len(part) <= self.chunk_size:
                good_splits.append(part)
            else:
                # 먼저 지금까지 모인 짧은 조각들을 병합하여 플러시
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, chosen_sep))
                    good_splits = []

                # 더 하위 구분자가 있으면 재귀 분할, 없으면 그대로 추가
                if remaining_seps:
                    final_chunks.extend(self._split_recursive(part, remaining_seps))
                else:
                    final_chunks.append(part)

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, chosen_sep))

        # 혹시 chunk_size를 초과하는 단일 조각이 남았을 경우 안전망으로 글자 단위 분할
        normalized: List[str] = []
        for chunk in final_chunks:
            c = chunk.strip()
            if not c:
                continue
            if len(c) > self.chunk_size:
                normalized.extend(self.fixed_chunk(c))
            else:
                normalized.append(c)

        return normalized

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """분할된 조각들을 chunk_size 한도 내에서 합치고 chunk_overlap만큼 문맥을 남김"""
        merged_docs: List[str] = []
        current_parts: List[str] = []
        current_len = 0
        sep_len = len(separator)

        for part in splits:
            part_len = len(part)
            added_len = part_len + (sep_len if current_parts else 0)

            # 현재 청크에 part를 더했을 때 chunk_size를 초과하면 기존 청크를 확정
            if current_parts and (current_len + added_len > self.chunk_size):
                merged_text = separator.join(current_parts).strip()
                if merged_text:
                    merged_docs.append(merged_text)

                # chunk_overlap 이하가 될 때까지 앞쪽 조각을 제거하여 경계 문맥 보존
                while current_parts and (
                    current_len + added_len > self.chunk_size
                    and current_len > self.chunk_overlap
                ):
                    removed = current_parts.pop(0)
                    current_len -= len(removed) + (sep_len if current_parts else 0)
                    added_len = part_len + (sep_len if current_parts else 0)

            current_parts.append(part)
            current_len += part_len + (sep_len if len(current_parts) > 1 else 0)

        if current_parts:
            merged_text = separator.join(current_parts).strip()
            if merged_text:
                merged_docs.append(merged_text)

        return merged_docs

    def chunk_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str,
        id_col: Optional[str] = None,
        method: str = "recursive",
    ) -> pd.DataFrame:
        """
        Pandas DataFrame의 텍스트 컬럼을 청킹하여
        original_id, chunk_index, chunk_text, char_length 컬럼을 가진 새 DataFrame 반환
        """
        if text_col not in df.columns:
            raise KeyError(f"컬럼 '{text_col}'이 DataFrame에 존재하지 않습니다.")
        if id_col is not None and id_col not in df.columns:
            raise KeyError(f"ID 컬럼 '{id_col}'이 DataFrame에 존재하지 않습니다.")
        if method not in {"recursive", "fixed"}:
            raise ValueError("method는 'recursive' 또는 'fixed'여야 합니다.")

        chunk_func = self.recursive_chunk if method == "recursive" else self.fixed_chunk
        rows = []

        for idx, row in df.iterrows():
            original_id = row[id_col] if id_col is not None else idx
            text_value = row[text_col]
            chunks = chunk_func(text_value)

            for chunk_idx, chunk_text in enumerate(chunks):
                rows.append(
                    {
                        "original_id": original_id,
                        "chunk_index": chunk_idx,
                        "chunk_text": chunk_text,
                        "char_length": len(chunk_text),
                    }
                )

        return pd.DataFrame(
            rows, columns=["original_id", "chunk_index", "chunk_text", "char_length"]
        )
