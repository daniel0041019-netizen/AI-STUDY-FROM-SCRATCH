# Text Chunking Strategies & RAG Pipeline Summary

> **Core Philosophy:** Pandas로 정제된 긴 문서를 임베딩 모델과 LLM이 소화하기 적합한 크기로 분할하는 청킹 핵심 원리부터 Fixed-size 한계, Recursive 계층 분할, Overlap 경계 보존, 그리고 DataFrame 기반 `chunkers.py` 파이프라인 구현 정리

---

## 1. 청킹의 필요성과 Fixed-size Chunking 한계

### RAG 파이프라인에서 청킹이 필요한 이유
- **의미 희석 방지**: 서로 다른 주제가 길게 섞인 원문을 통째로 임베딩하면 세부 질문에 대한 검색 민감도가 떨어짐
- **모델 입력 한계 대응**: 임베딩 모델과 생성 LLM은 처리 가능한 최대 길이가 정해져 있어 긴 문서를 적정 단위로 나누어야 함
- **검색 정밀도와 비용 제어**: 불필요한 문맥까지 한꺼번에 넘기면 노이즈가 늘어나고 토큰·연산 비용이 증가함

### Fixed-size Chunking과 문맥 파괴 (Context Fragmentation)
- **동작 방식**: 문단이나 문장 구조를 고려하지 않고 고정된 길이(`chunk_size`) 기준으로 문자열을 슬라이싱해 분할
- **문맥 파괴 현상**:
  - 단어, 형태소, 주어·서술어 구조가 중간에 잘려 독립적인 의미 단위가 무너짐
  - 지시 대상이나 핵심 조건이 앞뒤 조각으로 갈라져 임베딩 벡터의 표현 품질이 저하됨

---

## 2. 문자 길이(`len()`)와 토큰(Token)의 괴리

### 문자열 길이와 모델 입력 단위의 차이
- **글자 수 vs 토큰 수**: Python `len()`은 문자 개수를 세지만, 모델은 서브워드/토큰 단위로 입력을 처리함
- **언어 및 기호별 밀도 차이**: 한국어 문장이나 특수문자·기호가 섞인 텍스트는 동일한 글자 수라도 영어 대비 토큰 소모량이 달라질 수 있음

### `chunk_size` 설계 기준
- **너무 큰 크기**: 모델 입력 한도 초과(`Truncation`) 위험이 커지고, 한 청크에 여러 소주제가 섞여 검색 정확도가 희석됨
- **너무 작은 크기**: 문장이 완성되지 않아 맥락이 부족해지고, 전체 청크 수가 늘어나 임베딩 호출 및 유사도 계산 비용이 증가함
- **실무 조정 원칙**: 문서 특성에 맞는 초기값(예: `chunk_size=500`)으로 시작해 실제 검색 결과에서 정답 근거가 온전히 포함되는지 검증하며 조절

---

## 3. Recursive Character Chunking & Overlap 설계

### 계층적 구분자 기반 재귀 분할 (Recursive Character Chunking)
- **구분자 우선순위**: 큰 의미 단위에서 작은 의미 단위 순서로 구분자를 적용 (`"\n\n"` → `"\n"` → `". "` → `" "` → `""`)
- **분할과 병합 (`Split & Merge`)**:
  - `chunk_size`를 초과하는 조각은 하위 구분자로 재귀 분할하고, 짧은 조각들은 `chunk_size` 한도 내에서 병합해 의미 단위를 유지
- **RAG 검색에서의 장점**: 문단과 문장 경계를 존중하므로 문맥 완결성이 높아지고, 임베딩 벡터가 고유한 의미를 더 정확하게 반영함

### Overlap 설정과 경계선 문맥 보존
- **목적**: 청크 경계에 걸친 지시어·연결 문맥이 끊기지 않도록 이전 청크 끝부분의 일부(`chunk_overlap`)를 다음 청크 앞부분에 중첩
- **파라미터 트레이드오프**:
  - `chunk_overlap=0`: 저장 및 임베딩 비용은 최소지만 경계 문맥 단절 위험이 큼
  - 과도한 Overlap: 경계 손실은 줄어들지만 중복 청크가 늘어나 임베딩 비용과 검색 중복이 증가함
  - 기본 설정(예: `chunk_size=500`, `chunk_overlap=50`): 문맥 보존과 연산 효율의 균형을 맞추는 출발점

### Fixed-size vs Recursive Chunking 비교

| 비교 항목 | Fixed-size Chunking | Recursive Character Chunking |
| :--- | :--- | :--- |
| **분할 기준** | 고정 길이(`chunk_size`) 슬라이싱 | 계층적 구분자(`"\n\n"`, `"\n"`, `". "`, `" "`, `""`) 우선 적용 |
| **문맥 보존** | 낮음 (문장·단어 중간 절단 발생) | 높음 (문단·문장 경계 및 의미 단위 유지) |
| **구현 복잡도** | 낮음 | 중간 (재귀 분할, 병합, Overlap 제어 필요) |
| **RAG 검색 적합성** | 단편적 조각으로 인해 노이즈 발생 가능 | 의미 완결성이 높아 검색 정밀도 확보에 유리 |

---

## 4. Pandas DataFrame 연동 & `chunkers.py` 파이프라인

### 데이터프레임 변환 구조
- **1:N 정규화**: 원본 문서 1개를 여러 청크 행으로 펼쳐 `original_id`, `chunk_index`, `chunk_text`, `char_length` 컬럼으로 구성
- **메타데이터 추적성**: 검색 단계에서 상위 청크가 선택되었을 때 어떤 원본 문서(`original_id`)의 몇 번째 조각(`chunk_index`)인지 즉시 역추적 가능

### `chunkers.py` 핵심 구성 요소
- **`ChunkConfig`**: `chunk_size`, `chunk_overlap`, `separators`를 보관하고 `chunk_overlap < chunk_size` 등 유효성 조건을 검증하는 데이터 클래스
- **`TextChunker.fixed_chunk()`**: 고정 길이와 `step = chunk_size - chunk_overlap` 보폭으로 문자열을 분할
- **`TextChunker.recursive_chunk()`**: `_split_recursive()`와 `_merge_splits()`를 통해 계층 분할, 병합, Overlap 유지를 수행
- **`TextChunker.chunk_dataframe()`**: 입력 DataFrame의 결측/비문자열 값을 안전하게 처리하고 청크 단위 DataFrame을 반환

### 파이프라인 연계 흐름 (Week 1 → Week 2 → Week 3)
- **입력 (Week 1)**: 정제된 텍스트 컬럼을 포함한 Pandas DataFrame
- **처리 및 출력 (Week 2)**: `chunkers.py`로 생성된 청크 DataFrame (`chunk_text` 리스트 및 메타데이터)
- **다음 단계 연결 (Week 3)**: `chunk_df["chunk_text"].tolist()`를 Gemini 임베딩 모듈(`gemini_embeddings.py`)에 전달해 고차원 수치 벡터 행렬로 변환
