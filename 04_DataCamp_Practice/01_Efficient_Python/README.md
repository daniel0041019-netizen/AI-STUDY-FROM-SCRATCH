##  Efficient Python Code Summary

> **Pythonic Code:** 'The Zen of Python'의 철학을 따르며, 가독성과 성능(시간·메모리)을 모두 잡는 효율적인 코드 작성법

### 1. 코드 프로파일링 (Profiling)
* **`%timeit`**: 한 줄 또는 짧은 코드 블록의 실행 시간을 반복 측정 (IPython Magic Command)
* **`%lprun`**: 라인 단위로 실행 시간과 소요 비율을 정밀 측정 (`line_profiler` 필요)
* **`%mprun`**: 함수 단위의 메모리 사용량 측정 (`memory_profiler` 필요)


### 2. 표준 라이브러리를 통한 속도 개선
* **`itertools`**: 효율적인 조합, 순열, 누적 데이터 생성 (메모리 효율적인 제너레이터 반환)
* **`collections`**: 특화된 데이터 구조(예: `Counter`)를 활용해 카운팅 및 데이터 처리 속도 최적화
* **반복문 최적화**: 불필요한 `for` 루프 지양, 빌트인 함수 및 내장 메서드 활용

### 3. Pandas 성능 최적화
* **벡터화 연산 (Vectorization)**: Pandas/NumPy의 C 기반 벡터 연산을 최우선으로 활용
* **`itertuples()` vs `iterrows()`**: 행 단위 반복이 필요할 때, `iterrows`보다 속도가 훨씬 빠른 `itertuples` 사용 권장
* **`apply()`**: 적절한 축(axis) 설정과 함께 활용하여 데이터 프레임 효율적 조작
