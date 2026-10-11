import time
import pandas as pd

# 1. 테스트용 대용량 DataFrame 생성 (1만 행)
df = pd.DataFrame({
    'A': range(10000),
    'B': range(10000, 20000)
})

# --- [비교 1] iterrows() : 가장 느림 (Series 객체 생성 오버헤드) ---
start_time = time.time()
total_iterrows = 0
for row in df.iterrows():
    # row[1]은 Series 객체
    total_iterrows += row[1]['A'] + row[1]['B']
print(f"iterrows 소요 시간: {time.time() - start_time:.4f}초")


# --- [비교 2] itertuples() : 훨씬 빠름 (네이티브 튜플 반환) ---
start_time = time.time()
total_itertuples = 0
for row in df.itertuples():
    # row.A, row.B 형태로 접근 가능
    total_itertuples += row.A + row.B
print(f"itertuples 소요 시간: {time.time() - start_time:.4f}초")


# --- [비교 3] 벡터화 연산 (Vectorization) : 압도적으로 가장 빠름 ---
start_time = time.time()
total_vectorized = (df['A'] + df['B']).sum()
print(f"벡터화 연산 소요 시간: {time.time() - start_time:.4f}초")
