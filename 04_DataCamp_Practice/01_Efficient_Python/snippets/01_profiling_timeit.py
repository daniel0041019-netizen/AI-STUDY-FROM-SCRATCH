import timeit

# 테스트할 데이터 준비
data = range(1000)

# 방법 A: 일반적인 for 루프
def method_for_loop():
    result = []
    for i in data:
        result.append(i * 2)
    return result

# 방법 B: 리스트 컴프리헨션 (List Comprehension)
def method_comprehension():
    return [i * 2 for i in data]

# 1. IPython / Jupyter Notebook 환경인 경우 (Magic Command)
# %timeit [i * 2 for i in data]
# %timeit method_for_loop()

# 2. 일반 Python 스크립트 환경인 경우 (timeit 모듈)
time_loop = timeit.timeit(stmt="method_for_loop()", globals=globals(), number=10000)
time_comp = timeit.timeit(stmt="method_comprehension()", globals=globals(), number=10000)

print(f"For Loop 소요 시간: {time_loop:.5f초}")
print(f"List Comprehension 소요 시간: {time_comp:.5f초}")
