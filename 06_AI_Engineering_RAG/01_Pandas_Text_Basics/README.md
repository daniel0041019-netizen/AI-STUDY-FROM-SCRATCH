# Pandas Data Manipulation & Analysis Summary

> **Core Philosophy:** NumPy와 Matplotlib 기반의 Pandas 핵심 구조부터 정렬·필터링, 집계·그룹화·피벗 테이블, 인덱싱·슬라이싱, 시각화, 결측치 처리, 입출력, 그리고 `.str` 벡터화 문자열 전처리 패턴 정리

---

## 📌 Table of Contents
1. [Pandas 기초 & DataFrame 구조](#1-pandas-기초--dataframe-구조)
2. [DataFrame 정렬, 필터링 & 파생 변수](#2-dataframe-정렬-필터링--파생-변수)
3. [요약 통계, 집계, 그룹화 & 피벗 테이블](#3-요약-통계-집계-그룹화--피벗-테이블)
4. [인덱싱, 멀티인덱스 & 슬라이싱 (`.loc` / `.iloc`)](#4-인덱싱-멀티인덱스--슬라이싱-loc--iloc)
5. [데이터 시각화 (Matplotlib 연동)](#5-데이터-시각화-matplotlib-연동)
6. [결측치 처리, DataFrame 생성 & CSV 입출력](#6-결측치-처리-dataframe-생성--csv-입출력)
7. [문자열 전용 메서드 (`.str`) 전처리](#7-문자열 전용-메서드-str-전처리)

---

## 1. Pandas 기초 & DataFrame 구조

### 기반 라이브러리와 철학
- **NumPy 기반**: 데이터를 저장하는 다차원 배열 객체를 제공하여 수치 연산 및 배열 기반 처리를 지원
- **Matplotlib 기반**: 시각화를 위한 플로팅 및 차트 렌더링 기능과 결합
- **유연한 설계 철학**: 여러 접근 방식을 제공해 실전 분석의 자유도가 높으나, 초기 학습 곡선이 다소 존재

### 직사각형 데이터 (Rectangular Data) & 핵심 속성
- **구조**: 행(Row, 개별 관측값)과 열(Column, 변수/속성)로 구성된 표 형식 데이터(SQL 테이블, R DataFrame과 유사). 열 내부는 동일 타입, 열 간에는 서로 다른 타입 가능
- **탐색 메서드 및 속성**:
  - `.head()`: 앞부분 일부 행을 반환해 데이터 형태를 빠르게 확인
  - `.info()`: 열 이름, 데이터 타입, 결측치가 아닌 값의 개수 요약
  - `.shape`: `(행 수, 열 수)` 튜플을 반환하는 **속성**(괄호 없이 사용)
  - `.describe()`: 숫자형 열의 평균, 중앙값, `count`, 사분위수 등 요약 통계 제공
- **3가지 핵심 구성 요소**:
  - `.to_numpy()`: 데이터를 2차원 NumPy 배열로 반환
  - `.columns`: 열 레이블이 담긴 `Index` 객체
  - `.index`: 행 레이블이 담긴 `Index` 객체

---

## 2. DataFrame 정렬, 필터링 & 파생 변수

### 정렬 (`.sort_values()`)
- **단일/다중 기준 정렬**: `df.sort_values('열이름')` (기본 오름차순 `ascending=True`), `ascending=False`로 내림차순 지정
- **복수 열 및 방향 개별 제어**: `df.sort_values(['weight', 'height'], ascending=[True, False])`처럼 열 목록과 방향 리스트를 함께 전달

### 열 선택과 행 필터링
- **열 추출**: 단일 열 `df['name']`, 다중 열 `df[['name', 'weight']]`
- **조건 필터링**: 비교 연산(`>`, `<`, `==`), 날짜 문자열(`'YYYY-MM-DD'`) 비교, 다중 조건 결합 시 각 조건을 괄호로 감싸 `&` / `|` 사용 (`df[(조건1) & (조건2)]`)
- **범주형 포함 여부 (`.isin()`)**: `df[df['color'].isin(['Black', 'Brown'])]` 형태로 다중 후보 값 필터링

### 파생 변수 추가 & 체이닝
- **열 생성/변환**: `df['height_m'] = df['height_cm'] / 100`처럼 기존 열 연산으로 새 열 추가
- **복합 파이프라인**: 조건 필터링 → 정렬 → 열 선택을 연결해 원하는 분석 결과를 간결하게 도출

---

## 3. 요약 통계, 집계, 그룹화 & 피벗 테이블

### 요약 통계 & 누적 통계
- **기본/날짜 통계**: `.mean()`, `.median()`, `.sum()`, `.min()`, `.max()`, `.std()`, `.var()` 등을 숫자형 및 날짜/시간 열에 적용
- **맞춤 집계 (`.agg()`)**: 사용자 정의 함수나 통계 함수 리스트를 전달해 단일/복수 열에 동시 적용 (`sales['temperature_c'].agg([iqr, 'median'])`)
- **누적 통계**: `.cumsum()`, `.cummax()`, `.cummin()`, `.cumprod()`로 행 누적 흐름 계산

### 그룹화 (`.groupby()`) & 피벗 테이블 (`.pivot_table()`)
- **그룹별 집계**: `df.groupby('color')['weight'].mean()` 또는 다중 키 `df.groupby(['열1', '열2'])['집계할_열'].agg(['min', 'max', 'sum'])`
- **피벗 테이블**: `df.pivot_table(values='집계할열', index='행기준열', columns='열기준열', aggfunc='mean', fill_value=0, margins=True)`로 2차원 요약 표 구성 및 결측/총계 처리

---

## 4. 인덱싱, 멀티인덱스 & 슬라이싱 (`.loc` / `.iloc`)

### 명시적 인덱스 & 멀티인덱스
- **인덱스 제어**: `set_index()`로 특정 열을 인덱스로 승격, `reset_index()`(또는 `drop=True`)로 기본 인덱스로 복원
- **계층적 인덱스**: `set_index(['상위레벨_열', '하위레벨_열'])`로 다중 레벨 구성, `sort_index()`로 정렬
- **장단점 고려**: `.loc` 조회가 간결해지지만, Tidy Data 원칙이나 문법 일관성 측면에서 트레이드오프가 있으므로 코드 독해 및 목적에 맞춰 활용

### `.loc` vs `.iloc` 슬라이싱 규칙
| 구분 | `.loc` (레이블 기반) | `.iloc` (위치 번호 기반) |
| :--- | :--- | :--- |
| **기준** | 인덱스/열 이름(레이블) 및 조건 | 0부터 시작하는 정수 위치 인덱스 |
| **끝 경계 포함 여부** | **포함** (예: `'Labrador':'Poodle'`) | **미포함** (Python 슬라이싱과 동일) |
| **다중/시계열 활용** | 튜플 범위 지정, 날짜 구간(`'2014':'2016'`) 조회 지원 | 정수 위치 범위 및 열 위치 슬라이싱에 적합 |

---

## 5. 데이터 시각화 (Matplotlib 연동)

### 기본 차트 및 옵션
- **기본 설정**: `import matplotlib.pyplot as plt` 후 플롯을 그리고 `plt.show()`로 출력
- **주요 그래프**:
  - **히스토그램**: `df['height'].hist(bins=20)`로 수치 분포 확인
  - **막대 그래프**: 집계 결과에 `.plot(kind='bar', title='제목')` 적용
  - **선 그래프**: `df.plot(x='date', y='weight', kind='line', rot=45)`로 시계열 추이 시각화
  - **산포도**: `df.plot(x='height_cm', y='weight_kg', kind='scatter')`로 두 수치 변수 관계 파악
- **겹쳐 그리기 & 스타일링**: 여러 플롯을 이어 그린 뒤 `plt.legend([...])`로 범례를 추가하고, `alpha` 값으로 투명도를 조절해 중첩 데이터 가독성 확보

---

## 6. 결측치 처리, DataFrame 생성 & CSV 입출력

### 결측치 (`NaN`) 탐지 및 처리
- **탐지**: `.isna()`로 불리언 마스크 생성, `.isna().any()`로 열별 결측 유무 확인, `.isna().sum()`으로 열별 결측 개수 집계
- **처리**: `.dropna(subset=['review'])`로 대상 열 결측 행 제거하거나, `.fillna(...)`로 지정 값/통계치 대체

### DataFrame 직접 생성 & 파일 입출력
- **직접 생성**: 행 단위(`딕셔너리의 리스트`) 또는 열 단위(`리스트의 딕셔너리`) 구조로 구성 후 `pd.DataFrame()`에 전달
- **CSV 입출력**: `pd.read_csv('경로.csv')`로 불러오고, 가공 후 `.to_csv('출력.csv')`로 저장

---

## 7. 문자열 전용 메서드 (`.str`) 전처리

### 벡터화 연산의 장점
Series에 `.str` 접근자를 사용하면 Python 루프나 `.apply()` 없이 내부 벡터화 연산으로 문자열 시리즈를 일괄 처리할 수 있습니다.

### 주요 `.str` 메서드 패턴
- **공백 제거**: `.str.strip()`, `.str.lstrip()`, `.str.rstrip()`으로 앞뒤 공백 및 개행/탭 정리
- **치환 및 정규표현식 정제**: `.str.replace('기존', '신규')` 및 `df['review'].str.replace(r'[^가-힣a-zA-Z0-9 ]', '', regex=True)`로 노이즈 문자 제거
- **키워드 필터링**: `df['review'].str.contains('불만', case=False, na=False)`로 대소문자 구분 옵션과 결측 안전 처리(`na=False`)를 결합해 조건 추출
- **대소문자 변환**: `.str.lower()`, `.str.upper()`로 텍스트 표기 통일
- **길이 및 분할**: `.str.len()`으로 글자 수 기준 필터링, `.str.split(',')` 및 `.str[0]`으로 구분자 기반 토큰 분리
