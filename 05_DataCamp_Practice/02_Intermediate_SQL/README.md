# SQL & Data Manipulation Summary

> **Core Philosophy:** 데이터베이스 기본 구조부터 필터링, 조인(Join), 서브쿼리, 윈도우 함수까지 핵심 SQL 개념과 최적화 패턴 정리

---

## 1. 데이터베이스 기초 및 데이터 유형

* **기초 개념**
  * 데이터베이스는 **테이블(소문자 권장)** 단위로 구성
  * **행(Row / Record)**, **열(Column / Field)**, 고유 식별자인 **키(Key)**로 관리

* **스토리지**: 데이터는 서버 하드디스크에 저장되며, 네트워크를 통해 다수의 동시 요청 및 협업을 처리

* **주요 데이터 유형**
  * **`VARCHAR`**: 유연한 가변 길이 문자열 데이터
  * **`INT`**: 정수 데이터 (약 -20억 ~ +20억)
  * **`NUMERIC` / `FLOAT`**: 소수점 포함 실수형 데이터

* **스키마 (Schema)**: 테이블 구조, 필드 데이터 유형, 테이블 간 관계를 명시한 데이터베이스 설계도

---

## 2. Intermediate SQL: 데이터 조회 및 필터링

### 기본 조회 & 집계
* **`COUNT()`**: 누락값(`NULL`)을 제외한 행 개수 산출 (`COUNT(*)`는 `NULL` 포함)
* **`DISTINCT`**: 중복을 제거한 고유값 조회 (예: `COUNT(DISTINCT country)`)

### SQL 쿼리 실행 순서
```sql
FROM ➔ WHERE ➔ GROUP BY ➔ HAVING ➔ SELECT ➔ ORDER BY ➔ LIMIT
```

### 필터링 조건 (`WHERE`)
* **비교/범위 연산자**: `=`, `!=` (`<>`), `>`, `<`, `>=`, `<=`, `BETWEEN A AND B`
* **텍스트/패턴 연산자**
  * **`LIKE`**: 와일드카드 사용 (`%`: 0개 이상의 문자, `_`: 단일 문자)
  * **`NOT LIKE`**: 패턴 제외
  * **`IN (...)`**: 지정한 목록에 포함되는 데이터 필터링
* **결측치 연산자**: `IS NULL`, `IS NOT NULL`

---

## 3. 집계, 정렬 및 그룹화

* **집계 함수**: `AVG()`, `SUM()`, `MIN()`, `MAX()`, `COUNT()`
  * **숫자형**: 산술 연산 가능
  * **문자/날짜형**: 알파벳/시간 순서 비교 지원

* **소수점 처리**: `ROUND(값, 소수점_자릿수)` 활용

* **정렬 (`ORDER BY`)**: `ASC` (오름차순, 기본값) / `DESC` (내림차순) — 다중 필드 정렬 지원

* **그룹화 (`GROUP BY` & `HAVING`)**
  * **`GROUP BY`**: 범주형 필드 기준 요약 (SELECT 절의 비집계 필드는 반드시 포함)
  * **`HAVING`**: 집계 결과 및 그룹화된 데이터 필터링 (`WHERE`는 개별 레코드 필터링)

---

## 4. Joining Data: 테이블 결합 및 집합 연산

### 조인 (Join) 유형
* **`INNER JOIN`**: 두 테이블의 키 값이 일치하는 레코드만 결합 (`ON` 또는 `USING` 활용)
* **`LEFT JOIN` / `RIGHT JOIN`**: 기준 테이블의 전체 레코드 유지, 미매칭 시 `NULL` 표기
* **`FULL JOIN`**: 양쪽 테이블의 모든 레코드 포함 (미매칭 영역 `NULL`)
* **`CROSS JOIN`**: 두 테이블의 카테시안 곱(모든 가능한 조합) 생성
* **`SELF JOIN`**: 동일한 테이블 내 데이터 비교를 위해 별칭(`AS`)을 활용해 자가 조인

### 관계 유형
1. **일대다 (One-to-Many)**: 가장 흔한 관계 (예: 아티스트-노래, 작가-책)
2. **일대일 (One-to-One)**: 고유한 쌍 관계 (예: 사람-지문)
3. **다대다 (Many-to-Many)**: 상호 복합 연결 (예: 언어-국가)

### 집합 연산 (Set Operations)
* **`UNION`**: 중복 제거 병합
* **`UNION ALL`**: 중복 포함 병합
* **`INTERSECT`**: 교집합
* **`EXCEPT`**: 차집합
* **주의사항**: 결합하는 `SELECT` 구문의 열 개수와 데이터 타입이 서로 일치해야 함

---

## 5. Advanced Data Manipulation

### 서브쿼리 (Subquery)
* **위치별 활용**
  * **`WHERE`**: 조건 목록 및 범위 필터링
  * **`FROM`**: 임시 테이블(Derived Table) 형태로 활용
  * **`SELECT`**: 단일 스칼라 값 비교
* **상관 서브쿼리 (Correlated Subquery)**: 메인 쿼리의 행마다 반복 실행되어 조건을 비교하는 서브쿼리

### 조건문 (`CASE WHEN`)
```sql
CASE 
    WHEN 조건1 THEN 결과1
    WHEN 조건2 THEN 결과2
    ELSE 기본값 
END
```
* `SUM(CASE WHEN ... THEN 1 END)` 형태로 조건별 집계 연산에 응용 가능

### 공통 테이블 표현식 (CTE)
```sql
WITH cte_name AS (
    SELECT column1, column2
    FROM table_name
)
SELECT * FROM cte_name;
```
* `WITH` 구문을 활용하여 메인 쿼리 전에 가상 테이블을 선언함으로써 가독성 및 재사용성 향상

### 윈도우 함수 (Window Functions)
```sql
함수() OVER (PARTITION BY 컬럼명 ORDER BY 컬럼명)
```
* 원본 레코드 행 수(지점)를 유지하면서 그룹별 집계/순위(`RANK()`, `AVG()` 등) 계산 가능



