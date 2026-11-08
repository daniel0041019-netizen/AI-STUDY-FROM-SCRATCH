-- 1. 중복 제거 및 누락값 제외 조회
SELECT COUNT(DISTINCT country) AS unique_country_count
FROM users;

-- 2. 다양한 조건 및 패턴 필터링 (WHERE, LIKE, IN, IS NULL)
SELECT user_id, name, email
FROM users
WHERE status = 'active'
  AND age BETWEEN 20 AND 30
  AND name LIKE '김%'                  -- '김'으로 시작하는 이름
  AND country IN ('KR', 'US', 'JP')   -- 지정된 국가 목록
  AND phone IS NOT NULL;              -- 전화번호가 누락되지 않은 경우
