-- 1. 조건문 (CASE WHEN) 응용: 조건별 집계
SELECT 
    department,
    SUM(CASE WHEN status = 'Active' THEN 1 ELSE 0 END) AS active_count,
    SUM(CASE WHEN status = 'Resigned' THEN 1 ELSE 0 END) AS resigned_count
FROM employees
GROUP BY department;

-- 2. 공통 테이블 표현식 (CTE)
WITH high_value_orders AS (
    SELECT user_id, total_amount
    FROM orders
    WHERE total_amount >= 1000
)
SELECT user_id, AVG(total_amount) AS avg_high_amount
FROM high_value_orders
GROUP BY user_id;

-- 3. 윈도우 함수 (Window Functions - 순위 및 그룹별 평균)
SELECT 
    department,
    employee_name,
    salary,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_salary_rank,
    ROUND(AVG(salary) OVER (PARTITION BY department), 2) AS dept_avg_salary
FROM employees;
