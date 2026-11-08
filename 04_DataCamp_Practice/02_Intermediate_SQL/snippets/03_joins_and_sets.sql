-- 1. INNER JOIN & LEFT JOIN 예시
SELECT 
    o.order_id,
    u.name AS user_name,
    o.total_amount
FROM orders o
INNER JOIN users u ON o.user_id = u.user_id;

-- 2. 자가 조인 (SELF JOIN) 예시: 직원과 직원의 매니저 매칭
SELECT 
    e.employee_name AS employee,
    m.employee_name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.employee_id;

-- 3. 집합 연산 (UNION ALL)
SELECT user_id, email FROM 2025_users
UNION ALL
SELECT user_id, email FROM 2026_users;
