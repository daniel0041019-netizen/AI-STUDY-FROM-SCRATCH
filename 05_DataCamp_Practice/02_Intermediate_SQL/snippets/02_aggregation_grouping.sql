-- 그룹화 및 조건 필터링 (GROUP BY & HAVING)
-- 각 카테고리별 평균 가격을 구하되, 평균 가격이 50 이상인 카테고리만 반올림하여 조회
SELECT 
    category,
    COUNT(*) AS item_count,
    ROUND(AVG(price), 2) AS avg_price
FROM products
WHERE status = 'sale'
GROUP BY category
HAVING AVG(price) >= 50
ORDER BY avg_price DESC;              -- 내림차순 정렬
