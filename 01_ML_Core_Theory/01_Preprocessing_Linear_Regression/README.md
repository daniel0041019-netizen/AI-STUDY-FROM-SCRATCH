# 🏠 Kaggle House Prices: 규제 모델 최적화 및 파이프라인 트러블슈팅

> **"핸즈온 머신러닝" 1~4장 학습을 기반으로 일반 선형 모델의 한계를 진단하고, 수학적 규제 모델(Lasso)의 변수 선택 메커니즘을 직접 실험 및 증명한 보고서입니다.**

실무에서 발생하기 쉬운 **데이터 누수(Data Leakage)** 문제를 코드로 발견하여 구조적으로 해결하고, 단순 코드 구현을 넘어 가우스-마르코프 정리 및 비용함수의 기하학적 구조를 바탕으로 모델을 최적화한 과정을 기록했습니다.

---

## 🎯 핵심 실험 및 트러블슈팅 요약

상세한 이론적 배경과 분석 리포트는 [📄 기술 블로그 포스팅/문서](https://velog.io/@daniel3721/Hands-on-ML-01.-%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0-%ED%8A%9C%EB%8B%9D%EA%B3%BC-%EC%A0%84%EC%B2%98%EB%A6%AC-%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8%EC%9D%98-%ED%9E%98)에 별도 정리되어 있습니다.

### 🚨 1. 데이터 스누핑(Data Snooping) 발견 및 격리 (회귀)
* **문제 진단:** Train/Validation 세트 분리 전 전체 데이터 `X`에 대해 `full_pipeline.fit_transform(X)`을 수행하는 결함 식별. 검증 데이터의 통계량(중앙값, 평균, 표준편차)이 훈련 과정에 유출(Leakage)되는 맹점 확인.
* **구조적 해결:** 무가공 Raw Data 상태에서 `train_test_split`을 선행하여 데이터를 엄격히 격리. 이후 전처리 파이프라인과 예측 모델을 하나의 **메가 파이프라인(Mega Pipeline)**으로 결합하여 오직 훈련 세트만으로 `fit()`이 수행되도록 설계 구조 개선.

### 📈 2. 타겟 변수 로그 변환 ($log1p$)의 수학적 당위성 (회귀)
* 선형 회귀(OLS)의 대전제인 **가우스-마르코프 정리(정규성, 등분산성)** 충족 목적.
* 우측으로 꼬리가 긴(Skewed) 주택 가격 분포를 로그 스케일로 압축함으로써 고가 주택군에서 발생하는 이분산성(Heteroskedasticity)을 차단하고 모델의 안정적 최적화 기반 마련.

### 🎯 3. Lasso의 최적점 ($\alpha=0.001$)과 207개의 허수 특성 제거 (회귀)
* 5-Fold 교차 검증(`GridSearchCV`) 결과 최적 규제 강도 `alpha=0.001` 도출.
* 미미한 규제 강도임에도 L1 규제 특유의 기하학적 메커니즘에 의해 원-핫 인코딩으로 폭발했던 **287개 특성 중 207개의 가중치가 0으로 수렴하며 탈락 (80개 핵심 특성 생존)**. 다중공선성을 유발하는 노이즈 변수를 걸러내는 '자동 피처 엔지니어링' 효과를 수치로 증명.

### 📊 4. 멀티모달(Multimodal) 분포와 버킷타이징의 한계 극복 (피처 엔지니어링)
* **비선형성 우회:** 선형 모델의 단조성(Monotonicity) 가정을 깨뜨리는 낙타 등 형태의 다봉(Multimodal) 분포 제어 목적.
* **K-Means 기반 구간화:** 단순 등간격 분할이 아닌 `KBinsDiscretizer(strategy='kmeans')`를 도입하여 실제 데이터 밀집 지역의 중심점(Centroid)을 포착, 원-핫 인코딩을 통해 선형 모델 내에서 완벽한 비선형 회귀(Piecewise Constant Regression)를 구현.

### 📉 5. ROC 곡선 vs PR 곡선: 불균형 데이터 평가 프레임워크 (분류 메트릭)
* **착시 현상 규명:** 대량의 정상 데이터(TN)로 인해 오탐(FP)의 심각성이 은폐되어 ROC-AUC 점수가 왜곡(0.95 달성 후 정밀도 처참)되는 수식적 맹점 파악.
* **이원화 전략 구축:** 균형 데이터(5:5)에는 이진 분류 변별력을 측정하는 **ROC 곡선**을, 금융 사기 탐지(FDS)나 의료 진단과 같은 극단적 불균형(Imbalanced) 데이터에는 TN 노이즈를 완벽히 걷어내고 현미경을 들이대는 **PR 곡선(PR-AUC)**을 평가지표로 채택하도록 격리 전략 수립.

---

## 📊 실험 결과 및 비교 (Analysis)

| 모델 (Model) | Train RMSE | Validation RMSE | 진단 및 평가 |
| :--- | :---: | :---: | :--- |
| **Linear Regression (OLS)** | \$17,432.36 | \$22,724.31 | 차원 폭발로 인한 치명적인 **과대적합(Overfitting)** 발생 |
| **Optimized Lasso ($\alpha=0.001$)** | \$26,360.87 | \$25,157.79 | 207개 노이즈 변수 제거를 통한 **일반화 성능 확보 및 격차 최소화** |

### 📈 Lasso 상위 15개 핵심 예측 특성
모델 최적화 후 가중치(Coefficient) 절대값 기준 상위 15개 변수 추출 시각화 결과입니다.

![Lasso Top Features](./lasso_top_features.png)

---

## 🛠️ 기술 스택 및 라이브러리
* **Language:** Python 3.x
* **Data Science:** `pandas`, `numpy`, `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`

## 💻 핵심 코드 아키텍처 (`house_price_pipeline.py`)

```python
# 1. 전처리 파이프라인 설계 (ColumnTransformer를 통한 누수 방지 프레임 구축)
full_pipeline = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_attribs),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat_attribs)
])

# 2. 하이퍼파라미터 최적화 (GridSearchCV + 5-Fold CV)
lasso_grid = GridSearchCV(
    Lasso(max_iter=20000), 
    param_grid={'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 1.0, 10.0, 100.0]}, 
    cv=5,
    scoring='neg_root_mean_squared_error'
)
