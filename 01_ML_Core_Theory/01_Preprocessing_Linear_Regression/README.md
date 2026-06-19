# Kaggle House Prices: 규제 모델 최적화 및 파이프라인 트러블슈팅

"핸즈온 머신러닝" 1~4장 학습을 기반으로 일반 선형 모델의 한계를 진단하고, 수학적 규제 모델(Lasso)의 변수 선택 메커니즘을 직접 실험 및 증명한 보고서입니다.

실무에서 발생하기 쉬운 **데이터 누수(Data Leakage)** 문제를 코드로 발견하여 구조적으로 해결하고, 단순 코드 구현을 넘어 가우스-마르코프 정리 및 비용함수의 기하학적 구조를 바탕으로 모델을 최적화한 과정을 기록했습니다.

---

## 핵심 실험 및 트러블슈팅 요약

상세한 이론적 배경과 분석 리포트는 [📄 기술 블로그 포스팅/문서](https://velog.io/@daniel3721/Hands-on-ML-01.-%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0-%ED%8A%9C%EB%8B%9D%EA%B3%BC-%EC%A0%84%EC%B2%98%EB%A6%AC-%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8%EC%9D%98-%ED%9E%98)에 별도 정리되어 있습니다.

### 1. 데이터 스누핑(Data Snooping) 발견 및 격리 (회귀)
* **문제 진단:** Train/Validation 세트 분리 전 전체 데이터 `X`에 대해 `full_pipeline.fit_transform(X)`을 수행하는 결함 식별. 검증 데이터의 통계량(중앙값, 평균, 표준편차)이 훈련 과정에 유출(Leakage)되는 맹점 확인.
* **구조적 해결:** 무가공 Raw Data 상태에서 `train_test_split`을 선행하여 데이터를 엄격히 격리. 이후 전처리 파이프라인과 예측 모델을 하나의 **메가 파이프라인(Mega Pipeline)**으로 결합하여 오직 훈련 세트만으로 `fit()`이 수행되도록 설계 구조 개선.

### 2. 타겟 변수 로그 변환 ($log1p$)의 수학적 당위성 (회귀)
* 선형 회귀(OLS)의 대전제인 **가우스-마르코프 정리(정규성, 등분산성)** 충족 목적.
* 우측으로 꼬리가 긴(Skewed) 주택 가격 분포를 로그 스케일로 압축함으로써 고가 주택군에서 발생하는 이분산성(Heteroskedasticity)을 차단하고 모델의 안정적 최적화 기반 마련.

### 3. Lasso의 최적점 ($\alpha=0.001$)과 207개의 허수 특성 제거 (회귀)
* 5-Fold 교차 검증(`GridSearchCV`) 결과 최적 규제 강도 `alpha=0.001` 도출.
* 미미한 규제 강도임에도 L1 규제 특유의 기하학적 메커니즘에 의해 원-핫 인코딩으로 폭발했던 **287개 특성 중 207개의 가중치가 0으로 수렴하며 탈락 (80개 핵심 특성 생존)**. 다중공선성을 유발하는 노이즈 변수를 걸러내는 '자동 피처 엔지니어링' 효과를 수치로 증명.

### 4. 멀티모달(Multimodal) 분포와 버킷타이징의 한계 극복 (피처 엔지니어링)
* **비선형성 우회:** 선형 모델의 단조성(Monotonicity) 가정을 깨뜨리는 낙타 등 형태의 다봉(Multimodal) 분포 제어 목적.
* **K-Means 기반 구간화:** 단순 등간격 분할이 아닌 `KBinsDiscretizer(strategy='kmeans')`를 도입하여 실제 데이터 밀집 지역의 중심점(Centroid)을 포착, 원-핫 인코딩을 통해 선형 모델 내에서 완벽한 비선형 회귀(Piecewise Constant Regression)를 구현.

### 5. ROC 곡선 vs PR 곡선: 불균형 데이터 평가 프레임워크 (분류 메트릭)
* **착시 현상 규명:** 대량의 정상 데이터(TN)로 인해 오탐(FP)의 심각성이 은폐되어 ROC-AUC 점수가 왜곡(0.95 달성 후 정밀도 처참)되는 수식적 맹점 파악.
* **이원화 전략 구축:** 균형 데이터(5:5)에는 이진 분류 변별력을 측정하는 **ROC 곡선**을, 금융 사기 탐지(FDS)나 의료 진단과 같은 극단적 불균형(Imbalanced) 데이터에는 TN 노이즈를 완벽히 걷어내고 현미경을 들이대는 **PR 곡선(PR-AUC)**을 평가지표로 채택하도록 격리 전략 수립.

---

## 핵심 실증 실험 (Experiments & Results)

### 실험 1. 데이터 스누핑 차단 전처리 분리 및 타겟 변수 로그 변환 ($log1p$) 효과 검증
* **파일명:** `pipeline_gridsearch.ipynb` (섹션 1)
* **실험 목적 (Why):**
  * 전처리 통계량이 검증 세트로 유출되어 발생하는 모델 성능 왜곡(Data Leakage)을 차단합니다.
  * 왜도가 심한 타겟 변수(SalePrice)의 분포를 정규분포 형태로 변환하여 최소제곱법(OLS) 추정량의 가우스-마르코프 최적 조건(정규성)을 강제 충족시킵니다.
* **실험 방법 (How):**
  * `train_test_split`을 스누핑 없이 순수 데이터 상태에서 선행 수행하여 $X_{\text{train}}$과 $X_{\text{val}}$을 분리합니다.
  * 타겟 변수 `y`를 `np.log1p()` 함수를 통해 로그 스케일로 압축하여 연산에 주입하고, 최종 평가 시 `np.expm1()`로 복원하여 원본 스케일 RMSE를 측정합니다.
* **실험 결과 (Result):**
  * 로그 변환을 통해 고가 주택군으로 갈수록 잔차의 분산이 커지던 이분산성(Heteroskedasticity)이 제어되었으며, 전처리 통계 정보의 유출이 엄격히 격리되어 최종 검증 세트 스코어의 현실적인 일반화 신뢰도를 확보했습니다.

---

### 실험 2. 원-핫 인코딩 차원 폭발에 따른 OLS 선형 회귀 과대적합(Overfitting) 실측
* **파일명:** `pipeline_gridsearch.ipynb` (섹션 3)
* **실험 목적 (Why):**
  * 규제가 없는 일반 선형 회귀 모델(Ordinary Least Squares)이 범주형 변수의 다량 인코딩으로 인해 특성이 폭발한 고차원 공간에서 얼마나 취약하게 무너지는지 확인합니다.
* **실험 방법 (How):**
  * `ColumnTransformer` 내의 `OneHotEncoder`를 구동하여 생성된 수많은 파생 범주 변수들을 포함한 전처리 완료 배열(`X_train_processed`, `X_val_processed`)을 추출합니다.
  * `LinearRegression` 모델을 학습시키고, 훈련 데이터와 검증 데이터의 복원 RMSE를 각각 실측하여 비교합니다.
* **실험 결과 (Result):**
  * **Train RMSE: \$17,432.36 vs Val RMSE: \$22,724.31**로 두 지표 간 극심한 간극이 발생했습니다. 
  * 모델이 고차원 다중공선성 노이즈에 완벽히 동화되어 훈련 셋의 세부 오차를 완벽히 외웠으나, 검증 셋에서는 전형적인 과대적합(Overfitting) 맹점을 드러냄을 숫자로 규명했습니다.

---

### 실험 3. Mega 파이프라인 기반 GridSearchCV 최적 규제 강도($\alpha$) 도출
* **파일명:** `pipeline_gridsearch.ipynb` (섹션 4)
* **실험 목적 (Why):**
  * 하이퍼파라미터 튜닝 시 발생할 수 있는 내부 교차 검증단에서의 데이터 스누핑을 차단하기 위해 전처리 프로세서와 예측 모델이 원자적으로 결합된 구조적 한계 최적화를 수행합니다.
* **실험 방법 (How):**
  * `ColumnTransformer` 기반 `preprocessor`와 `Lasso(max_iter=20000)` 모델을 단일 `Pipeline` 오브젝트로 체이닝합니다.
  * `model__alpha` 파라미터 그리드 `[0.001, 0.005, 0.01, 0.05, 0.1, 1.0, 10.0, 100.0]` 범위를 지정하고 5-Fold 교차 검증(`GridSearchCV`)을 통해 내부 검증 RMSE가 최소화되는 전역 최적점을 찾아냅니다.
* **실험 결과 (Result):**
  * 내부 파이프라인 메커니즘 연산을 통해 **최적의 하이퍼파라미터: alpha = 0.001**이 안정적으로 검출되었습니다. 검증 과정 중 모든 폴드 세트의 정보가 격리되어 안정적인 하이퍼스페이스 탐색이 수행됨을 확인했습니다.

---

### 실험 4. L1 Lasso 규제의 기하학적 메커니즘을 통한 가중치 희소화(Sparsity) 계측
* **파일명:** `pipeline_gridsearch.ipynb` (섹션 5)
* **실험 목적 (Why):**
  * Lasso 모델의 다이아몬드형($L_1$ norm) 제약 조건이 비용 함수 공간의 축과 만남으로써 유도되는 강력한 변수 자동 선택(Feature Selection) 능력을 통계 수치로 실증합니다.
* **실험 방법 (How):**
  * 최적 메가 파이프라인에서 인코더 결합을 통해 도출된 총 특성 리스트 `all_features`를 전수 복원합니다.
  * 최적화가 끝난 Lasso 모델 내부 가중치 배열(`model.coef_`)을 추적하여 계수(Coefficient) 값이 칼같이 **0으로 수렴하여 지워진 허수 특성**과, 최종 생존한 유효 특성의 개수를 산출합니다.
* **실험 결과 (Result):**
  * 인코딩과 파이프라인 연산으로 인해 폭발했던 **전체 287개 특성 중 오직 80개만 가중치를 부여받고 생존했으며, 나머지 207개의 노이즈 변수는 0으로 제거**되었습니다. 
  * 모델이 스스로 다중공선성을 유발하는 파편화 변수들을 정밀 컷오프(Cut-off)하여 수치적 안정성을 고도화하는 과정을 증명했습니다.

---

### 실험 5. 다봉(Multimodal) 변수의 K-Means 기반 구간화(Bucketizing) 우회 학습
* **파일명:** `pipeline_gridsearch.ipynb` (섹션 2)
* **실험 목적 (Why):**
  * 일반적인 등간격(Uniform) 분할이나 분위수(Quantile) 분할이 무시하기 쉬운 밀집 지역의 분포 밸리를 반영하여 선형 모델이 비선형 형태의 회귀 곡선을 모사하도록 유도합니다.
* **실험 방법 (How):**
  * 다봉 성향이 뚜렷한 `GarageArea` 특성을 단독 격리 타겟으로 설정합니다.
  * `KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='kmeans')` 파이프라인을 구축하여 실제 군집 데이터의 센트로이드(Centroid) 중심점을 기반으로 영역을 구획하고 고차원 밀도 정보를 모델에 전달합니다.
* **실험 결과 (Result):**
  * 단순 단조 선형 트렌드만을 그리던 일반 회귀 모델과 달리, K-Means 구간화를 거쳐 원-핫 백터 형태로 매핑된 피처가 선형 비용 함수 내에서 구간별 상수 회귀(Piecewise Constant)를 완벽히 구현해 내며 특정 구간에서 변동하는 비선형 밀집 특성을 부드럽게 소화해 냈습니다.

---

## 실험 데이터 수치 요약 (Analysis)

### 1) 규제 유무에 따른 모델 일반화 성능 대조
| 모델 및 최적 옵션 (Model) | 학습 특성 수 (Features) | Train RMSE | Validation RMSE | 진단 및 평가 |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression (OLS)** | 287개 (전수) | **\$17,432.36** | \$22,724.31 | 차원 폭발로 인해 훈련 셋 오차를 맹목적으로 외운 **과대적합** 발생 |
| **Optimized Lasso ($\alpha=0.001$)** | **80개 (선택)** | \$26,360.87 | **\$25,157.79** | 207개 허수 변수 제거를 통한 **일반화 성능 확보 및 격차 최소화** |

### 2) 파이프라인 변수 희소화 통계
* **전체 파이프라인 주입 특성 개수:** `287개`
* **L1 규제에 의해 가중치 0 처리된 노이즈 특성 개수:** `207개`
* **최종 활성화 가중치(Active Coefficients) 개수:** `80개`

---

## 기술 스택 및 라이브러리
* **Language:** Python 3.10+
* **Data Science:** `pandas`, `numpy`, `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`

---

## 핵심 소스코드 아키텍처 (`pipeline_gridsearch.ipynb`)

```python
# 1. 고급 전처리 파이프라인 설계 (ColumnTransformer 통합 및 멀티모달 처리)
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_attribs),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat_attribs),
    ('multimodal', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('bucketizer', KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='kmeans'))
    ]), [multimodal_feature])
])

# 2. 메가 파이프라인 구성 및 하이퍼파라미터 최적화 (Data Snooping 원천 차단)
mega_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Lasso(max_iter=20000))
])

lasso_grid = GridSearchCV(
    mega_pipeline, 
    param_grid={'model__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 1.0, 10.0, 100.0]}, 
    cv=5,
    scoring='neg_root_mean_squared_error'
)

# 순수 Raw Data 상태에서 분할된 X_train만 주입하여 교차 검증 내 데이터 누수 방지
lasso_grid.fit(X_train, y_train)
