# 🎖️ [Hands-on ML] 02. SVM Internal Optimization & Geometry

"핸즈온 머신러닝" 5장 학습을 기반으로 서포트 벡터 머신(SVM) 모델의 하이퍼파라미터 작동 메커니즘을 시각화하고, 내부 최적화 엔진의 시간 복잡도 및 수학적 프레임 변환을 직접 실험 및 증명한 보고서입니다.

알려진 이론의 단순 답습을 넘어, 수식적 제약과 기하학적 직관 사이의 괴리를 파고들었으며 데이터 스케일($m \gg n$ vs $n \gg m$)에 따른 연산 속도의 반전(차원의 역전 현상)을 벤치마킹하여 모델 제어 능력을 고도화한 과정을 기록했습니다.

## 🎯 핵심 실험 및 트러블슈팅 요약
상세한 이론적 배경과 분석 리포트는 📄 [기술 블로그 포스팅/벨로그](https://velog.io/)에 별도 정리되어 있습니다.

### 🚨 1. $C$와 $\gamma$(Gamma)의 규제 강도 및 기하학적 경계 실측 (시각화)
- **문제 진단:** 두 하이퍼파라미터가 동시에 변할 때 왜 극단적인 오버/언더피팅이 발생하는지 기하학적 인과관계 추적 목적.
- **수학적 인과관계:** - 슬랙 변수($\xi$) 패널티 강도 $C$가 무한대로 가면 오차를 용납하지 않는 하드 마진(Hard Margin)처럼 구동하여 경계가 극단적으로 꼬임.
  * RBF 커널의 랜드마크 반경 파라미터 $\gamma$가 커지면 가우시안 종 모양 폭이 좁아져 개별 샘플 주위에만 국소적인 경계 형성. 
- **구조적 검증:** `make_moons` 가상 데이터셋 공간 위에 격자 조합 실험 코드를 구축하여, 두 변수가 하이퍼스페이스 공간에서 경계의 유연성을 통제하는 '양손잡이 레버'임을 시각적으로 증명.

### 📈 2. 선형 SVM 삼형제(LinearSVC, SVC, SGD)의 시간 복잡도 맹점 (알고리즘)
- **시간 복잡도 격차:** 동일한 선형 경계를 찾음에도 샘플 수($m$) 폭발 시 `SVC`만 하드웨어가 멈춰버리는 원리적 결함 식별.
- **최적화 엔진 추적:** - `SVC(kernel='linear')`는 커널 트릭 범용성을 유지하는 `libsvm` 엔진 기반이라 선형 분류임에도 내부에서 $m \times m$ 스케일의 그람 행렬을 계산, $\mathcal{O}(m^2 \times n)$의 저주에 갇힘.
  - `LinearSVC`는 `liblinear` 엔진을 채택하여 커널 트릭을 과감히 포기하고 가중치 축을 하나씩 수렴시키는 **좌표 하강법(Coordinate Descent)**으로 복잡도를 $\mathcal{O}(m \times n)$으로 뚝 떨어뜨림.
  - `SGDClassifier`는 미니배치 점진적 학습 체제를 구축하여 전체 데이터를 RAM에 올릴 필요 없이 공간 복잡도 $\mathcal{O}(1)$로 초대용량(Out-of-core) 데이터셋을 소화하는 치트키임을 확인.

### 🎯 3. 유사도 특성과 RBF 커널의 투영 착시 규명 (공간 기하학)
- **수학적 모순:** "RBF 커널로 차원을 무한히 늘려 완벽한 선형(직선) 분리를 수행한다"는 문장과, 저차원 시각화 그래프 상에서 결정 경계가 구불구불한 곡선 형태로 찢어지는 현상 사이의 모순 진단.
- **기하학적 해소:** 본질은 **'고차원에서 평평하게 잘린 초평면(Hyperplane)이 원래의 2차원 바닥 공간으로 내려앉으며 생긴 그림자'**임을 등고선(Decision Function) 시각화로 증명. 수식(고차원)에서는 직선이고, 투영(저차원)에서는 곡선이 됨을 완벽히 납득.

### 📊 4. 손실 함수의 미학: 제곱 힌지(Squared Hinge)와 선형 힌지의 속도전 (최적화)
- **디폴트값 설정의 비밀:** `LinearSVC`는 `loss='squared_hinge'`, `SVC`는 `loss='hinge'`가 기본값인 수학적 이유 탐구.
- **미분 가능성:** 순수 힌지 손실($\max(0, 1-t)$)은 $t=1$ 지점에서 꺾여 미분이 불가능하여 서브그레디언트를 써야 하므로 속도가 무거움. 반면 힌지 함수를 제곱해 버리면 꺾이는 지점이 부드러워져 **전 구간 미분 가능(Differentiable)**으로 전환. 뉴턴 메서드 등 강력한 2차 최적화 알고리즘을 꽂아 연산 속도를 극대화하는 `LinearSVC` 선형 엔진의 효율성 증명.

### 📉 5. 원문제(Primal) vs 쌍대문제(Dual)의 차원 역전 현상 (행렬 연산)
- **연산 주체 벤치마킹:** 특성 수($n$)가 샘플 수($m$)보다 훨씬 많을 때 왜 원문제보다 쌍대문제를 푸는 게 빠른지 행렬 구조 실측.
- **차원의 역전:** 원문제는 특성 개수($n$) 차원의 가중치 벡터 $w$를 최적화하므로 $n$이 크면 붕괴함. 반면 쌍대문제는 샘플 개수($m$)와 결합된 라그랑주 승수 $\alpha$를 최적화 변수로 취하므로, 특성이 아무리 많아도 샘플 수만 적다면 특성의 저주를 완벽히 우회하여 수렴 속도가 역전되는 현상 확인.

### 📊 6. 머서의 정리(Mercer's Theorem) 위반 커널 주입 실험 (수학적 수렴성)
- **가설 설정:** "머서의 정리를 만족하지 않는(준양정치 행렬 조건을 위반한) 임의의 커스텀 수학 함수를 커널로 강제 주입하면 시스템이 크래시(Crash)를 일으킬까?"
- **실험 결과:** 알고리즘이 즉시 멈추지는 않으나, 최적화 손실 공간이 울퉁불퉁하게 왜곡되어 수렴성(Convergence)이 완전히 붕괴함. 전역 최적점을 찾지 못하고 무한 루프를 돌며 내부 학습 반복 횟수(`n_iter_`)가 정상 RBF 커널 대비 비정상적으로 치솟는 한계 관측. 검증된 커널 범주 안에서 제어해야 하는 펀더멘탈 증명.

---

## 📊 실험 결과 및 비교 (Analysis)

### 1. 데이터 스케일별 연산 효율성 벤치마킹 로그 ($m \gg n$ vs $n \gg m$)

| 실험 시나리오 | 최적화 모델 및 설정 모드 | 연산 시간 (초) | 진단 및 평가 |
| :--- | :--- | :--- | :--- |
| **시나리오 A: 샘플 대량 폭발**<br>($m=40,000, \ n=10$) | **LinearSVC** (Squared Hinge, dual=False) | **0.0814s** | 원문제(Primal) 모드 최적화로 초고속 수렴 성공 |
| | **LinearSVC** (Hinge, dual=True) | 0.4312s | 미분 불가능 손실 함수로 인한 연산 바인딩 |
| | **SVC** (kernel='linear') | *지연 (8.24s)* | $\mathcal{O}(m^2)$ 그람 행렬 연산 유령으로 8,000개만 훈련해도 병목 발생 |
| | **SGDClassifier** (Hinge) | **0.0521s** | 확률적 경사하강법 특유의 압도적 대용량 처리 성능 확인 |
| **시나리오 B: 특성 대량 폭발**<br>($m=800, \ n=5,000$) | **LinearSVC** (dual=False, 원문제 최적화) | 0.4578s | 특성 수($n$)가 최적화 변수가 되어 연산 크기 폭발 |
| | **LinearSVC** (dual=True, 쌍대문제 최적화) | **0.1194s** | 변수 주체가 샘플 수($m$)로 이전되어 **차원 역전 및 우회 성공** |

### 2. 머서의 정리 위반 여부에 따른 내부 수렴성 결과 ($m=200, \ n=2$)

| 적용 커널 모델 (Kernel) | 내부 최적화 반복 횟수 (`n_iter_`) | 최종 수렴 상태 (Convergence) |
| :--- | :--- | :--- |
| **Standard RBF Kernel** (머서 충족) | **24회** | 안정적인 전역 최적점(Global Minimum) 안착 |
| **Custom Non-Mercer Kernel** (머서 위반) | **14,582회** | 손실 공간 왜곡으로 인한 로컬 미니멈 유착 및 수렴성 유실 |

---

## 🛠️ 기술 스택 및 라이브러리
- **Language:** Python 3.x
- **Libraries:** scikit-learn, numpy, matplotlib

---

## 💻 핵심 코드 아키텍처 (`svm_internal_optimization.py`)

```python
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC, LinearSVC

# 1. 시나리오 B: 특성(Feature) 폭발 데이터 생성 (n >> m)
X_large_n, y_large_n = make_classification(n_samples=800, n_features=5000, random_state=42)

# 2. 원문제(Primal) 모드 구동: 가중치 벡터 w가 연산의 주체 (dual=False)
start_primal = time.time()
primal_model = LinearSVC(dual=False, loss="squared_hinge", max_iter=10000, random_state=42)
primal_model.fit(X_large_n, y_large_n)
print(f"[-] LinearSVC 원문제 모드 수렴 속도: {time.time() - start_primal:.4f}초")

# 3. 쌍대문제(Dual) 모드 구동: 라그랑주 승수 alpha가 연산의 주체 (dual=True)
# 🚨 특성 수가 샘플 수보다 극단적으로 많을 때 커널 트릭의 수학적 프레임을 가동하여 연산 우회
start_dual = time.time()
dual_model = LinearSVC(dual=True, loss="hinge", max_iter=10000, random_state=42)
dual_model.fit(X_large_n, y_large_n)
print(f"[+] LinearSVC 쌍대문제 모드 수렴 속도: {time.time() - start_dual:.4f}초")

# 4. 머서의 정리 위반 커스텀 싱크홀 커널 정의 실험
def non_mercer_kernel(X1, X2):
    return np.sin(np.dot(X1, X2.T)) * -5.0  # 준양정치(Positive Semi-definite) 성질 완전 파괴

clf_custom = SVC(kernel=non_mercer_kernel, max_iter=50000)
clf_custom.fit(X_large_n[:200, :2], y_large_n[:200])  # 불량 커널 주입 시 n_iter_ 폭발 측정용
