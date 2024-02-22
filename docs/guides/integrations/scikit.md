---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Scikit-Learn

wandb를 사용하면 몇 줄의 코드만으로 scikit-learn 모델의 성능을 시각화하고 비교할 수 있습니다. [**예제 시도하기 →**](http://wandb.me/scikit-colab)

## :fire: 시작하기

### wandb에 가입하고 로그인하기

a) [**가입하기**](https://wandb.ai/site) 무료 계정 생성

b) `wandb` 라이브러리를 Pip으로 설치

c) 학습 스크립트에서 로그인하려면 www.wandb.ai에 로그인해 있어야 하며, [**인증 페이지**](https://wandb.ai/authorize)**에서 API 키를 찾을 수 있습니다.**

Weights and Biases를 처음 사용하는 경우 [퀵스타트](../../quickstart.md)를 확인해 보세요

<Tabs
  defaultValue="cli"
  values={[
    {label: '명령 줄', value: 'cli'},
    {label: '노트북', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```
pip install wandb

wandb login
```

  </TabItem>
  <TabItem value="notebook">

```notebook
!pip install wandb

wandb.login()
```

  </TabItem>
</Tabs>

### 메트릭 로깅

```python
import wandb

wandb.init(project="visualize-sklearn")

y_pred = clf.predict(X_test)
accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)

# 시간에 따른 메트릭 로깅 시 wandb.log 사용
wandb.log({"accuracy": accuracy})

# 혹은 학습 종료 시 최종 메트릭을 로깅하려면 wandb.summary도 사용할 수 있음
wandb.summary["accuracy"] = accuracy
```

### 플롯 생성

#### 단계 1: wandb를 가져오고 새 실행을 초기화합니다.

```python
import wandb

wandb.init(project="visualize-sklearn")
```

#### 단계 2: 개별 플롯 시각화

모델을 학습하고 예측값을 생성한 후 wandb에서 플롯을 생성하여 예측을 분석할 수 있습니다. 지원되는 차트의 전체 목록은 아래 **지원되는 플롯** 섹션을 참조하세요.

```python
# 단일 플롯 시각화
wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)
```

#### 혹은 한 번에 모든 플롯 시각화

W&B에는 `plot_classifier`와 같이 관련된 여러 플롯을 시각화하는 함수가 있습니다:

```python
# 모든 분류기 플롯 시각화
wandb.sklearn.plot_classifier(
    clf,
    X_train,
    X_test,
    y_train,
    y_test,
    y_pred,
    y_probas,
    labels,
    model_name="SVC",
    feature_names=None,
)

# 모든 회귀 플롯
wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test, model_name="Ridge")

# 모든 클러스터링 플롯
wandb.sklearn.plot_clusterer(
    kmeans, X_train, cluster_labels, labels=None, model_name="KMeans"
)
```

**혹은 기존의 matplotlib 플롯 시각화:**

Matplotlib에서 생성된 플롯도 W&B 대시보드에 로그할 수 있습니다. 이를 위해서는 먼저 `plotly`를 설치해야 합니다.

```
pip install plotly
```

마지막으로, 플롯은 W&B의 대시보드에 다음과 같이 로그할 수 있습니다:

```python
import matplotlib.pyplot as plt
import wandb

wandb.init(project="visualize-sklearn")

# 여기서 plt.plot(), plt.scatter() 등을 모두 수행합니다.
# ...

# plt.show() 대신에 실행:
wandb.log({"plot": plt})
```

### 지원되는 플롯

#### 학습 곡선

![](/images/integrations/scikit_learning_curve.png)

다양한 길이의 데이터세트에서 모델을 학습시키고, 학습 및 테스트 세트에 대한 교차 검증 점수와 데이터세트 크기의 그래프를 생성합니다.

`wandb.sklearn.plot_learning_curve(model, X, y)`

* model (clf 또는 reg): 학습된 회귀모델 또는 분류기를 입력합니다.
* X (arr): 데이터세트 특징.
* y (arr): 데이터세트 레이블.

#### ROC

![](/images/integrations/scikit_roc.png)

ROC 곡선은 참 긍정률(y축) 대 거짓 긍정률(x축)을 그립니다. 이상적인 점수는 TPR = 1이고 FPR = 0인데, 이는 왼쪽 상단의 점입니다. 일반적으로 ROC 곡선 아래 영역(AUC-ROC)을 계산하며, AUC-ROC이 클수록 더 좋습니다.

`wandb.sklearn.plot_roc(y_true, y_probas, labels)`

* y\_true (arr): 테스트 세트 레이블.
* y\_probas (arr): 테스트 세트 예측 확률.
* labels (list): 대상 변수(y)의 명명된 레이블.

#### 클래스 비율

![](/images/integrations/scikic_class_props.png)

학습 및 테스트 세트의 대상 클래스 분포를 그립니다. 불균형한 클래스를 감지하고 한 클래스가 모델에 지나치게 큰 영향을 미치지 않도록 하는 데 유용합니다.

`wandb.sklearn.plot_class_proportions(y_train, y_test, ['dog', 'cat', 'owl'])`

* y\_train (arr): 학습 세트 레이블.
* y\_test (arr): 테스트 세트 레이블.
* labels (list): 대상 변수(y)의 명명된 레이블.

#### 정밀도 재현율 곡선

![](/images/integrations/scikit_precision_recall.png)

다양한 임계값에 대한 정밀도와 재현율 사이의 트레이드오프를 계산합니다. 곡선 아래 영역이 클수록 재현율과 정밀도가 모두 높은 것을 의미하며, 여기서 높은 정밀도는 낮은 거짓 긍정률과 관련이 있고, 높은 재현율은 낮은 거짓 부정률과 관련이 있습니다.

두 점수 모두 높은 것은 분류기가 정확한 결과를 반환하고 있고(높은 정밀도), 모든 긍정적 결과의 대부분을 반환하고 있음을 나타냅니다(높은 재현율). PR 곡선은 클래스가 매우 불균형할 때 유용합니다.

`wandb.sklearn.plot_precision_recall(y_true, y_probas, labels)`

* y\_true (arr): 테스트 세트 레이블.
* y\_probas (arr): 테스트 세트 예측 확률.
* labels (list): 대상 변수(y)의 명명된 레이블.

#### 특징 중요도

![](/images/integrations/scikit_feature_importances.png)

분류 작업에 대한 각 특징의 중요도를 평가하고 그립니다. `feature_importances_` 속성이 있는 분류기(예: 트리)에만 작동합니다.

`wandb.sklearn.plot_feature_importances(model, ['width', 'height, 'length'])`

* model (clf): 학습된 분류기를 입력합니다.
* feature\_names (list): 특징 이름. 특징 인덱스를 해당 이름으로 대체하여 플롯을 읽기 쉽게 합니다.

#### 보정 곡선

![](/images/integrations/scikit_calibration_curve.png)

분류기의 예측 확률이 얼마나 잘 보정되어 있는지와 보정되지 않은 분류기를 어떻게 보정할 수 있는지를 보여주는 그래프를 그립니다. 기준 논리 회귀 모델, 인수로 전달된 모델, 그리고 그것의 등각 보정 및 시그모이드 보정에 의한 예측 확률을 비교합니다.

보정 곡선이 대각선에 가까울수록 더 좋습니다. 전치된 시그모이드 곡선은 과적합된 분류기를 나타내고, 시그모이드 곡선은 과소적합된 분류기를 나타냅니다. 모델의 등각 및 시그모이드 보정을 학습하고 그 곡선을 비교함으로써 모델이 과대적합되었는지 과소적합되었는지를 알아낼 수 있으며, 그렇다면 어떤 보정(시그모이드 또는 등각)이 이를 수정하는 데 도움이 될지 파악할 수 있습니다.

자세한 내용은 [sklearn 문서](https://scikit-learn.org/stable/auto\_examples/calibration/plot\_calibration\_curve.html)를 확인하세요.

`wandb.sklearn.plot_calibration_curve(clf, X, y, 'RandomForestClassifier')`

* model (clf): 학습된 분류기를 입력합니다.
* X (arr): 학습 세트 특징.
* y (arr): 학습 세트 레이블.
* model\_name (str): 모델 이름. 기본값은 'Classifier'

#### 혼동 행렬

![](/images/integrations/scikit_confusion_matrix.png)

분류의 정확성을 평가하기 위해 혼동 행렬을 계산합니다. 모델 예측의 품질을 평가하고 모델이 잘못된 예측을 하는 패턴을 찾는 데 유용합니다. 대각선은 모델이 맞춘 예측, 즉 실제 레이블이 예측 레이블과 같은 경우를 나타냅니다.

`wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)`

* y\_true (arr): 테스트 세트 레이블.
* y\_pred (arr): 테스트 세트 예측 레이블.
* labels (list): 대상 변수(y)의 명명된 레이블.

#### 요약 메트릭

![](/images/integrations/scikit_summary_metrics.png)

회귀 및 분류 알고리즘에 대해 요약 메트릭(f1, 정확도, 정밀도, 재현율 등 분류 및 mse, mae, r2 점수 등 회귀)을 계산합니다.

`wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_test, y_test)`

* model (clf 또는 reg): 학습된 회귀모델 또는 분류기를 입력합니다.
* X (arr): 학습 세트 특징.
* y (arr): 학습 세트 레이블.
  * X\_test (arr): 테스트 세트 특징.
* y\_test (arr): 테스트 세트 레이블.

#### 엘보우 플롯

![](/images/integrations/scikit_elbow_plot.png)

클러스터 개수의 함수로 설명된 분산의 비율과 학습 시간을 측정하고 그립니다. 최적의 클러스터 개수를 선택하는 데 유용합니다.

`wandb.sklearn.plot_elbow_curve(model, X_train)`

* model (clusterer): 학습된 클러스터를 입력합니다.
* X (arr): 학습 세트 특징.

#### 실루엣 플롯

![](/images/integrations/scikit_silhouette_plot.png)

한 클러스터 내의 각 점이 인접 클러스터의 점들과 얼마나 가까운지 측정하고 그립니다. 클러스터의 두께는 클러스터 크기를 나타냅니다. 수직선은 모든 점의 평균 실루엣 점수를 나타냅니다.

실루엣 계수가 +1에 가까우면 샘플이 인접 클러스터에서 멀리 떨어져 있음을 나타냅니다. 0 값은 샘플이 두 인접 클러스터 사이의 결정 경계에 있거나 매우 가깝다는 것을 나타내며, 음수 값은 해당 샘플이 잘못된 클러스터에 할당되었을 수 있음을 나타냅니다.

일반적으로 우리는 모든 실루엣 클러스터 점수가 평균(빨간 선 너머)보다 높고 가능한 한 1에 가까운 것을 원합니다. 또한 데이터의 기본 패턴을 반영하는 클러스터 크기를 선호합니다.

`wandb.sklearn.plot_silhouette(model, X_train, ['spam', 'not spam'])`

* model (clusterer): 학습된 클러스터를 입력합니다.
* X (arr): 학습 세트 특징.
  * cluster\_labels (list): 클러스터 레이블의 이름. 클러스터 인덱스를 해당 이름으로 대체하여 플롯을 읽기 쉽게 합니다.

#### 이상치 후보 플롯

![](/images/integrations/scikit_outlier_plot.png)

쿡의 거리를 통해 회귀 모델에 대한 데이터 포인트의 영향을 측정합니다. 크게 편향된 영향을 미치는 인스턴스는 이상치일 가능성이 있습니다. 이상치 탐지에 유용합니다.

`wandb.sklearn.plot_outlier_candidates(model, X, y)`

* model (regressor): 학습된 분류기를 입력합니다.
* X (arr): 학습 세트 특징.
* y (arr): 학습 세트 레이블.

#### 잔차 플롯

![](/images/integrations/scikit_residuals_plot.png)

예측된 타겟 값(y축)과 실제와 예측된 타겟 값의 차이(x축)를 측정하고 그립니다. 또한 잔차 오류의 분포도 그립니다.

일반적으로 잘 맞는 모델의 잔차는 무작위로 분포되어야 합니다. 왜냐하면 좋은 모델은 데이터 세트의 대부분의 현상을 설명할 수 있으며, 무작위 오류를 제외하고는 말입니다.

`wandb.sklearn.plot_residuals(model, X, y)`

* model (regressor): 학습된 분류기를 입력합니다.
* X (arr): 학습 세