---
title: Scikit-Learn
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

wandb를 사용하면 몇 줄의 코드만으로 scikit-learn 모델의 성능을 시각화하고 비교할 수 있습니다. [**예제를 시도해보세요 →**](http://wandb.me/scikit-colab)

## :fire: 시작하기

### wandb 가입 및 로그인

a) [**가입하기**](https://wandb.ai/site) 에서 무료 계정을 만드세요

b) `wandb` 라이브러리를 pip로 설치하세요

c) 트레이닝 스크립트에서 로그인을 하려면 www.wandb.ai에서 계정에 로그인되어 있어야 하며, **그러면** [**승인 페이지**](https://wandb.ai/authorize)**에서 API 키를 찾을 수 있습니다.**

Weights and Biases를 처음 사용하는 경우 [퀵스타트](../../quickstart.md)를 확인해 보세요.

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
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

# 메트릭을 시간에 따라 로깅할 경우, wandb.log를 사용하세요
wandb.log({"accuracy": accuracy})

# OR 트레이닝 종료 시 최종 메트릭을 로깅하려면 wandb.summary를 사용하세요
wandb.summary["accuracy"] = accuracy
```

### 플롯 만들기

#### 1단계: wandb를 가져와서 새로운 run을 초기화합니다.

```python
import wandb

wandb.init(project="visualize-sklearn")
```

#### 2단계: 개별 플롯 시각화

모델을 트레이닝하고 예측을 한 후, wandb에서 예측을 분석하기 위한 플롯을 생성할 수 있습니다. 아래 **지원되는 플롯** 섹션에서 지원되는 차트의 전체 목록을 확인하세요.

```python
# 개별 플롯 시각화
wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)
```

#### 또는 모든 플롯을 한꺼번에 시각화

W&B에는 여러 관련 플롯을 그릴 수 있는 `plot_classifier`와 같은 함수가 있습니다.

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

**또는 기존의 matplotlib 플롯 그리기:**

Matplotlib에서 생성된 플롯도 W&B 대시보드에 로그할 수 있습니다. 이를 위해 먼저 `plotly`를 설치해야 합니다.

```
pip install plotly
```

마지막으로, 다음과 같이 W&B의 대시보드에 플롯을 로그할 수 있습니다.

```python
import matplotlib.pyplot as plt
import wandb

wandb.init(project="visualize-sklearn")

# 여기에서 모든 plt.plot(), plt.scatter() 등을 수행하세요.
# ...

# plt.show() 대신에 다음을 수행하세요:
wandb.log({"plot": plt})
```

### 지원되는 플롯

#### 학습 곡선

![](/images/integrations/scikit_learning_curve.png)

데이터셋의 길이를 다르게 하여 모델을 학습시키고, 교차 검증된 점수와 데이터셋 크기의 플롯을 생성합니다. 트레이닝 세트와 테스트 세트 모두에 대한 플롯이 생성됩니다.

`wandb.sklearn.plot_learning_curve(model, X, y)`

* model (clf 또는 reg): 피팅된 회귀모델 또는 분류기를 사용합니다.
* X (arr): 데이터셋 특징.
* y (arr): 데이터셋 라벨.

#### ROC

![](/images/integrations/scikit_roc.png)

ROC 곡선은 진양성율(y축) 대 위양성율(x축)의 플롯입니다. 이상적인 점수는 TPR = 1 및 FPR = 0이며, 이는 왼쪽 상단의 점입니다. 일반적으로 우리는 ROC 곡선 아래의 면적(AUC-ROC)을 계산하며, AUC-ROC가 높을수록 좋습니다.

`wandb.sklearn.plot_roc(y_true, y_probas, labels)`

* y_true (arr): 테스트 세트 라벨.
* y_probas (arr): 테스트 세트 예측 확률.
* labels (list): 목표 변수(y)의 이름 라벨.

#### 클래스 비율

![](/images/integrations/scikic_class_props.png)

트레이닝 세트와 테스트 세트에서 목표 클래스의 분포를 플롯합니다. 불균형 클래스 탐지 및 한 클래스가 모델에 과도한 영향을 미치지 않도록 확인하는 데 유용합니다.

`wandb.sklearn.plot_class_proportions(y_train, y_test, ['dog', 'cat', 'owl'])`

* y_train (arr): 트레이닝 세트 라벨.
* y_test (arr): 테스트 세트 라벨.
* labels (list): 목표 변수(y)의 이름 라벨.

#### Precision-Recall 곡선

![](/images/integrations/scikit_precision_recall.png)

다양한 임계값에 대해 정밀도와 재현율 간의 균형을 계산합니다. 곡선 아래의 면적이 높을수록 정밀도와 재현율이 모두 높음을 나타내며, 높은 정밀도는 낮은 위양성율과 관련이 있고, 높은 재현율은 낮은 위음성율과 관련이 있습니다.

높은 점수는 분류기가 정확한 결과(높은 정밀도)를 반환하는 동시에 대부분의 모든 긍정 결과를 반환(높은 재현율)하고 있음을 보여줍니다. PR 곡선은 클래스가 매우 불균형할 때 유용합니다.

`wandb.sklearn.plot_precision_recall(y_true, y_probas, labels)`

* y_true (arr): 테스트 세트 라벨.
* y_probas (arr): 테스트 세트 예측 확률.
* labels (list): 목표 변수(y)의 이름 라벨.

#### 특징 중요도

![](/images/integrations/scikit_feature_importances.png)

분류 작업을 위한 각 특징의 중요도를 평가하고 플롯합니다. 트리와 같은 `feature_importances_` 속성이 있는 분류기에만 적용됩니다.

`wandb.sklearn.plot_feature_importances(model, ['width', 'height, 'length'])`

* model (clf): 피팅된 분류기를 사용합니다.
* feature_names (list): 특징에 대한 이름. 특징 인덱스를 해당 이름으로 대체하여 플롯을 읽기 쉽게 만듭니다.

#### 보정 곡선

![](/images/integrations/scikit_calibration_curve.png)

분류기의 예측 확률이 얼마나 잘 보정되었는지 및 보정되지 않은 분류기를 어떻게 보정할지를 플롯합니다. 기본 로지스틱 회귀모델, 인수로 전달된 모델, 그리고 그 이소톤 보정 및 시그모이드 보정에 의해 추정된 예측 확률을 비교합니다.

보정 곡선이 대각선에 가까울수록 좋습니다. 뒤집어진 시그모이드 같은 곡선은 과제적합된 분류기를 나타내고, 시그모이드 같은 곡선은 과소적합된 분류기를 나타냅니다. 모델을 이소톤 및 시그모이드 보정을 하여 그 곡선을 비교함으로써 모델이 과제적합 또는 과소적합되었는지, 만약 그렇다면 어느 보정 방식(시그모이드 또는 이소톤)이 문제를 해결할 수 있는지 알아낼 수 있습니다.

자세한 내용은 [sklearn의 문서](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html)를 참조하세요.

`wandb.sklearn.plot_calibration_curve(clf, X, y, 'RandomForestClassifier')`

* model (clf): 피팅된 분류기를 사용합니다.
* X (arr): 트레이닝 세트 특징.
* y (arr): 트레이닝 세트 라벨.
* model_name (str): 모델 이름. 기본값은 'Classifier'입니다.

#### 혼동 행렬

![](/images/integrations/scikit_confusion_matrix.png)

분류의 정확도를 평가하기 위해 혼동 행렬을 계산합니다. 모델 예측의 품질을 평가하고 모델이 실수하는 패턴을 찾는 데 유용합니다. 대각선은 실제 라벨이 예측 라벨과 일치하는 경우, 즉 모델이 맞춘 예측을 나타냅니다.

`wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)`

* y_true (arr): 테스트 세트 라벨.
* y_pred (arr): 테스트 세트 예측 라벨.
* labels (list): 목표 변수(y)의 이름 라벨.

#### 요약 메트릭

![](/images/integrations/scikit_summary_metrics.png)

회귀 및 분류 알고리즘 모두에 대해 f1, 정확도, 정밀도 및 재현율과 같은 요약 메트릭을 계산합니다. (mse, mae, r2 점수 등)

`wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_test, y_test)`

* model (clf 또는 reg): 피팅된 회귀모델 또는 분류기를 사용합니다.
* X (arr): 트레이닝 세트 특징.
* y (arr): 트레이닝 세트 라벨.
* X_test (arr): 테스트 세트 특징.
* y_test (arr): 테스트 세트 라벨.

#### 앨보우 플롯

![](/images/integrations/scikit_elbow_plot.png)

클러스터의 수에 따른 설명 분산의 비율과 트레이닝 시간을 측정하고 플롯합니다. 최적의 클러스터 수를 선택하는 데 유용합니다.

`wandb.sklearn.plot_elbow_curve(model, X_train)`

* model (clusterer): 피팅된 클러스터러를 사용합니다.
* X (arr): 트레이닝 세트 특징.

#### 실루엣 플롯

![](/images/integrations/scikit_silhouette_plot.png)

한 클러스터 내 각 점이 이웃 클러스터의 점에 얼마나 가까이 있는지를 측정하고 플롯합니다. 클러스터의 두께는 클러스터 크기에 해당합니다. 수직선은 모든 점의 평균 실루엣 점수를 나타냅니다.

실루엣 계수가 +1에 가까울수록 샘플이 이웃 클러스터에서 멀리 떨어져 있음을 나타냅니다. 0은 샘플이 두 이웃 클러스터 사이의 결정 경계에 매우 가깝다는 것을 나타내고, 음수 값은 샘플이 잘못된 클러스터에 할당되었을 수 있음을 나타냅니다.

일반적으로 모든 실루엣 클러스터 점수가 평균 이상이어야 하며(빨간색 선을 넘기), 가능한 한 1에 가까워야 합니다. 또한, 클러스터 크기는 데이터의 기본 패턴을 반영해야 합니다.

`wandb.sklearn.plot_silhouette(model, X_train, ['spam', 'not spam'])`

* model (clusterer): 피팅된 클러스터러를 사용합니다.
* X (arr): 트레이닝 세트 특징.
* cluster_labels (list): 클러스터 라벨의 이름. 클러스터 인덱스를 해당 이름으로 대체하여 플롯을 읽기 쉽게 만듭니다.

#### 이상치 후보 플롯

![](/images/integrations/scikit_outlier_plot.png)

쿡의 거리를 통해 회귀모델에 대한 데이터 포인트의 영향을 측정합니다. 심하게 왜곡된 영향을 가진 인스턴스는 잠재적으로 이상치일 수 있습니다. 이상치 탐지에 유용합니다.

`wandb.sklearn.plot_outlier_candidates(model, X, y)`

* model (regressor): 피팅된 회귀모델을 사용합니다.
* X (arr): 트레이닝 세트 특징.
* y (arr): 트레이닝 세트 라벨.

#### 잔차 플롯

![](/images/integrations/scikit_residuals_plot.png)

예측된 목표 값(y축)과 실제 목표 값과 예측된 목표 값의 차이(x축)를 측정하고, 잔차 오차의 분포를 플롯합니다.

일반적으로, 잘 맞춰진 모델의 잔차는 무작위로 분포되어야 하는데, 이는 좋은 모델이 데이터 세트의 대부분의 현상을 설명하며, 임의의 오차만 제외하기 때문입니다.

`wandb.sklearn.plot_residuals(model, X, y)`

* model (regressor): 피팅된 회귀모델을 사용합니다.
* X (arr): 트레이닝 세트 특징.
* y (arr): 트레이닝 세트 라벨.

궁금한 점이 있으면 [slack 커뮤니티](http://wandb.me/slack)에 질문을 남겨주시면 기쁘게 답변하겠습니다.

## 예제

* [Colab에서 실행하기](http://wandb.me/scikit-colab): 시작하기 위한 간단한 노트북
