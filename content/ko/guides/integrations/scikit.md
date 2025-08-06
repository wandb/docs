---
title: Scikit-Learn
menu:
  default:
    identifier: ko-guides-integrations-scikit
    parent: integrations
weight: 380
---

wandb 를 사용하면 몇 줄의 코드만으로 scikit-learn 모델의 성능을 시각화하고 비교할 수 있습니다. [예제 살펴보기 →](https://wandb.me/scikit-colab)

## 시작하기

### 회원가입 및 API 키 생성

API 키는 W&B 에서 내 머신을 인증하는 역할을 합니다. API 키는 사용자 프로필에서 생성할 수 있습니다.

{{% alert %}}
더 간편한 방법으로는 [W&B 인증 페이지](https://wandb.ai/authorize)에서 바로 API 키를 생성하실 수 있습니다. 표시되는 API 키를 복사하여 패스워드 관리 프로그램 등 안전한 장소에 저장해 주세요.
{{% /alert %}}

1. 오른쪽 상단의 사용자 프로필 아이콘을 클릭합니다.
1. **User Settings**를 선택하고, **API Keys** 섹션까지 스크롤합니다.
1. **Reveal**을 클릭합니다. 표시된 API 키를 복사합니다. API 키를 숨기려면 페이지를 새로고침하세요.

### `wandb` 라이브러리 설치 및 로그인

`wandb` 라이브러리를 로컬에 설치하고 로그인하려면 아래 안내를 따르세요.

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 본인의 API 키로 설정합니다.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` 라이브러리를 설치하고 로그인합니다.

    ```shell
    pip install wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

### 메트릭 기록하기

```python
import wandb

wandb.init(project="visualize-sklearn") as run:

  y_pred = clf.predict(X_test)
  accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)

  # 메트릭을 시간에 따라 기록하려면 run.log를 사용하세요
  run.log({"accuracy": accuracy})

  # 또는 트레이닝 종료 후 최종 메트릭만 기록하려면 run.summary를 사용하실 수 있습니다
  run.summary["accuracy"] = accuracy
```

### 플롯 생성하기

#### 1단계: wandb 임포트 및 새로운 run 시작

```python
import wandb

run = wandb.init(project="visualize-sklearn")
```

#### 2단계: 플롯 시각화

#### 개별 플롯

모델 트레이닝 및 예측 후에는 wandb 에서 예측값 분석을 위한 다양한 플롯을 생성할 수 있습니다. 지원되는 모든 차트 목록은 **Supported Plots**에서 확인하실 수 있습니다.

```python
# 혼동 행렬 단일 플롯 시각화
wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)
```

#### 전체 플롯

W&B 에는 여러 관련 플롯을 한 번에 그릴 수 있는 `plot_classifier` 등의 함수가 있습니다.

```python
# 분류기 전체 플롯 시각화
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

# 회귀 전체 플롯
wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test, model_name="Ridge")

# 군집 전체 플롯
wandb.sklearn.plot_clusterer(
    kmeans, X_train, cluster_labels, labels=None, model_name="KMeans"
)

run.finish()
```

#### 기존의 Matplotlib 플롯

Matplotlib 으로 그린 플롯도 W&B 대시보드에 업로드할 수 있습니다. 먼저 `plotly`를 설치가 필요합니다.

```bash
pip install plotly
```

그 후 아래와 같이 플롯을 W&B 대시보드에 기록할 수 있습니다.

```python
import matplotlib.pyplot as plt
import wandb

with wandb.init(project="visualize-sklearn") as run:

  # 여기에서 plt.plot(), plt.scatter() 등 다양한 플롯을 그리세요.
  # ...

  # plt.show() 대신 다음과 같이 기록합니다:
  run.log({"plot": plt})
```

## 지원되는 플롯

### 러닝 커브

{{< img src="/images/integrations/scikit_learning_curve.png" alt="Scikit-learn learning curve" >}}

여러 길이의 데이터셋으로 모델을 트레이닝한 다음, 데이터셋 크기별 교차 검증 점수를 트레이닝/테스트 세트 각각에 대해 플롯으로 보여줍니다.

`wandb.sklearn.plot_learning_curve(model, X, y)`

* model (clf 또는 reg): 학습이 완료된 회귀기 또는 분류기를 입력합니다.
* X (arr): 데이터셋 특징값.
* y (arr): 데이터셋 라벨.

### ROC

{{< img src="/images/integrations/scikit_roc.png" alt="Scikit-learn ROC curve" >}}

ROC 곡선은 참 긍정률(TPR, y축)과 거짓 긍정률(FPR, x축)을 그립니다. 최적의 지점은 왼쪽 위 (TPR=1, FPR=0)이고, 곡선 아래 면적(AUC-ROC)은 높을수록 더 좋은 성능을 의미합니다.

`wandb.sklearn.plot_roc(y_true, y_probas, labels)`

* y_true (arr): 테스트 세트 라벨.
* y_probas (arr): 테스트 세트의 예측 확률값.
* labels (list): 타겟 변수(y)에 대한 이름 목록.

### 클래스 비율

{{< img src="/images/integrations/scikic_class_props.png" alt="Scikit-learn classification properties" >}}

트레이닝 및 테스트 세트의 타겟 클래스 분포를 그립니다. 불균형 데이터 탐지 및 특정 클래스가 모델에 과도한 영향을 미치지 않는지 확인할 때 유용합니다.

`wandb.sklearn.plot_class_proportions(y_train, y_test, ['dog', 'cat', 'owl'])`

* y_train (arr): 트레이닝 세트 라벨.
* y_test (arr): 테스트 세트 라벨.
* labels (list): 타겟 변수(y)에 대한 이름 목록.

### PR 곡선

{{< img src="/images/integrations/scikit_precision_recall.png" alt="Scikit-learn precision-recall curve" >}}

다양한 임계값에서 정밀도와 재현율 간의 트레이드오프를 계산합니다. 곡선 아래 면적이 높을수록 높은 정밀도와 재현율을 모두 가졌음을 의미하며, PR 곡선은 클래스가 매우 불균형할 때 특히 유용합니다.

정확한 예측(높은 정밀도)과 양성 샘플 대부분을 검출(높은 재현율)했는지 평가할 수 있습니다.

`wandb.sklearn.plot_precision_recall(y_true, y_probas, labels)`

* y_true (arr): 테스트 세트 라벨.
* y_probas (arr): 테스트 세트의 예측 확률값.
* labels (list): 타겟 변수(y)에 대한 이름 목록.

### Feature importances

{{< img src="/images/integrations/scikit_feature_importances.png" alt="Scikit-learn feature importance chart" >}}

분류 작업에서 각 특징의 중요도를 평가하고 플롯합니다. 트리와 같이 `feature_importances_` 속성이 있는 분류기에서 지원됩니다.

`wandb.sklearn.plot_feature_importances(model, ['width', 'height, 'length'])`

* model (clf): 학습이 완료된 분류기.
* feature_names (list): 특징 이름. 인덱스 대신 이름으로 표시되어 플롯이 더 읽기 쉽게 됩니다.

### 캘리브레이션 곡선

{{< img src="/images/integrations/scikit_calibration_curve.png" alt="Scikit-learn calibration curve" >}}

분류기의 예측 확률이 얼마나 잘 보정(calibrate)되었는지, 또는 어떻게 보정할 수 있는지를 플롯합니다. 베이스라인 로지스틱 회귀, 인자로 전달된 모델, 이소토닉 및 시그모이드 보정 모델의 캘리브레이션 곡선들을 비교합니다.

캘리브레이션 곡선이 대각선에 가까운 형태일수록 좋습니다. 시그모이드 모양은 과소적합, 그 반대는 과적합을 나타냅니다. 이소토닉/시그모이드 캘리브레이션을 직접 비교하여 최적의 해결책을 탐색할 수 있습니다.

자세한 정보는 [sklearn 문서](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html)를 참고하세요.

`wandb.sklearn.plot_calibration_curve(clf, X, y, 'RandomForestClassifier')`

* model (clf): 학습이 완료된 분류기.
* X (arr): 트레이닝 세트 특징값.
* y (arr): 트레이닝 세트 라벨.
* model_name (str): 모델 이름 (기본값: 'Classifier')

### 혼동 행렬

{{< img src="/images/integrations/scikit_confusion_matrix.png" alt="Scikit-learn confusion matrix" >}}

분류기의 정확도를 평가하기 위해 혼동 행렬을 계산합니다. 올바르게 예측된 결과는 대각선 상에 나타나며, 오답 유형을 쉽게 확인할 수 있습니다.

`wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)`

* y_true (arr): 테스트 세트 라벨.
* y_pred (arr): 테스트 세트 예측 라벨.
* labels (list): 타겟 변수(y)에 대한 이름 목록.

### 요약 메트릭

{{< img src="/images/integrations/scikit_summary_metrics.png" alt="Scikit-learn summary metrics" >}}

- 분류 문제에서는 `mse`, `mae`, `r2` 등 각종 요약 메트릭을 계산합니다.
- 회귀 문제에서는 `f1`, 정확도, 정밀도, 재현율을 계산합니다.

`wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_test, y_test)`

* model (clf 또는 reg): 학습이 완료된 회귀기 또는 분류기.
* X (arr): 트레이닝 세트 특징값.
* y (arr): 트레이닝 세트 라벨.
  * X_test (arr): 테스트 세트 특징값.
* y_test (arr): 테스트 세트 라벨.

### 엘보우 플롯

{{< img src="/images/integrations/scikit_elbow_plot.png" alt="Scikit-learn elbow plot" >}}

클러스터 개수에 따른 분산 설명 비율 및 트레이닝 시간의 변화를 플롯합니다. 최적의 클러스터 개수 결정에 유용합니다.

`wandb.sklearn.plot_elbow_curve(model, X_train)`

* model (clusterer): 학습이 완료된 클러스터러.
* X (arr): 트레이닝 세트 특징값.

### 실루엣 플롯

{{< img src="/images/integrations/scikit_silhouette_plot.png" alt="Scikit-learn silhouette plot" >}}

각 클러스터 내 데이터 포인트가 인접 클러스터와 얼마나 가까운지 측정하여 플롯합니다. 클러스터 두께는 해당 클러스터의 크기를 의미하며, 수직선은 전체 포인트의 평균 실루엣 점수를 나타냅니다.

+1에 가까운 값은 이웃 클러스터와 멀리 떨어져 있음을, 0은 두 클러스터 경계에 있음을, 음수는 잘못된 클러스터에 할당됐을 수 있음을 시사합니다.

실루엣 점수는 전체적으로 평균 이상(빨간 선 이상)이며 1에 가까울수록 좋습니다. 클러스터 크기도 데이터의 실제 패턴을 잘 반영해야 합니다.

`wandb.sklearn.plot_silhouette(model, X_train, ['spam', 'not spam'])`

* model (clusterer): 학습이 완료된 클러스터러.
* X (arr): 트레이닝 세트 특징값.
  * cluster_labels (list): 클러스터 라벨 이름. 인덱스 대신 이름을 사용해 가독성을 높입니다.

### 이상치 후보 플롯

{{< img src="/images/integrations/scikit_outlier_plot.png" alt="Scikit-learn outlier plot" >}}

회귀 분석에서 각 데이터 포인트가 모델에 미치는 영향을 cook's distance로 측정합니다. 영향이 비정상적으로 큰 데이터는 이상치로 의심해 볼 수 있으며, 이상치 탐지에 유용합니다.

`wandb.sklearn.plot_outlier_candidates(model, X, y)`

* model (regressor): 학습이 완료된 회귀기.
* X (arr): 트레이닝 세트 특징값.
* y (arr): 트레이닝 세트 라벨.

### 잔차 플롯

{{< img src="/images/integrations/scikit_residuals_plot.png" alt="Scikit-learn residuals plot" >}}

예측 타겟 값(y축)과 실제 값과 예측값의 차이(x축), 잔차의 분포를 플롯합니다.

잘 맞는 모델의 경우 잔차는 무작위로 분포해야 하며, 이는 데이터셋 내 대부분의 패턴을 모델이 설명함을 의미합니다.

`wandb.sklearn.plot_residuals(model, X, y)`

* model (regressor): 학습이 완료된 회귀기.
* X (arr): 트레이닝 세트 특징값.
*   y (arr): 트레이닝 세트 라벨.

질문이 있으시면 [slack 커뮤니티](https://wandb.me/slack) 에서 언제든 대답해드리겠습니다.

## 예제

* [colab에서 실행하기](https://wandb.me/scikit-colab): 바로 시작할 수 있는 간단한 노트북 예제입니다.