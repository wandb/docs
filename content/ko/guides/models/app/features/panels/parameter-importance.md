---
title: 파라미터 중요도
description: 모델의 하이퍼파라미터와 출력 메트릭 간의 관계를 시각화하세요
menu:
  default:
    identifier: ko-guides-models-app-features-panels-parameter-importance
    parent: panels
weight: 60
---

여러분의 하이퍼파라미터 중 어떤 것이 메트릭의 바람직한 값과 가장 잘 예측되며, 높은 상관관계를 가지는지 확인해보세요.

{{< img src="/images/general/parameter-importance-1.png" alt="Parameter importance panel" >}}

**상관관계(Correlation)** 는 하이퍼파라미터와 선택한 메트릭(여기서는 val_loss) 간의 선형 상관관계를 의미합니다. 상관계수가 높다는 것은, 하이퍼파라미터의 값이 높을 때 메트릭도 함께 높아지는 경향이 있고, 반대의 경우도 마찬가지임을 뜻합니다. 상관관계는 살펴볼 만한 훌륭한 메트릭이지만, 입력값들 간의 2차 상호작용(비선형 관계)은 잡아내지 못하고, 값의 범위가 크게 다르면 비교가 어려워질 수 있습니다.

그래서 W&B에서는 **중요도(importance)** 메트릭도 함께 제공합니다. W&B는 하이퍼파라미터들을 입력으로, 메트릭을 타겟 출력으로 사용하여 랜덤 포레스트를 학습시킨 뒤 해당 모델의 특성 중요도(feature importance) 값을 리포트합니다.

이 기법의 아이디어는 [Jeremy Howard](https://twitter.com/jeremyphoward)와의 대화에서 영감을 얻은 것으로, 그는 [Fast.ai](https://fast.ai)에서 랜덤 포레스트의 특성 중요도를 하이퍼파라미터 탐색에 활용하는 방법을 개척했습니다. W&B에서도 이 분석의 동기를 더 깊게 이해하고 싶다면 [이 강의](https://course18.fast.ai/lessonsml1/lesson4.html)와 [이 노트](https://forums.fast.ai/t/wiki-lesson-thread-lesson-4/7540)를 추천합니다.

하이퍼파라미터 중요도 패널은 상관관계가 높은 여러 하이퍼파라미터 간 복잡한 상호작용을 효과적으로 해석하여 보여줍니다. 이를 통해 어떤 하이퍼파라미터가 모델 성능에 가장 큰 영향을 미치는지 확인, 더욱 효과적으로 하이퍼파라미터 탐색을 튜닝할 수 있습니다.

## 하이퍼파라미터 중요도 패널 만들기

1. W&B 프로젝트로 이동합니다.
2. **Add panels** 버튼을 선택합니다.
3. **CHARTS** 드롭다운을 펼치고, **Parallel coordinates**를 선택합니다.

{{% alert %}}
빈 패널이 나타난다면, run이 그룹화되어 있지 않은지 확인하세요.
{{% /alert %}}

{{< img src="/images/app_ui/hyperparameter_importance_panel.gif" alt="Automatic parameter visualization" >}}

파라미터 매니저를 사용하면 보이거나 숨겨진 파라미터를 직접 설정할 수 있습니다.

{{< img src="/images/app_ui/hyperparameter_importance_panel_manual.gif" alt="Manually setting the visible and hidden fields" >}}

## 하이퍼파라미터 중요도 패널 해석하기

{{< img src="/images/general/parameter-importance-4.png" alt="Feature importance analysis" >}}

이 패널에서는 트레이닝 스크립트 상의 [wandb.Run.config]({{< relref path="/guides/models/track/config/" lang="ko" >}}) 오브젝트에 전달된 모든 파라미터가 표시됩니다. 그리고 선택한 모델 메트릭(여기서는 `val_loss`)에 대해 해당 config 파라미터들의 특성 중요도 및 상관관계가 보여집니다.

### 중요도(Importance)

중요도 칼럼은 각 하이퍼파라미터가 선택한 메트릭을 예측하는 데 얼마나 도움이 되었는지를 나타냅니다. 예를 들어, 여러 하이퍼파라미터를 조정해 보면서 이 그래프를 사용해 추가 조사가 필요한 파라미터를 추려낼 수 있습니다. 이후 스윕에서는 가장 중요한 하이퍼파라미터만을 대상으로 탐색할 수 있으므로, 더 나은 모델을 더 빠르고 저렴하게 찾을 수 있습니다.

{{% alert %}}
W&B는 중요도 계산에서 선형 모델이 아닌 트리 기반 모델을 사용합니다. 트리 기반 모델은 범주형 데이터나 정규화되지 않은 데이터에서도 더 높은 내성을 보입니다.
{{% /alert %}}

위 이미지에서 `epochs`, `learning_rate`, `batch_size`, `weight_decay`가 중요한 파라미터임을 알 수 있습니다.

### 상관관계(Correlations)

상관관계는 개별 하이퍼파라미터와 메트릭 값 사이의 선형적 관계를 보여줍니다. 예를 들어, SGD 옵티마이저를 사용할 때 `val_loss`와 유의미한 관계가 있는지(여기서는 "있다" 입니다)를 확인할 수 있습니다. 상관계수는 -1~1 사이의 값을 가지며, 양의 값은 양의 상관관계, 음의 값은 음의 상관관계를 의미하고, 0은 상관관계가 없음을 뜻합니다. 일반적으로 절대값이 0.7 이상이면 강한 상관관계로 간주합니다.

이 그래프를 이용해 메트릭과 상관계수가 큰 값을 가지는 조합(예를 들어, stochastic gradient descent나 adam이 rmsprop, nadam보다 더 높음)에 대해 좀 더 자세히 탐구하거나, 더 많은 에포크로 트레이닝을 시도할 수도 있습니다.

{{% alert %}}
* 상관관계는 '연관성'만 보여줄 뿐, '인과성'을 반드시 의미하지 않습니다.
* 상관관계는 이상치에 민감해서, 작은 하이퍼파라미터 샘플에서 강한 관계가 중간 수준으로 완화될 수 있습니다.
* 상관관계는 하이퍼파라미터와 메트릭 간 선형 관계만을 포착합니다. 관계가 강한 다항식(비선형) 형태라면 상관관계로는 잡히지 않습니다.
{{% /alert %}}

중요도와 상관관계 지표가 차이를 보이는 이유는, 중요도는 하이퍼파라미터들 간의 상호작용까지 고려하지만 상관관계는 각 개별 하이퍼파라미터가 메트릭에 미치는 영향만 분리해서 보기 때문입니다. 그리고 상관관계는 오직 선형적 관계만 포착하는 반면, 중요도는 훨씬 복잡한 관계까지도 잡아낼 수 있습니다.

이처럼 중요도와 상관관계 모두 하이퍼파라미터가 모델 성능에 어떤 영향을 미치는지 이해할 수 있는 강력한 툴입니다.