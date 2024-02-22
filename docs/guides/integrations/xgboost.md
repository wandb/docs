---
description: Track your trees with W&B.
displayed_sidebar: default
---

# XGBoost

[**여기에서 Colab 노트북으로 시도해보세요 →**](https://wandb.me/xgboost)

`wandb` 라이브러리에는 XGBoost로 학습할 때 메트릭, 구성 및 저장된 부스터를 기록하기 위한 `WandbCallback` 콜백이 있습니다. 여기에서 XGBoost `WandbCallback`의 출력물이 담긴 **[실시간 Weights & Biases 대시보드](https://wandb.ai/morg/credit_scorecard)** 를 볼 수 있습니다.

![Weights & Biases 대시보드에서 XGBoost 사용](/images/integrations/xgb_dashboard.png)

## 시작하기

XGBoost 메트릭, 구성 및 부스터 모델을 Weights & Biases에 기록하는 것은 XGBoost에 `WandbCallback`을 전달하는 것만큼 쉽습니다:

```python
from wandb.xgboost import WandbCallback
import xgboost as XGBClassifier

...
# wandb 실행을 시작합니다
run = wandb.init()

# 모델에 WandbCallback을 전달합니다
bst = XGBClassifier()
bst.fit(X_train, y_train, callbacks=[WandbCallback(log_model=True)])

# wandb 실행을 종료합니다
run.finish()
```

XGBoost와 Weights & Biases를 사용한 기록에 대한 종합적인 내용은 **[이 노트북](https://wandb.me/xgboost)** 을 확인하세요

## WandbCallback

### 기능
XGBoost 모델에 `WandbCallback`을 전달하면 다음을 수행합니다:
- 부스터 모델 구성을 Weights & Biases에 기록합니다
- XGBoost가 수집한 평가 메트릭(예: rmse, 정확도 등)을 Weights & Biases에 기록합니다
- XGBoost가 수집한 학습 메트릭을 기록합니다(만약 eval_set에 데이터를 제공한 경우)
- 최고 점수와 최고 반복을 기록합니다
- 훈련된 모델을 Weights & Biases 아티팩트에 저장하고 업로드합니다(`log_model = True`일 때)
- `log_feature_importance=True`(기본값일 때)일 경우 피처 중요도 플롯을 기록합니다
- `define_metric=True`(기본값일 때)일 경우 `wandb.summary`에 최고의 평가 메트릭을 캡처합니다.

### 인수
`log_model`: (boolean) True인 경우 모델을 Weights & Biases 아티팩트에 저장하고 업로드합니다

`log_feature_importance`: (boolean) True인 경우 피처 중요도 막대 그래프를 기록합니다

`importance_type`: (str) 트리 모델의 경우 {weight, gain, cover, total_gain, total_cover} 중 하나입니다. 선형 모델의 경우 weight입니다.

`define_metric`: (boolean) True(기본값)인 경우 `wandb.summary`에 학습의 마지막 단계가 아닌 최고 성능 단계의 모델 성능을 캡처합니다.


WandbCallback의 소스 코드는 [여기](https://github.com/wandb/wandb/blob/main/wandb/integration/xgboost/xgboost.py)에서 찾을 수 있습니다.

:::안내
더 많은 작동 코드 예제를 찾고 계십니까? [GitHub의 예제 저장소](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)를 확인하거나 [Colab 노트북](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit\_Scorecards\_with\_XGBoost\_and\_W%26B.ipynb)을 시도해보세요.
:::

## 하이퍼파라미터 조정하기

모델의 최대 성능을 얻기 위해서는 트리 깊이와 학습률과 같은 하이퍼파라미터를 조정해야 합니다. Weights & Biases는 하이퍼파라미터 테스팅 실험을 구성, 조정 및 분석하기 위한 강력한 도구인 [Sweeps](../sweeps/)를 포함하고 있습니다.

:::안내
이 도구에 대해 더 알아보고 XGBoost와 Sweeps를 사용하는 방법의 예를 보려면 [이 상호작용적인 Colab 노트북](http://wandb.me/xgb-sweeps-colab)을 확인하거나 이 XGBoost & Sweeps [파이썬 스크립트를 여기에서 시도해보세요](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost\_tune.py)
:::

![tl;dr: 이 분류 데이터세트에서는 트리가 선형 학습기보다 더 우수한 성능을 보입니다.](/images/integrations/xgboost_sweeps_example.png)