---
description: Track your trees with W&B.
displayed_sidebar: default
---

# XGBoost

[**여기에서 Colab 노트북으로 시도해보세요 →**](https://wandb.me/xgboost)

`wandb` 라이브러리는 XGBoost로 트레이닝할 때 메트릭, 설정 및 저장된 부스터를 로깅하기 위한 `WandbCallback` 콜백을 가지고 있습니다. 여기에서 XGBoost `WandbCallback`의 출력물을 포함한 **[실시간 Weights & Biases 대시보드](https://wandb.ai/morg/credit_scorecard)**를 볼 수 있습니다.

![XGBoost를 사용한 Weights & Biases 대시보드](/images/integrations/xgb_dashboard.png)

## 시작하기

XGBoost 메트릭, 설정 및 부스터 모델을 Weights & Biases에 로깅하는 것은 XGBoost에 `WandbCallback`을 전달하는 것만큼 쉽습니다:

```python
from wandb.xgboost import WandbCallback
import xgboost as XGBClassifier

...
# wandb run을 시작합니다
run = wandb.init()

# 모델에 WandbCallback을 전달합니다
bst = XGBClassifier()
bst.fit(X_train, y_train, callbacks=[WandbCallback(log_model=True)])

# wandb run을 종료합니다
run.finish()
```

XGBoost와 Weights & Biases로 로깅하는 것에 대한 자세한 내용은 **[이 노트북](https://wandb.me/xgboost)**을 열어보세요

## WandbCallback

### 기능
XGBoost 모델에 `WandbCallback`을 전달하면 다음을 수행합니다:
- 부스터 모델 설정을 Weights & Biases에 로그합니다
- XGBoost에 의해 수집된 평가 메트릭(예: rmse, 정확도 등)을 Weights & Biases에 로그합니다
- XGBoost에 의해 수집된 트레이닝 메트릭을 로그합니다(`eval_set`에 데이터를 제공한 경우)
- 최고 점수와 최고 반복을 로그합니다
- `log_model = True`일 때 훈련된 모델을 저장하고 Weights & Biases Artifacts에 업로드합니다
- `log_feature_importance=True`일 때(기본값) 피처 중요도 플롯을 로그합니다.
- `define_metric=True`일 때(기본값) 트레이닝의 마지막 단계가 아니라 최고 단계에서의 모델 성능을 `wandb.summary`에 캡처합니다.

### 인수
`log_model`: (boolean) True일 경우 모델을 저장하고 Weights & Biases Artifacts에 업로드합니다

`log_feature_importance`: (boolean) True일 경우 피처 중요도 바 플롯을 로그합니다

`importance_type`: (str) 트리 모델의 경우 {weight, gain, cover, total_gain, total_cover} 중 하나. 선형 모델의 경우 weight.

`define_metric`: (boolean) True일 경우(기본값) `wandb.summary`에서 트레이닝의 마지막 단계가 아니라 최고 단계에서의 모델 성능을 캡처합니다.


WandbCallback의 소스 코드는 [여기](https://github.com/wandb/wandb/blob/main/wandb/integration/xgboost/xgboost.py)에서 찾을 수 있습니다

:::안내
더 많은 작동 코드 예제를 찾고 계신가요? [GitHub의 예제 저장소](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)를 확인하거나 [Colab 노트북](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit\_Scorecards\_with\_XGBoost\_and\_W%26B.ipynb)을 시도해 보세요.
:::

## 하이퍼파라미터를 Sweeps로 튜닝하기

모델의 최대 성능을 달성하려면, 트리 깊이와 학습 속도와 같은 하이퍼파라미터를 튜닝해야 합니다. Weights & Biases에는 하이퍼파라미터 테스트 실험을 구성, 조율 및 분석하기 위한 강력한 툴킷인 [Sweeps](../sweeps/)가 포함되어 있습니다.

:::안내
이 툴에 대해 자세히 알아보고 XGBoost와 Sweeps를 사용하는 방법의 예를 보려면 [이 인터랙티브 Colab 노트북](http://wandb.me/xgb-sweeps-colab)을 확인하거나 이 XGBoost & Sweeps [파이썬 스크립트](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost\_tune.py)를 시도해 보세요.
:::

![tl;dr: 이 분류 데이터셋에서는 트리가 선형 학습자보다 우수합니다.](/images/integrations/xgboost_sweeps_example.png)