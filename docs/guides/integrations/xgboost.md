---
title: XGBoost
description: W&B로 당신의 나무를 추적하세요.
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W%26B.ipynb"></CTAButtons>

`wandb` 라이브러리는 XGBoost로 트레이닝한 메트릭, 설정 및 저장된 부스터를 로깅하기 위한 `WandbCallback` 콜백을 제공합니다. 여기에서 XGBoost `WandbCallback`의 출력과 함께하는 **[실시간 Weights & Biases 대시보드](https://wandb.ai/morg/credit_scorecard)**를 확인할 수 있습니다.

![XGBoost를 사용한 Weights & Biases 대시보드](/images/integrations/xgb_dashboard.png)

## 시작하기

XGBoost 메트릭, 설정 및 부스터 모델을 Weights & Biases로 로깅하는 것은 XGBoost에 `WandbCallback`을 전달하는 것만큼 간단합니다:

```python
from wandb.integration.xgboost import WandbCallback
import xgboost as XGBClassifier

...
# wandb run 시작
run = wandb.init()

# 모델에 WandbCallback 전달
bst = XGBClassifier()
bst.fit(X_train, y_train, callbacks=[WandbCallback(log_model=True)])

# wandb run 종료
run.finish()
```

**[이 노트북](https://wandb.me/xgboost)**을 열어 XGBoost와 Weights & Biases를 사용한 로깅에 대해 자세히 알아보세요.

## WandbCallback

### 기능
XGBoost 모델에 `WandbCallback`을 전달하면 다음이 가능합니다:
- 부스터 모델 설정을 Weights & Biases에 로그
- XGBoost가 수집한 평가 메트릭(예: rmse, 정확도 등)을 Weights & Biases에 로그
- XGBoost가 수집한 트레이닝 메트릭을 eval_set에 데이터 제공 시 로그
- 최고 점수 및 최고의 반복 로그
- 트레이닝된 모델을 Weights & Biases Artifacts에 저장 및 업로드 (`log_model = True`일 때)
- `log_feature_importance=True` (기본값)일 때 특징 중요도 그래프 로그
- `define_metric=True` (기본값)일 때 최고 평가 메트릭을 `wandb.summary`에 캡처

### 인수
`log_model`: (boolean) True일 때 모델을 Weights & Biases Artifacts에 저장 및 업로드

`log_feature_importance`: (boolean) True일 때 특징 중요도 막대 그래프 로그

`importance_type`: (str) 트리 모델의 경우 `{weight, gain, cover, total_gain, total_cover}` 중 하나. 선형 모델의 경우 weight.

`define_metric`: (boolean) True일 때 (기본값) 트레이닝의 마지막 단계보다 최고의 단계에서 모델 성능을 `wandb.summary`에 캡처.

WandbCallback의 소스 코드는 [여기](https://github.com/wandb/wandb/blob/main/wandb/integration/xgboost/xgboost.py)에서 찾을 수 있습니다.

:::info
더 많은 작동하는 코드 예제를 찾고 계신가요? [GitHub에서 우리의 예제 저장소](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)를 확인해보세요.
:::

## Sweeps로 하이퍼파라미터 튜닝하기

모델에서 최대 성능을 얻으려면 트리 깊이 및 학습 속도와 같은 하이퍼파라미터를 튜닝해야 합니다. Weights & Biases는 대규모 하이퍼파라미터 테스트 실험을 구성, 조정 및 분석하기 위한 강력한 툴킷 [Sweeps](../sweeps/)를 포함합니다.

:::info
이 툴에 대해 자세히 알아보고 XGBoost와 Sweeps를 사용하는 예제를 보기 위해 다음 Colab 노트북을 참조하세요.

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb"></CTAButtons>

또한 이 [XGBoost & Sweeps Python 스크립트](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py)를 시도해 볼 수 있습니다.
:::

![요약: 트리는 이 분류 데이터셋에서 선형 학습자를 능가합니다.](/images/integrations/xgboost_sweeps_example.png)