---
title: XGBoost
description: W&B로 트리들을 추적하세요.
menu:
  default:
    identifier: ko-guides-integrations-xgboost
    parent: integrations
weight: 460
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W%26B.ipynb" >}}

`wandb` 라이브러리에는 XGBoost를 사용한 트레이닝에서 메트릭, 설정 및 저장된 부스터를 로깅하기 위한 `WandbCallback` 콜백이 있습니다. 여기에서 XGBoost `WandbCallback`의 출력이 포함된 **[라이브 Weights & Biases 대시보드](https://wandb.ai/morg/credit_scorecard)**를 볼 수 있습니다.

{{< img src="/images/integrations/xgb_dashboard.png" alt="Weights & Biases dashboard using XGBoost" >}}

## 시작하기

Weights & Biases에 XGBoost 메트릭, 설정 및 부스터 모델을 로깅하는 것은 `WandbCallback`을 XGBoost에 전달하는 것만큼 쉽습니다.

```python
from wandb.integration.xgboost import WandbCallback
import xgboost as XGBClassifier

...
# wandb run 시작
run = wandb.init()

# WandbCallback을 모델에 전달
bst = XGBClassifier()
bst.fit(X_train, y_train, callbacks=[WandbCallback(log_model=True)])

# wandb run 종료
run.finish()
```

XGBoost 및 Weights & Biases를 사용한 로깅에 대한 포괄적인 내용은 **[이 노트북](https://wandb.me/xgboost)**을 열어 확인하십시오.

## `WandbCallback` 참조

### 기능
`WandbCallback`을 XGBoost 모델에 전달하면 다음과 같은 작업이 수행됩니다.
- 부스터 모델 설정을 Weights & Biases에 로깅합니다.
- XGBoost에서 수집한 평가 메트릭(예: rmse, 정확도 등)을 Weights & Biases에 로깅합니다.
- XGBoost에서 수집한 트레이닝 메트릭을 로깅합니다(eval_set에 데이터를 제공하는 경우).
- 최상의 점수와 최적의 반복을 로깅합니다.
- 트레이닝된 모델을 저장하고 Weights & Biases Artifacts에 업로드합니다(`log_model = True`인 경우).
- `log_feature_importance=True` (기본값)일 때 특징 중요도 플롯을 로깅합니다.
- `define_metric=True` (기본값)일 때 `wandb.summary`에서 최상의 평가 메트릭을 캡처합니다.

### 인수
- `log_model`: (boolean) True인 경우 모델을 저장하고 Weights & Biases Artifacts에 업로드합니다.

- `log_feature_importance`: (boolean) True인 경우 특징 중요도 막대 플롯을 로깅합니다.

- `importance_type`: (str) 트리 모델의 경우 `{weight, gain, cover, total_gain, total_cover}` 중 하나입니다. 선형 모델의 경우 weight입니다.

- `define_metric`: (boolean) True (기본값)인 경우 트레이닝의 마지막 단계 대신 최상의 단계에서 모델 성능을 `wandb.summary`에 캡처합니다.

[WandbCallback 소스 코드](https://github.com/wandb/wandb/blob/main/wandb/integration/xgboost/xgboost.py)를 검토할 수 있습니다.

추가 예제는 [GitHub의 예제 저장소](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)를 참조하십시오.

## Sweeps로 하이퍼파라미터 튜닝하기

모델에서 최대 성능을 얻으려면 트리 깊이 및 학습률과 같은 하이퍼파라미터를 튜닝해야 합니다. Weights & Biases에는 대규모 하이퍼파라미터 테스팅 Experiments를 구성, 오케스트레이션 및 분석하기 위한 강력한 툴킷인 [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}})가 포함되어 있습니다.

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

이 [XGBoost & Sweeps Python 스크립트](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py)를 사용해 볼 수도 있습니다.

{{< img src="/images/integrations/xgboost_sweeps_example.png" alt="Summary: trees outperform linear learners on this classification dataset." >}}
