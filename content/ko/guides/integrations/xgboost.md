---
title: XGBoost
description: W&B로 나무를 추적하세요.
menu:
  default:
    identifier: ko-guides-integrations-xgboost
    parent: integrations
weight: 460
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W%26B.ipynb" >}}

`wandb` 라이브러리는 XGBoost 트레이닝 중 메트릭, 설정값, 저장된 booster를 자동으로 기록해주는 `WandbCallback` 콜백을 제공합니다. 아래 링크에서 XGBoost의 `WandbCallback`으로 수집한 결과를 실시간으로 볼 수 있는 [W&B Dashboard](https://wandb.ai/morg/credit_scorecard)를 확인해보세요.

{{< img src="/images/integrations/xgb_dashboard.png" alt="XGBoost를 사용하는 W&B Dashboard" >}}

## 시작하기

XGBoost의 메트릭, 설정값, booster 모델을 W&B에 기록하려면, 단순히 XGBoost에 `WandbCallback`을 넘겨주기만 하면 됩니다.

```python
from wandb.integration.xgboost import WandbCallback
import xgboost as XGBClassifier

...
# wandb run 시작
with wandb.init() as run:
  # 모델에 WandbCallback 전달
  bst = XGBClassifier()
  bst.fit(X_train, y_train, callbacks=[WandbCallback(log_model=True)])
```
XGBoost와 W&B를 활용한 로깅 방법이 더 궁금하다면 [이 노트북](https://wandb.me/xgboost)을 확인해보세요.

## `WandbCallback` 참고 문서

### 기능
XGBoost 모델에 `WandbCallback`을 전달하면 다음과 같은 기능이 제공됩니다:
- booster 모델 설정값을 W&B에 로그
- XGBoost에서 수집한 평가 메트릭(rmse, accuracy 등)을 W&B에 로그
- XGBoost에서 수집한 트레이닝 메트릭(만약 eval_set에 데이터를 지정한 경우)을 로그
- 최고 점수와 최고 반복 횟수를 로그
- 트레이닝된 모델을 W&B Artifacts에 저장 및 업로드 (`log_model = True`일 때)
- `log_feature_importance=True`(기본값)일 때, 피처 중요도 bar 플롯을 로그
- `define_metric=True`(기본값)일 때, 최고의 평가 메트릭을 `wandb.Run.summary`에 저장

### 인수
- `log_model`: (boolean) True로 설정 시, 모델을 W&B Artifacts에 저장 및 업로드

- `log_feature_importance`: (boolean) True로 설정 시, 피처 중요도 bar 플롯을 로그

- `importance_type`: (str) 트리 모델에서는 `{weight, gain, cover, total_gain, total_cover}` 중 하나, 선형 모델인 경우 weight 사용

- `define_metric`: (boolean) True(기본값)이면 마지막 step 기준이 아니라 최고의 step에서 모델 성능을 `run.summary`에 저장

[WandbCallback 소스 코드](https://github.com/wandb/wandb/blob/main/wandb/integration/xgboost/xgboost.py)도 직접 확인할 수 있습니다.

더 많은 예제는 [GitHub 예제 저장소](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)에서 찾아볼 수 있습니다.

## Sweeps로 하이퍼파라미터 튜닝하기

모델 성능을 극대화하려면 트리 깊이, 러닝레이트와 같은 하이퍼파라미터 튜닝이 매우 중요합니다. W&B [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}})는 대규모 하이퍼파라미터 실험의 구성, 관리, 분석을 돕는 강력한 툴킷입니다.

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

[XGBoost & Sweeps 파이썬 스크립트](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py)로 직접 실습해볼 수도 있습니다.

{{< img src="/images/integrations/xgboost_sweeps_example.png" alt="XGBoost 성능 비교" >}}