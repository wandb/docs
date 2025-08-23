---
title: LightGBM
description: W&B로 나무를 추적하세요.
menu:
  default:
    identifier: ko-guides-integrations-lightgbm
    parent: integrations
weight: 190
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple_LightGBM_Integration.ipynb" >}}

`wandb` 라이브러리에는 [LightGBM](https://lightgbm.readthedocs.io/en/latest/) 전용 콜백이 포함되어 있습니다. 또한 W&B의 일반 로그 기능을 사용하여 하이퍼파라미터 스윕과 같은 대규모 실험도 손쉽게 추적할 수 있습니다.

```python
from wandb.integration.lightgbm import wandb_callback, log_summary
import lightgbm as lgb

# 메트릭을 W&B에 로그합니다
gbm = lgb.train(..., callbacks=[wandb_callback()])

# 피처 중요도 그래프를 로그하고, 모델 체크포인트를 W&B에 업로드합니다
log_summary(gbm, save_model_checkpoint=True)
```

{{% alert %}}
작동하는 코드 예제가 필요하신가요? [GitHub의 예제 저장소](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)를 확인해 보세요.
{{% /alert %}}

## Sweeps로 하이퍼파라미터 튜닝하기

모델에서 최고의 성능을 얻으려면 트리 깊이, 학습률 같은 하이퍼파라미터를 튜닝해야 합니다. W&B [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}})는 대규모 하이퍼파라미터 실험을 구성, 관리, 분석할 수 있는 강력한 툴킷입니다.

이 툴들에 대해 더 알아보고, Sweeps를 XGBoost와 함께 사용하는 예제를 보고 싶으시다면 아래 Colab 노트북을 참고해 주세요.

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

{{< img src="/images/integrations/lightgbm_sweeps.png" alt="LightGBM 성능 비교" >}}