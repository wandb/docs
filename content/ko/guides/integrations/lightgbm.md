---
title: LightGBM
description: W&B로 트리들을 추적하세요.
menu:
  default:
    identifier: ko-guides-integrations-lightgbm
    parent: integrations
weight: 190
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple_LightGBM_Integration.ipynb" >}}

`wandb` 라이브러리에는 [LightGBM](https://lightgbm.readthedocs.io/en/latest/)을 위한 특별한 콜백이 포함되어 있습니다. Weights & Biases의 일반적인 로깅 기능을 사용하여 하이퍼파라미터 스윕과 같은 대규모 Experiments를 추적하는 것도 간편합니다.

```python
from wandb.integration.lightgbm import wandb_callback, log_summary
import lightgbm as lgb

# Log metrics to W&B
gbm = lgb.train(..., callbacks=[wandb_callback()])

# Log feature importance plot and upload model checkpoint to W&B
log_summary(gbm, save_model_checkpoint=True)
```

{{% alert %}}
작동하는 코드 예제를 찾고 계십니까? [GitHub의 예제 저장소](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)를 확인하세요.
{{% /alert %}}

## Sweeps를 사용하여 하이퍼파라미터 튜닝하기

모델에서 최대 성능을 얻으려면 트리 깊이 및 학습률과 같은 하이퍼파라미터를 튜닝해야 합니다. Weights & Biases에는 대규모 하이퍼파라미터 테스트 Experiments를 구성, 조정 및 분석하기 위한 강력한 툴킷인 [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}})가 포함되어 있습니다.

이러한 툴에 대해 자세히 알아보고 XGBoost와 함께 Sweeps를 사용하는 방법에 대한 예제를 보려면 이 대화형 Colab 노트북을 확인하세요.

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb" >}}

{{< img src="/images/integrations/lightgbm_sweeps.png" alt="Summary: trees outperform linear learners on this classification dataset." >}}
