---
title: LightGBM
description: W&B로 나무를 추적하세요.
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple_LightGBM_Integration.ipynb'/>

`wandb` 라이브러리는 [LightGBM](https://lightgbm.readthedocs.io/en/latest/)에 대한 특별한 콜백을 포함하고 있습니다. 또한 Weights & Biases의 일반적인 로그 기능을 사용하여 하이퍼파라미터 탐색과 같은 대규모 실험을 추적하는 것도 쉽습니다.

```python
from wandb.integration.lightgbm import wandb_callback, log_summary
import lightgbm as lgb

# 메트릭을 W&B에 로그
gbm = lgb.train(..., callbacks=[wandb_callback()])

# 특성 중요도 플롯을 로그하고 모델 체크포인트를 W&B에 업로드
log_summary(gbm, save_model_checkpoint=True)
```

:::info
작동하는 코드 예제를 찾고 계신가요? [GitHub의 예제 저장소](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)를 확인하세요.
:::

## Sweeps를 사용하여 하이퍼파라미터 튜닝하기

모델에서 최대 성능을 얻기 위해서는 트리 깊이 및 학습 속도와 같은 하이퍼파라미터를 튜닝해야 합니다. Weights & Biases는 대규모 하이퍼파라미터 테스트 실험을 구성, 조정 및 분석하기 위한 강력한 툴킷인 [Sweeps](../sweeps/)를 포함하고 있습니다.

:::info
이 툴에 대해 더 알아보고 XGBoost와 함께 Sweeps를 사용하는 방법의 예제를 보려면 이 인터랙티브 Colab 노트북을 확인하세요.

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb'/>
:::

![요약: 이 분류 데이터셋에서 트리가 선형 학습자를 능가합니다.](/images/integrations/lightgbm_sweeps.png)