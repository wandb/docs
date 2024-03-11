---
description: Track your trees with W&B.
displayed_sidebar: default
---

# LightGBM

`wandb` 라이브러리에는 [LightGBM](https://lightgbm.readthedocs.io/en/latest/)을 위한 특별한 콜백이 포함되어 있습니다. 또한, 하이퍼파라미터 탐색과 같은 큰 실험을 추적하기 위해 Weights & Biases의 일반 로깅 기능을 사용하기도 쉽습니다.

```python
from wandb.lightgbm import wandb_callback, log_summary
import lightgbm as lgb

# W&B에 메트릭 로그
gbm = lgb.train(..., callbacks=[wandb_callback()])

# W&B에 특성 중요도 그래프 로그 및 모델 체크포인트 업로드
log_summary(gbm, save_model_checkpoint=True)
```

:::info
작동하는 코드 예제를 찾고 계신가요? [GitHub의 예제 저장소](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)를 확인하거나 [Colab 노트북](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple\_LightGBM\_Integration.ipynb)을 사용해보세요.
:::

## Sweeps를 사용하여 하이퍼파라미터 조정하기

모델의 최대 성능을 달성하기 위해서는, 트리 깊이와 학습률과 같은 하이퍼파라미터를 조정해야 합니다. Weights & Biases는 하이퍼파라미터 테스트 실험을 구성, 조정 및 분석하기 위한 강력한 툴킷인 [Sweeps](../sweeps/)를 포함하고 있습니다.

:::info
이 툴에 대해 더 알아보고 XGBoost와 함께 Sweeps를 사용하는 방법의 예제를 보려면 [이 인터랙티브 Colab 노트북을 확인하세요.](http://wandb.me/xgb-sweeps-colab)
:::

![tl;dr: 이 분류 데이터셋에서는 트리가 선형 학습자보다 우수한 성능을 보입니다.](/images/integrations/lightgbm_sweeps.png)