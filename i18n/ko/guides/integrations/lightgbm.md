---
description: Track your trees with W&B.
displayed_sidebar: default
---

# LightGBM

`wandb` 라이브러리에는 [LightGBM](https://lightgbm.readthedocs.io/en/latest/)을 위한 특별한 콜백이 포함되어 있습니다. 또한 Weights & Biases의 일반적인 로깅 기능을 사용하여 하이퍼파라미터 스윕과 같은 대규모 실험을 추적하기도 쉽습니다.

```python
from wandb.lightgbm import wandb_callback, log_summary
import lightgbm as lgb

# W&B에 메트릭 로깅
gbm = lgb.train(..., callbacks=[wandb_callback()])

# 특성 중요도 그래프를 로그하고 모델 체크포인트를 W&B에 업로드
log_summary(gbm, save_model_checkpoint=True)
```

:::info
작동하는 코드 예시를 찾고 있나요? [GitHub에서 예시 모음을 확인해 보세요](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms) 또는 [Colab 노트북을 시도해보세요](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple\_LightGBM\_Integration.ipynb).
:::

## 하이퍼파라미터 스윕을 사용한 튜닝

모델에서 최대 성능을 얻으려면 트리 깊이와 학습률과 같은 하이퍼파라미터를 조정해야 합니다. Weights & Biases는 대규모 하이퍼파라미터 테스팅 실험을 구성, 조정 및 분석하는 강력한 툴킷인 [Sweeps](../sweeps/)를 포함하고 있습니다.

:::info
이 도구에 대해 더 알아보고 XGBoost와 Sweeps를 사용하는 예를 보려면 [이 대화형 Colab 노트북을 확인하세요.](http://wandb.me/xgb-sweeps-colab)
:::

![tl;dr: 이 분류 데이터세트에서는 트리가 선형 학습기보다 성능이 더 우수합니다.](/images/integrations/lightgbm_sweeps.png)