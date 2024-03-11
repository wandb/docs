---
description: How to integrate W&B with Skorch.
slug: /guides/integrations/skorch
displayed_sidebar: default
---

# Skorch

Weights & Biases를 Skorch와 함께 사용하면 가장 성능이 좋은 모델을 자동으로 로그하고, 모든 모델 성능 메트릭, 모델 토폴로지 및 각 에포크 후의 컴퓨트 리소스를 로그할 수 있습니다. wandb_run.dir에 저장된 모든 파일은 자동으로 W&B 서버에 로그됩니다.

[예시 run](https://app.wandb.ai/borisd13/skorch/runs/s20or4ct?workspace=user-borisd13)을 참조하세요.

## 파라미터

| 파라미터 | 타입 | 설명 |
| :--- | :--- | :--- |
| `wandb_run` |  wandb.wandb_run.Run | 데이터를 로그하는 데 사용되는 wandb run. |
|`save_model` | bool (기본값=True)| 가장 좋은 모델의 체크포인트를 저장하고 W&B 서버의 Run에 업로드할지 여부.|
|`keys_ignored`| str 또는 str의 리스트 (기본값=None) | tensorboard에 로그되지 않아야 할 키 또는 키 리스트. 사용자가 제공한 키뿐만 아니라 `event_`로 시작하거나 `_best`로 끝나는 키는 기본적으로 무시됩니다.|

## 예시 코드

통합 작동 방식을 확인할 수 있는 몇 가지 예시를 만들었습니다:

* [Colab](https://colab.research.google.com/drive/1Bo8SqN1wNPMKv5Bn9NjwGecBxzFlaNZn?usp=sharing): 통합을 시도해 볼 수 있는 간단한 데모
* [단계별 가이드](https://app.wandb.ai/cayush/uncategorized/reports/Automate-Kaggle-model-training-with-Skorch-and-W%26B--Vmlldzo4NTQ1NQ): Skorch 모델 성능 추적

```python
# wandb 설치
... pip install wandb

import wandb
from skorch.callbacks import WandbLogger

# wandb Run 생성
wandb_run = wandb.init()
# 대안: W&B 계정 없이 wandb Run 생성
wandb_run = wandb.init(anonymous="allow")

# 하이퍼파라미터 로그 (선택사항)
wandb_run.config.update({"learning rate": 1e-3, "batch size": 32})

net = NeuralNet(..., callbacks=[WandbLogger(wandb_run)])
net.fit(X, y)
```

## 메소드

| 메소드 | 설명 |
| :--- | :--- |
| `initialize`\(\) | 콜백의 초기 상태를 (재)설정합니다. |
| `on_batch_begin`\(net\[, X, y, 트레이닝\]\) | 각 배치의 시작에 호출됩니다. |
| `on_batch_end`\(net\[, X, y, 트레이닝\]\) | 각 배치의 끝에 호출됩니다. |
| `on_epoch_begin`\(net\[, dataset\_train, …\]\) | 각 에포크의 시작에 호출됩니다. |
| `on_epoch_end`\(net, \*\*kwargs\) | 마지막 히스토리 단계에서 값 로그 및 최고 모델 저장 |
| `on_grad_computed`\(net, named\_parameters\[, X, …\]\) | 그레이디언트가 계산된 후 한 배치에 한 번씩 업데이트 단계가 수행되기 전에 호출됩니다. |
| `on_train_begin`\(net, \*\*kwargs\) | 모델 토폴로지 로그 및 그레이디언트에 대한 훅 추가 |
| `on_train_end`\(net\[, X, y\]\) | 트레이닝의 끝에 호출됩니다. |