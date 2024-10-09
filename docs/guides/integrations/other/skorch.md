---
title: Skorch
description: W&B를 Skorch와 통합하는 방법.
slug: /guides/integrations/skorch
displayed_sidebar: default
---

Weights & Biases와 Skorch를 함께 사용하여 가장 높은 성능을 보이는 모델과 모든 모델 성능 메트릭, 모델 토폴로지 및 각 에포크 이후의 컴퓨팅 리소스를 자동으로 로그할 수 있습니다. wandb_run.dir에 저장된 모든 파일은 자동으로 W&B 서버에 로그됩니다.

[example run](https://app.wandb.ai/borisd13/skorch/runs/s20or4ct?workspace=user-borisd13)를 참조하세요.

## Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `wandb_run` |  wandb.wandb_run.Run | 데이터를 로그하기 위해 사용되는 wandb run. |
|`save_model` | bool (default=True)| 최적의 모델 체크포인트를 저장하여 W&B 서버의 Run에 업로드할지 여부.|
|`keys_ignored`| str 또는 str의 리스트 (기본값=None) | tensorboard에 로그되지 않아야 하는 키 또는 키의 리스트. 사용자에 의해 제공된 키 외에도 `event_`로 시작하거나 `_best`로 끝나는 키는 기본적으로 무시됩니다.|

## Example Code

우리는 인테그레이션이 어떻게 작동하는지 보기 위한 몇 가지 예제를 만들었습니다:

* [Colab](https://colab.research.google.com/drive/1Bo8SqN1wNPMKv5Bn9NjwGecBxzFlaNZn?usp=sharing): 인테그레이션을 시도해 보는 간단한 데모
* [A step by step guide](https://app.wandb.ai/cayush/uncategorized/reports/Automate-Kaggle-model-training-with-Skorch-and-W%26B--Vmlldzo4NTQ1NQ): Skorch 모델 성능 추적을 위한 가이드

```python
# wandb 설치
... pip install wandb

import wandb
from skorch.callbacks import WandbLogger

# wandb Run 생성
wandb_run = wandb.init()
# 대안: W&B 계정 없이 wandb Run 생성
wandb_run = wandb.init(anonymous="allow")

# 하이퍼 파라미터 로그 (선택 사항)
wandb_run.config.update({"learning rate": 1e-3, "batch size": 32})

net = NeuralNet(..., callbacks=[WandbLogger(wandb_run)])
net.fit(X, y)
```

## Methods

| Method | Description |
| :--- | :--- |
| `initialize`\(\) | 콜백의 초기 상태를 (재)설정합니다. |
| `on_batch_begin`\(net\[, X, y, training\]\) | 각 배치의 시작 시 호출됩니다. |
| `on_batch_end`\(net\[, X, y, training\]\) | 각 배치의 끝에서 호출됩니다. |
| `on_epoch_begin`\(net\[, dataset_train, …\]\) | 각 에포크의 시작 시 호출됩니다. |
| `on_epoch_end`\(net, \*\*kwargs\) | 마지막 기록 단계의 값을 로그하고 최상의 모델을 저장합니다. |
| `on_grad_computed`\(net, named_parameters\[, X, …\]\) | 그레이디언트가 계산된 후 업데이트 단계가 수행되기 전에 각 배치당 한 번 호출됩니다. |
| `on_train_begin`\(net, \*\*kwargs\) | 모델 토폴로지를 로그하고 그레이디언트를 위한 훅을 추가합니다. |
| `on_train_end`\(net\[, X, y\]\) | 트레이닝이 끝날 때 호출됩니다. |