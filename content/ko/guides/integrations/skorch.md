---
title: Skorch
description: Skorch와 W&B를 통합하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-skorch
    parent: integrations
weight: 400
---

Skorch와 함께 Weights & Biases를 사용하여 모든 모델 성능 메트릭, 모델 토폴로지 및 컴퓨팅 리소스와 함께 최고의 성능을 보이는 모델을 각 에포크 후에 자동으로 기록할 수 있습니다. `wandb_run.dir`에 저장된 모든 파일은 자동으로 W&B 서버에 기록됩니다.

[실행 예시](https://app.wandb.ai/borisd13/skorch/runs/s20or4ct?workspace=user-borisd13)를 참조하세요.

## 파라미터

| 파라미터 | 타입 | 설명 |
| :--- | :--- | :--- |
| `wandb_run` | `wandb.wandb_run`. Run | 데이터를 기록하는 데 사용되는 wandb run. |
| `save_model` | bool (default=True) | 최적 모델의 체크포인트를 저장하고 W&B 서버의 Run에 업로드할지 여부입니다. |
| `keys_ignored` | str 또는 str의 리스트 (default=None) | tensorboard에 기록되지 않아야 하는 키 또는 키 리스트입니다. 사용자가 제공한 키 외에도 `event_`로 시작하거나 `_best`로 끝나는 키는 기본적으로 무시됩니다. |

## 코드 예시

통합이 어떻게 작동하는지 보여주는 몇 가지 예제를 만들었습니다.

* [Colab](https://colab.research.google.com/drive/1Bo8SqN1wNPMKv5Bn9NjwGecBxzFlaNZn?usp=sharing): 통합을 시도하는 간단한 데모
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

# 하이퍼 파라미터 기록 (선택 사항)
wandb_run.config.update({"learning rate": 1e-3, "batch size": 32})

net = NeuralNet(..., callbacks=[WandbLogger(wandb_run)])
net.fit(X, y)
```

## 메소드 참조

| 메소드 | 설명 |
| :--- | :--- |
| `initialize`\(\) | 콜백의 초기 상태를 (재)설정합니다. |
| `on_batch_begin`\(net\[, X, y, training\]\) | 각 배치 시작 시 호출됩니다. |
| `on_batch_end`\(net\[, X, y, training\]\) | 각 배치 종료 시 호출됩니다. |
| `on_epoch_begin`\(net\[, dataset_train, …\]\) | 각 에포크 시작 시 호출됩니다. |
| `on_epoch_end`\(net, \*\*kwargs\) | 마지막 기록 단계의 값을 기록하고 최적 모델을 저장합니다. |
| `on_grad_computed`\(net, named_parameters\[, X, …\]\) | 그래디언트가 계산되었지만 업데이트 단계가 수행되기 전에 배치당 한 번 호출됩니다. |
| `on_train_begin`\(net, \*\*kwargs\) | 모델 토폴로지를 기록하고 그래디언트를 위한 훅을 추가합니다. |
| `on_train_end`\(net\[, X, y\]\) | 트레이닝 종료 시 호출됩니다. |
