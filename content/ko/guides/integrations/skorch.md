---
title: Skorch
description: W&B를 Skorch와 통합하는 방법
menu:
  default:
    identifier: ko-guides-integrations-skorch
    parent: integrations
weight: 400
---

W&B를 Skorch와 함께 사용하면, 가장 성능이 좋은 모델, 모든 모델 성능 메트릭, 모델 토폴로지, 그리고 에포크마다의 컴퓨팅 리소스를 자동으로 로그할 수 있습니다. `wandb_run.dir`에 저장된 모든 파일도 자동으로 W&B에 기록됩니다.

[예시 Run 보기](https://app.wandb.ai/borisd13/skorch/runs/s20or4ct?workspace=user-borisd13).

## 파라미터

| 파라미터 | 타입 | 설명 |
| :--- | :--- | :--- |
| `wandb_run` | `wandb.wandb_run`. Run | 데이터를 로그하는 데 사용되는 wandb run입니다. |
| `save_model` | bool (기본값=True) | 최고의 모델 체크포인트를 저장하고 W&B의 Run에 업로드할지 여부입니다. |
| `keys_ignored` | str 또는 str의 리스트 (기본값=None) | tensorboard에 로그하지 않을 키 또는 키의 리스트입니다. 사용자가 제공한 키 외에도, `event_`로 시작하거나 `_best`로 끝나는 키는 기본적으로 무시됩니다. |

## 예제 코드

인테그레이션이 어떻게 동작하는지 쉽게 확인하실 수 있도록 몇 가지 예제를 준비했습니다:

* [Colab](https://colab.research.google.com/drive/1Bo8SqN1wNPMKv5Bn9NjwGecBxzFlaNZn?usp=sharing): 인테그레이션을 손쉽게 체험할 수 있는 데모
* [단계별 가이드](https://app.wandb.ai/cayush/uncategorized/reports/Automate-Kaggle-model-training-with-Skorch-and-W%26B--Vmlldzo4NTQ1NQ): Skorch 모델 성능 추적하기

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

## 메소드 참조

| 메소드 | 설명 |
| :--- | :--- |
| `initialize`() | 콜백의 초기 상태를 (재)설정합니다. |
| `on_batch_begin`(net[, X, y, training]) | 각 배치가 시작될 때 호출됩니다. |
| `on_batch_end`(net[, X, y, training]) | 각 배치가 끝날 때 호출됩니다. |
| `on_epoch_begin`(net[, dataset_train, …]) | 각 에포크가 시작될 때 호출됩니다. |
| `on_epoch_end`(net, **kwargs) | 마지막 history 단계의 값 로그 및 최고의 모델 저장 |
| `on_grad_computed`(net, named_parameters[, X, …]) | 그레이디언트가 계산된 후, 업데이트 전에 각 배치마다 한 번 호출됩니다. |
| `on_train_begin`(net, **kwargs) | 모델 토폴로지 로그 후, 그레이디언트에 훅을 추가합니다. |
| `on_train_end`(net[, X, y]) | 트레이닝이 종료될 때 호출됩니다. |