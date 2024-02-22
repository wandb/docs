---
description: How to integrate W&B with Skorch.
slug: /guides/integrations/skorch
displayed_sidebar: default
---

# Skorch

Weights & Biases를 Skorch와 함께 사용하여 모델의 최고 성능을 자동으로 로그하고, 모든 모델 성능 메트릭, 모델 토폴로지, 그리고 각 에포크 후의 컴퓨트 리소스를 자동으로 기록할 수 있습니다. wandb_run.dir에 저장된 모든 파일은 자동으로 W&B 서버에 로그됩니다.

[예제 실행](https://app.wandb.ai/borisd13/skorch/runs/s20or4ct?workspace=user-borisd13)을 참조하세요.

## 파라미터

| 파라미터 | 유형 | 설명 |
| :--- | :--- | :--- |
| `wandb_run` |  wandb.wandb_run.Run | 데이터를 로그하는데 사용되는 wandb 실행. |
|`save_model` | bool (기본값=True)| 최고의 모델의 체크포인트를 저장하고 W&B 서버에 업로드할지 여부.|
|`keys_ignored`| str 또는 str의 리스트 (기본값=None) | tensorboard에 로그되지 않아야 할 키 또는 키의 리스트. 사용자가 제공한 키 외에도, `event_`로 시작하거나 `_best`로 끝나는 키와 같은 키는 기본적으로 무시됩니다.|

## 예제 코드

통합이 어떻게 작동하는지 확인할 수 있는 몇 가지 예제를 만들었습니다:

* [Colab](https://colab.research.google.com/drive/1Bo8SqN1wNPMKv5Bn9NjwGecBxzFlaNZn?usp=sharing): 통합을 시도해 볼 수 있는 간단한 데모
* [단계별 가이드](https://app.wandb.ai/cayush/uncategorized/reports/Automate-Kaggle-model-training-with-Skorch-and-W%26B--Vmlldzo4NTQ1NQ): Skorch 모델 성능 추적

```python
# wandb 설치
... pip install wandb

import wandb
from skorch.callbacks import WandbLogger

# wandb 실행 생성
wandb_run = wandb.init()
# 대안: W&B 계정 없이 wandb 실행 생성
wandb_run = wandb.init(anonymous="allow")

# 하이퍼파라미터 로그 (선택사항)
wandb_run.config.update({"learning rate": 1e-3, "batch size": 32})

net = NeuralNet(..., callbacks=[WandbLogger(wandb_run)])
net.fit(X, y)
```

## 메서드

| 메서드 | 설명 |
| :--- | :--- |
| `initialize`\(\) | 콜백의 초기 상태를 (재)설정합니다. |
| `on_batch_begin`\(net\[, X, y, training\]\) | 각 배치의 시작에 호출됩니다. |
| `on_batch_end`\(net\[, X, y, training\]\) | 각 배치의 끝에 호출됩니다. |
| `on_epoch_begin`\(net\[, dataset\_train, …\]\) | 각 에포크의 시작에 호출됩니다. |
| `on_epoch_end`\(net, \*\*kwargs\) | 마지막 히스토리 단계의 값들을 로그하고 최고의 모델을 저장합니다 |
| `on_grad_computed`\(net, named\_parameters\[, X, …\]\) | 그레이디언트가 계산된 후 하나의 배치 당 한 번씩, 업데이트 단계가 수행되기 전에 호출됩니다. |
| `on_train_begin`\(net, \*\*kwargs\) | 모델 토폴로지를 로그하고 그레이디언트에 대한 후크를 추가합니다 |
| `on_train_end`\(net\[, X, y\]\) | 학습의 끝에 호출됩니다. |