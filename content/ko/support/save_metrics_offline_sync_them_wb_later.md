---
title: Is it possible to save metrics offline and sync them to W&B later?
menu:
  support:
    identifier: ko-support-save_metrics_offline_sync_them_wb_later
tags:
- experiments
- environment variables
- metrics
toc_hide: true
type: docs
---

기본적으로 `wandb.init`는 메트릭을 클라우드에 실시간으로 동기화하는 프로세스를 시작합니다. 오프라인으로 사용하려면 두 개의 환경 변수를 설정하여 오프라인 모드를 활성화하고 나중에 동기화하세요.

다음 환경 변수를 설정합니다.

1. `WANDB_API_KEY=$KEY`. 여기서 `$KEY`는 [설정 페이지](https://app.wandb.ai/settings)에서 API 키 입니다.
2. `WANDB_MODE="offline"`.

다음은 스크립트에서 이를 구현하는 예시입니다.

```python
import wandb
import os

os.environ["WANDB_API_KEY"] = "YOUR_KEY_HERE"
os.environ["WANDB_MODE"] = "offline"

config = {
    "dataset": "CIFAR10",
    "machine": "offline cluster",
    "model": "CNN",
    "learning_rate": 0.01,
    "batch_size": 128,
}

wandb.init(project="offline-demo")

for i in range(100):
    wandb.log({"accuracy": i})
```

샘플 터미널 출력은 아래와 같습니다.

{{< img src="/images/experiments/sample_terminal_output.png" alt="" >}}

작업을 완료한 후 다음 코맨드를 실행하여 데이터를 클라우드에 동기화합니다.

```shell
wandb sync wandb/dryrun-folder-name
```

{{< img src="/images/experiments/sample_terminal_output_cloud.png" alt="" >}}
