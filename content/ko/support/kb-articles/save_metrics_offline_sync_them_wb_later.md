---
title: 메트릭을 오프라인으로 저장한 후 나중에 W&B에 동기화할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-save_metrics_offline_sync_them_wb_later
support:
- Experiments
- 환경 변수
- 메트릭
toc_hide: true
type: docs
url: /support/:filename
---

기본적으로, `wandb.init`은 메트릭을 실시간으로 클라우드에 동기화하는 프로세스를 시작합니다. 오프라인으로 사용하려면, 두 개의 환경 변수를 설정하여 오프라인 모드를 활성화하고 나중에 동기화할 수 있습니다.

다음 환경 변수를 설정하세요:

1. `WANDB_API_KEY=$KEY` — 여기서 `$KEY`는 [설정 페이지](https://app.wandb.ai/settings)에서 확인한 API 키입니다.
2. `WANDB_MODE="offline"`

아래는 이를 스크립트에서 구현하는 예시입니다:

```python
import wandb
import os

# 환경 변수 설정
os.environ["WANDB_API_KEY"] = "YOUR_KEY_HERE"
os.environ["WANDB_MODE"] = "offline"

config = {
    # 데이터셋 지정
    "dataset": "CIFAR10",
    "machine": "offline cluster",
    "model": "CNN",
    "learning_rate": 0.01,
    "batch_size": 128,
}

# offline-demo 프로젝트에서 run을 시작
with wandb.init(project="offline-demo") as run:
    for i in range(100):
        run.log({"accuracy": i})
```

아래는 예시 터미널 출력입니다:

{{< img src="/images/experiments/sample_terminal_output.png" alt="Offline mode terminal output" >}}

작업 완료 후, 다음 코맨드를 실행하여 데이터를 클라우드로 동기화하세요:

```shell
wandb sync wandb/dryrun-folder-name
```

{{< img src="/images/experiments/sample_terminal_output_cloud.png" alt="Cloud sync terminal output" >}}