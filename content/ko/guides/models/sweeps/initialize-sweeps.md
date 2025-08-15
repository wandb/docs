---
title: 스윕 초기화하기
description: W&B 스윕 초기화
menu:
  default:
    identifier: ko-guides-models-sweeps-initialize-sweeps
    parent: sweeps
weight: 4
---

W&B는 _Sweep 컨트롤러_ 를 사용하여 클라우드(standard) 또는 하나 이상의 머신에서 로컬(local)로 sweeps를 관리합니다. 각 run이 종료되면 sweep 컨트롤러는 새로운 run을 실행하기 위한 새로운 명령 세트를 발행합니다. 이 명령을 _에이전트_ 가 감지하여 실제로 run을 수행합니다. 일반적인 W&B Sweep에서 컨트롤러는 W&B 서버에 위치하며, 에이전트는 _사용자_ 의 머신에 위치합니다.

다음 코드조각은 CLI, Jupyter 노트북, 또는 Python 스크립트에서 sweeps를 어떻게 초기화하는지 보여줍니다.

{{% alert color="secondary" %}}
1. sweep을 초기화하기 전에, 반드시 YAML 파일 또는 스크립트 내 중첩 Python 딕셔너리 오브젝트 형태로 sweep 구성이 정의되어 있어야 합니다. 자세한 정보는 [스윕 구성 정의하기]({{< relref path="/guides/models/sweeps/define-sweep-configuration.md" lang="ko" >}})를 참고하세요.
2. W&B Sweep과 W&B Run은 반드시 동일한 프로젝트 안에 있어야 합니다. 따라서 W&B를 초기화할 때 입력하는 이름([`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}}))과 W&B Sweep을 초기화할 때 입력하는 프로젝트 이름([`wandb.sweep()`]({{< relref path="/ref/python/sdk/functions/sweep.md" lang="ko" >}}))이 일치해야 합니다.
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="Python 스크립트 또는 노트북" %}}

W&B SDK를 사용하여 sweep을 초기화하세요. 스윕 설정 딕셔너리를 `sweep` 파라미터로 전달합니다. 프로젝트 파라미터(`project`)를 통해 W&B Run 결과를 저장하고 싶은 프로젝트의 이름을 선택적으로 지정할 수 있습니다. 프로젝트를 지정하지 않으면 run은 "Uncategorized" 프로젝트에 저장됩니다.

```python
import wandb

# 예시 sweep 구성
sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [5, 10, 15]},
        "lr": {"max": 0.1, "min": 0.0001},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="project-name")
```

[`wandb.sweep()`]({{< relref path="/ref/python/sdk/functions/sweep.md" lang="ko" >}}) 함수는 sweep ID를 반환합니다. sweep ID에는 entity 이름과 project 이름이 포함되어 있습니다. sweep ID를 꼭 기록해 두세요.

{{% /tab %}}
{{% tab header="CLI" %}}

W&B CLI를 사용하여 sweep을 초기화하세요. 설정 파일의 이름을 제공하세요. `project` 플래그를 이용해 프로젝트 이름을 선택적으로 지정할 수 있습니다. 프로젝트를 지정하지 않으면 W&B Run은 "Uncategorized" 프로젝트에 저장됩니다.

[`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ko" >}}) 커맨드를 사용해 sweep을 초기화합니다. 아래 예제에서는 `sweeps_demo` 프로젝트와 `config.yaml` 파일을 사용하여 sweep을 초기화합니다.

```bash
wandb sweep --project sweeps_demo config.yaml
```

이 명령을 실행하면 sweep ID가 출력됩니다. sweep ID에는 entity 이름과 project 이름이 포함되어 있습니다. sweep ID를 꼭 기록해 두세요.

{{% /tab %}}
{{< /tabpane >}}