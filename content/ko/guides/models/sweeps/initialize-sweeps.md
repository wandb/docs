---
title: Initialize a sweep
description: W&B 스윕 초기화
menu:
  default:
    identifier: ko-guides-models-sweeps-initialize-sweeps
    parent: sweeps
weight: 4
---

W&B는 클라우드 (표준), 로컬 (local) 환경에서 하나 이상의 머신에서 Sweeps 를 관리하기 위해 _Sweep Controller_ 를 사용합니다. Run이 완료되면, 스윕 컨트롤러는 실행할 새로운 Run을 설명하는 새로운 명령어 세트를 발행합니다. 이 명령어는 실제로 Run을 수행하는 _에이전트_ 에 의해 선택됩니다. 일반적인 W&B 스윕에서 컨트롤러는 W&B 서버에 존재합니다. 에이전트는 _사용자_ 의 머신에 존재합니다.

다음 코드 조각은 CLI 및 Jupyter Notebook 또는 Python 스크립트 내에서 스윕을 초기화하는 방법을 보여줍니다.

{{% alert color="secondary" %}}
1. 스윕을 초기화하기 전에 YAML 파일 또는 스크립트의 중첩된 Python dictionary 오브젝트에 스윕 구성이 정의되어 있는지 확인하세요. 자세한 내용은 [스윕 구성 정의]({{< relref path="/guides/models/sweeps/define-sweep-configuration.md" lang="ko" >}})를 참조하세요.
2. W&B Sweep 과 W&B Run 은 동일한 Projects 에 있어야 합니다. 따라서 W&B를 초기화할 때 제공하는 이름([`wandb.init`]({{< relref path="/ref/python/init.md" lang="ko" >}}))은 W&B Sweep 을 초기화할 때 제공하는 Projects 이름([`wandb.sweep`]({{< relref path="/ref/python/sweep.md" lang="ko" >}}))과 일치해야 합니다.
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="Python script or notebook" %}}

W&B SDK를 사용하여 스윕을 초기화합니다. 스윕 구성 dictionary 를 `sweep` 파라미터에 전달합니다. 선택적으로 W&B Run 의 출력을 저장할 Projects 파라미터 (`project`)에 대한 프로젝트 이름을 제공합니다. 프로젝트가 지정되지 않은 경우 Run은 "Uncategorized" 프로젝트에 배치됩니다.

```python
import wandb

# 스윕 구성 예시
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

[`wandb.sweep`]({{< relref path="/ref/python/sweep" lang="ko" >}}) 함수는 스윕 ID를 반환합니다. 스윕 ID에는 Entities 이름과 Projects 이름이 포함됩니다. 스윕 ID를 기록해 두세요.

{{% /tab %}}
{{% tab header="CLI" %}}

W&B CLI를 사용하여 스윕을 초기화합니다. 구성 파일 이름을 제공합니다. 선택적으로 `project` 플래그에 대한 프로젝트 이름을 제공합니다. 프로젝트가 지정되지 않은 경우 W&B Run은 "Uncategorized" 프로젝트에 배치됩니다.

[`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ko" >}}) 코맨드를 사용하여 스윕을 초기화합니다. 다음 코드 예제는 `sweeps_demo` 프로젝트에 대한 스윕을 초기화하고 구성에 `config.yaml` 파일을 사용합니다.

```bash
wandb sweep --project sweeps_demo config.yaml
```

이 코맨드는 스윕 ID를 출력합니다. 스윕 ID에는 Entities 이름과 Projects 이름이 포함됩니다. 스윕 ID를 기록해 두세요.

{{% /tab %}}
{{< /tabpane >}}
