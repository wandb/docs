---
description: Pause, resume, and cancel a W&B Sweep with the CLI.
displayed_sidebar: default
---

# 스윕 일시 중지, 재개, 중단 및 취소

<head>
    <title>W&B 스윕 일시 중지, 재개, 중단 또는 취소</title>
</head>

W&B 스윕을 CLI를 사용하여 일시 중지, 재개 및 취소합니다. W&B 스윕을 일시 중지하면 W&B 에이전트에 새로운 W&B 실행이 스윕이 재개될 때까지 실행되지 않도록 지시합니다. 스윕을 재개하면 에이전트에게 새로운 W&B 실행을 계속 실행하도록 지시합니다. W&B 스윕을 중단하면 W&B 스윕 에이전트에게 새로운 W&B 실행을 생성하거나 실행하지 않도록 지시합니다. W&B 스윕을 취소하면 스윕 에이전트에게 현재 실행 중인 W&B 실행을 종료하고 새로운 실행을 중단하도록 지시합니다.

각 경우에 W&B 스윕 ID를 제공합니다. 이 ID는 W&B 스윕을 초기화할 때 생성되었습니다. 필요한 경우 새 터미널 창을 열어 다음 명령을 실행할 수 있습니다. 새 터미널 창을 사용하면 현재 터미널 창에 출력 문을 출력하는 W&B 스윕이 있는 경우 명령을 실행하기가 더 쉽습니다.

스윕을 일시 중지, 재개 및 취소하는 다음 가이드를 사용하십시오.

### 스윕 일시 중지

W&B 스윕을 일시 중지하여 새로운 W&B 실행을 일시적으로 중단합니다. `wandb sweep --pause` 명령을 사용하여 W&B 스윕을 일시 중지합니다. 일시 중지하려는 W&B 스윕 ID를 제공합니다.

```bash
wandb sweep --pause entity/project/sweep_ID
```

### 스윕 재개

`wandb sweep --resume` 명령을 사용하여 일시 중지된 W&B 스윕을 재개합니다. 재개하려는 W&B 스윕 ID를 제공합니다:

```bash
wandb sweep --resume entity/project/sweep_ID
```

### 스윕 중단

새로운 W&B 실행을 중단하고 현재 실행 중인 실행이 완료되도록 W&B 스윕을 마무리합니다.

```bash
wandb sweep --stop entity/project/sweep_ID
```

### 스윕 취소

모든 실행 중인 실행을 종료하고 새로운 실행을 중단하려면 스윕을 취소합니다. `wandb sweep --cancel` 명령을 사용하여 W&B 스윕을 취소합니다. 취소하려는 W&B 스윕 ID를 제공합니다.

```bash
wandb sweep --cancel entity/project/sweep_ID
```

CLI 명령 옵션의 전체 목록은 [wandb sweep](../../ref/cli/wandb-sweep.md) CLI 참조 가이드를 참조하십시오.

### 여러 에이전트에서 스윕 일시 중지, 재개, 중단 및 취소

단일 터미널에서 여러 에이전트에 걸쳐 W&B 스윕을 일시 중지, 재개, 중단 또는 취소합니다. 예를 들어, 멀티 코어 기계를 가지고 있다고 가정해 보겠습니다. W&B 스윕을 초기화한 후 새 터미널 창을 열고 각 새 터미널에 스윕 ID를 복사합니다.

어떤 터미널에서든 `wandb sweep` CLI 명령을 사용하여 W&B 스윕을 일시 중지, 재개, 중단 또는 취소할 수 있습니다. 예를 들어, 다음 코드 조각은 CLI를 사용하여 여러 에이전트에 걸쳐 W&B 스윕을 일시 중지하는 방법을 보여줍니다:

```
wandb sweep --pause entity/project/sweep_ID
```

스윕 ID와 함께 `--resume` 플래그를 지정하여 에이전트에서 스윕을 재개합니다:

```
wandb sweep --resume entity/project/sweep_ID
```

W&B 에이전트를 병렬화하는 방법에 대한 자세한 내용은 [에이전트 병렬화](./parallelize-agents.md)를 참조하십시오.