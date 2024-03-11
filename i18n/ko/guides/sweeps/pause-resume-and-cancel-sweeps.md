---
description: Pause, resume, and cancel a W&B Sweep with the CLI.
displayed_sidebar: default
---

# Sweeps 일시 중지, 재개, 중단 및 취소

<head>
    <title>W&B Sweeps 일시 중지, 재개, 중단 또는 취소</title>
</head>

CLI를 사용하여 W&B Sweep을 일시 중지, 재개 및 취소하세요. W&B Sweep을 일시 중지하면 에이전트에게 새 W&B Runs를 실행하지 않도록 지시합니다. Sweep이 재개되면 에이전트는 새 W&B Runs를 계속 실행하도록 지시받습니다. W&B Sweep을 중단하면 W&B 스윕 에이전트에게 새 W&B Runs를 생성하거나 실행하지 않도록 지시합니다. W&B Sweep을 취소하면 스윕 에이전트에게 현재 실행 중인 W&B Runs를 종료하고 새 Runs를 실행하지 않도록 지시합니다.

각 경우에는 W&B Sweep을 초기화할 때 생성된 W&B Sweep ID를 제공하세요. 선택적으로 새 터미널 창을 열어 다음 코맨드를 실행하세요. 새 터미널 창을 사용하면 현재 터미널 창에 출력문이 출력되고 있는 경우 코맨드를 쉽게 실행할 수 있습니다.

Sweeps를 일시 중지, 재개 및 취소하는 데 다음 가이드를 사용하세요.

### Sweeps 일시 중지

W&B Sweep을 일시 중지하여 새 W&B Runs의 실행을 일시적으로 중단하세요. W&B Sweep을 일시 중지하려면 `wandb sweep --pause` 코맨드를 사용하세요. 일시 중지하려는 W&B Sweep ID를 제공하세요.

```bash
wandb sweep --pause entity/project/sweep_ID
```

### Sweeps 재개

일시 중지된 W&B Sweep을 `wandb sweep --resume` 코맨드로 재개하세요. 재개하려는 W&B Sweep ID를 제공하세요:

```bash
wandb sweep --resume entity/project/sweep_ID
```

### Sweeps 중단

새 W&B Runs의 실행을 중단하고 현재 실행 중인 Runs를 완료하도록 W&B sweep을 마무리하세요.

```bash
wandb sweep --stop entity/project/sweep_ID
```

### Sweeps 취소

진행 중인 모든 runs를 종료하고 새 runs를 실행하지 않도록 sweep을 취소하세요. W&B Sweep을 취소하려면 `wandb sweep --cancel` 코맨드를 사용하세요. 취소하려는 W&B Sweep ID를 제공하세요.

```bash
wandb sweep --cancel entity/project/sweep_ID
```

CLI 코맨드 옵션의 전체 목록은 [wandb sweep](../../ref/cli/wandb-sweep.md) CLI 참조 가이드를 참조하세요.

### 여러 에이전트에서 sweep 일시 중지, 재개, 중단 및 취소

단일 터미널에서 여러 에이전트에 걸쳐 W&B Sweep을 일시 중지, 재개, 중단 또는 취소하세요. 예를 들어, 멀티 코어 머신을 사용하는 경우, W&B Sweep을 초기화한 후 새 터미널 창을 열고 각 새 터미널에 Sweep ID를 복사합니다.

터미널 내에서 `wandb sweep` CLI 코맨드를 사용하여 W&B Sweep을 일시 중지, 재개, 중단 또는 취소하세요. 예를 들어, 다음 코드조각은 CLI를 사용하여 여러 에이전트에 걸쳐 W&B Sweep을 일시 중지하는 방법을 보여줍니다:

```
wandb sweep --pause entity/project/sweep_ID
```

에이전트에서 Sweep을 재개하려면 Sweep ID와 함께 `--resume` 플래그를 지정하세요:

```
wandb sweep --resume entity/project/sweep_ID
```

W&B 에이전트를 병렬로 실행하는 방법에 대한 자세한 내용은 [에이전트 병렬화하기](./parallelize-agents.md)를 참조하세요.