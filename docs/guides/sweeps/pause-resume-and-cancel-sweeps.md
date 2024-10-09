---
title: Manage sweeps with the CLI
description: CLI를 사용하여 W&B 스윕을 일시 중지, 재개 및 취소합니다.
displayed_sidebar: default
---

W&B Sweep을 CLI로 일시 중지, 재개 및 취소하십시오. W&B Sweep을 일시 중지하면 W&B 에이전트에게 Sweep이 재개될 때까지 새로운 W&B Run이 실행되지 않음을 알려줍니다. Sweep을 재개하면 에이전트에게 새로운 W&B Run을 계속 실행하도록 지시합니다. W&B Sweep을 중지하면 W&B Sweep 에이전트에게 새로운 W&B Run을 생성하거나 실행하지 않도록 지시합니다. W&B Sweep을 취소하면 Sweep 에이전트에게 현재 실행 중인 W&B Run을 중지하고 새로운 Run 실행을 중단하도록 지시합니다.

각 경우에는 W&B Sweep을 초기화할 때 생성된 W&B Sweep ID를 제공하십시오. 진행할 명령을 실행하려면 선택적으로 새로운 터미널 창을 열 수 있습니다. 새로운 터미널 창은 현재 터미널 창에 W&B Sweep이 출력 문을 인쇄하고 있을 때 명령을 실행하기 쉽게 만듭니다.

다음의 가이드를 사용하여 sweep을 일시 중지, 재개 및 취소하십시오.

### Pause sweeps

W&B Sweep을 일시 중지하여 새로운 W&B Run 실행을 일시적으로 중단합니다. `wandb sweep --pause` 코맨드를 사용하여 W&B Sweep을 일시 중지하십시오. 일시 중지하려는 W&B Sweep ID를 제공하십시오.

```bash
wandb sweep --pause entity/project/sweep_ID
```

### Resume sweeps

`wandb sweep --resume` 코맨드를 사용하여 일시 중지된 W&B Sweep을 재개하십시오. 재개하려는 W&B Sweep ID를 제공하십시오:

```bash
wandb sweep --resume entity/project/sweep_ID
```

### Stop sweeps

W&B sweep을 종료하여 새로운 W&B Run 실행을 중단하고 현재 실행 중인 Run을 완료하도록 합니다.

```bash
wandb sweep --stop entity/project/sweep_ID
```

### Cancel sweeps

모든 실행 중인 run을 중지하고 새로운 run 실행을 중단하려면 sweep을 취소하십시오. `wandb sweep --cancel` 코맨드를 사용하여 W&B Sweep을 취소하십시오. 취소하려는 W&B Sweep ID를 제공하십시오.

```bash
wandb sweep --cancel entity/project/sweep_ID
```

CLI 코맨드 옵션의 전체 목록은 [wandb sweep](../../ref/cli/wandb-sweep.md) CLI Reference Guide를 참조하십시오.

### 여러 에이전트에 걸쳐 sweep을 일시 중지, 재개, 중지 및 취소하기

여러 에이전트에 걸쳐 단일 터미널에서 W&B Sweep을 일시 중지, 재개, 중지 또는 취소하십시오. 예를 들어, 다중 코어 머신이 있다고 가정하겠습니다. W&B Sweep을 초기화한 후, 새로운 터미널 창을 열고 각 새로운 터미널에 Sweep ID를 복사합니다.

어느 터미널에서든 `wandb sweep` CLI 코맨드를 사용하여 W&B Sweep을 일시 중지, 재개, 중지 또는 취소하십시오. 예를 들어, 진행 중인 코드조각은 CLI로 여러 에이전트에 걸쳐 W&B Sweep을 일시 중지하는 방법을 보여줍니다:

```
wandb sweep --pause entity/project/sweep_ID
```

에이전트에 걸쳐 Sweep을 재개하려면 `--resume` 플래그와 Sweep ID를 지정하십시오:

```
wandb sweep --resume entity/project/sweep_ID
```

W&B 에이전트를 병렬화하는 방법에 대한 자세한 내용은 [Parallelize agents](./parallelize-agents.md)를 참조하십시오.