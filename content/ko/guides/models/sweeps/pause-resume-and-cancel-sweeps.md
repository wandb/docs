---
title: CLI로 스윕 관리하기
description: CLI를 사용하여 W&B 스윕을 일시 중지, 재개 및 취소하세요.
menu:
  default:
    identifier: ko-guides-models-sweeps-pause-resume-and-cancel-sweeps
    parent: sweeps
weight: 8
---

CLI 를 사용하여 W&B Sweep 을 일시 중지, 재개, 취소할 수 있습니다. Sweep 을 일시 중지하면 W&B 에이전트에게 Sweep 이 재개될 때까지 새로운 W&B Runs 를 실행하지 않도록 알립니다. Sweep 을 재개하면 에이전트가 새로운 W&B Runs 를 계속 실행하도록 지시합니다. W&B Sweep 을 중지하면 W&B Sweep 에이전트에게 새로운 W&B Runs 생성 또는 실행을 중단하도록 합니다. Sweep 을 취소하면 Sweep 에이전트가 현재 실행 중인 W&B Runs 를 강제 종료하고, 새로운 Run 실행도 멈춥니다.

각 경우마다, W&B Sweep 을 초기화할 때 생성된 W&B Sweep ID 를 입력해야 합니다. 새로운 터미널 창을 열어서 아래의 코맨드를 실행하는 것이 편할 수 있습니다. 만약 현재 터미널에서 W&B Sweep 이 출력문을 계속해서 띄우고 있다면, 새로운 터미널을 사용하면 명령 실행이 더 수월해집니다.

아래 안내에 따라 sweeps 를 일시 중지, 재개, 취소할 수 있습니다.

### Sweep 일시 중지하기

W&B Sweep 을 일시 중지하면 일시적으로 새로운 W&B Runs 실행이 중지됩니다. `wandb sweep --pause` 코맨드를 사용하여 Sweep 을 일시 중지할 수 있습니다. 일시 중지할 W&B Sweep ID 를 입력하세요.

```bash
wandb sweep --pause entity/project/sweep_ID
```

### Sweep 재개하기

일시 중지된 W&B Sweep 을 `wandb sweep --resume` 코맨드로 재개할 수 있습니다. 재개할 Sweep 의 ID 를 입력하세요:

```bash
wandb sweep --resume entity/project/sweep_ID
```

### Sweep 중지하기

W&B sweep 을 종료하여 새로운 W&B Runs 실행을 중단하고, 현재 실행 중인 Run 들이 마무리되도록 합니다.

```bash
wandb sweep --stop entity/project/sweep_ID
```

### Sweep 취소하기

실행 중인 모든 run 을 종료하고, 새로운 run 도 실행하지 않도록 sweep 을 취소합니다. `wandb sweep --cancel` 코맨드를 사용해서 Sweep 을 취소하세요. 취소할 Sweep ID 를 입력해야 합니다.

```bash
wandb sweep --cancel entity/project/sweep_ID
```

CLI 코맨드의 전체 옵션 목록은 [wandb sweep]({{< relref path="/ref/cli/wandb-sweep.md" lang="ko" >}}) CLI 레퍼런스 가이드를 참고하세요.

### 여러 에이전트에서 sweep 일시 중지, 재개, 중지, 취소하기

단일 터미널에서 여러 에이전트에 걸쳐 W&B Sweep 을 일시 중지, 재개, 중지 또는 취소할 수 있습니다. 예를 들어, 멀티코어 머신이 있고, W&B Sweep 을 초기화한 후 새로운 터미널 창 여러 개를 열어 각 터미널에 Sweep ID 를 복사한다고 가정합니다.

어느 터미널에서든 `wandb sweep` CLI 코맨드를 사용하여 Sweep 을 일시 중지, 재개, 중지 또는 취소할 수 있습니다. 예를 들어, 아래 코드조각처럼 CLI 를 활용하여 여러 에이전트에서 Sweep 을 일시 중지할 수 있습니다:

```
wandb sweep --pause entity/project/sweep_ID
```

`--resume` 옵션과 Sweep ID 를 함께 지정하면 여러 에이전트에서 Sweep 을 재개할 수 있습니다:

```
wandb sweep --resume entity/project/sweep_ID
```

W&B 에이전트를 병렬로 실행하는 방법에 대한 자세한 내용은 [에이전트 병렬화]({{< relref path="./parallelize-agents.md" lang="ko" >}})를 참고하세요.