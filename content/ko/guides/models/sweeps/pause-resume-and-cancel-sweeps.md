---
title: Manage sweeps with the CLI
description: CLI를 사용하여 W&B 스윕을 일시 중지, 재개 및 취소합니다.
menu:
  default:
    identifier: ko-guides-models-sweeps-pause-resume-and-cancel-sweeps
    parent: sweeps
weight: 8
---

CLI를 사용하여 W&B 스윕을 일시 중지, 재개 및 취소합니다. W&B 스윕을 일시 중지하면 스윕이 재개될 때까지 새로운 W&B Runs이 실행되지 않도록 W&B 에이전트에 알립니다. 스윕을 재개하면 에이전트가 새로운 W&B Runs을 계속 실행합니다. W&B 스윕을 중지하면 W&B 스윕 에이전트가 새로운 W&B Runs의 생성 또는 실행을 중지합니다. W&B 스윕을 취소하면 스윕 에이전트가 현재 실행 중인 W&B Runs을 중단하고 새로운 Runs 실행을 중지합니다.

각각의 경우, W&B 스윕을 초기화할 때 생성된 W&B 스윕 ID를 제공합니다. 선택적으로 새 터미널 창을 열어 다음 명령을 실행합니다. 새 터미널 창은 W&B 스윕이 현재 터미널 창에 출력문을 인쇄하는 경우 명령을 더 쉽게 실행할 수 있도록 합니다.

다음 지침에 따라 스윕을 일시 중지, 재개 및 취소합니다.

### 스윕 일시 중지

새로운 W&B Runs 실행을 일시적으로 중단하도록 W&B 스윕을 일시 중지합니다. `wandb sweep --pause` 코맨드를 사용하여 W&B 스윕을 일시 중지합니다. 일시 중지할 W&B 스윕 ID를 제공합니다.

```bash
wandb sweep --pause entity/project/sweep_ID
```

### 스윕 재개

`wandb sweep --resume` 코맨드로 일시 중지된 W&B 스윕을 재개합니다. 재개할 W&B 스윕 ID를 제공합니다.

```bash
wandb sweep --resume entity/project/sweep_ID
```

### 스윕 중지

새로운 W&B Runs 실행을 중지하고 현재 실행 중인 Runs을 완료하도록 W&B 스윕을 완료합니다.

```bash
wandb sweep --stop entity/project/sweep_ID
```

### 스윕 취소

실행 중인 모든 Runs을 중단하고 새로운 Runs 실행을 중지하도록 스윕을 취소합니다. `wandb sweep --cancel` 코맨드를 사용하여 W&B 스윕을 취소합니다. 취소할 W&B 스윕 ID를 제공합니다.

```bash
wandb sweep --cancel entity/project/sweep_ID
```

전체 CLI 코맨드 옵션 목록은 [wandb sweep]({{< relref path="/ref/cli/wandb-sweep.md" lang="ko" >}}) CLI 레퍼런스 가이드를 참조하세요.

### 여러 에이전트에서 스윕 일시 중지, 재개, 중지 및 취소

단일 터미널에서 여러 에이전트에 걸쳐 W&B 스윕을 일시 중지, 재개, 중지 또는 취소합니다. 예를 들어, 다중 코어 머신이 있다고 가정합니다. W&B 스윕을 초기화한 후 새 터미널 창을 열고 각 새 터미널에 스윕 ID를 복사합니다.

터미널 내에서 `wandb sweep` CLI 코맨드를 사용하여 W&B 스윕을 일시 중지, 재개, 중지 또는 취소합니다. 예를 들어, 다음 코드 조각은 CLI를 사용하여 여러 에이전트에서 W&B 스윕을 일시 중지하는 방법을 보여줍니다.

```
wandb sweep --pause entity/project/sweep_ID
```

에이전트에서 스윕을 재개하려면 스윕 ID와 함께 `--resume` 플래그를 지정합니다.

```
wandb sweep --resume entity/project/sweep_ID
```

W&B 에이전트를 병렬화하는 방법에 대한 자세한 내용은 [에이전트 병렬화]({{< relref path="./parallelize-agents.md" lang="ko" >}})를 참조하세요.
