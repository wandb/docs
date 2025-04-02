---
title: View launch jobs
menu:
  launch:
    identifier: ko-launch-create-and-deploy-jobs-launch-view-jobs
    parent: create-and-deploy-jobs
url: guides/launch/launch-view-jobs
---

다음 페이지에서는 대기열에 추가된 Launch 작업에 대한 정보를 보는 방법을 설명합니다.

## 작업 보기

W&B 앱으로 대기열에 추가된 작업을 봅니다.

1. https://wandb.ai/home에서 W&B 앱으로 이동합니다.
2. 왼쪽 사이드바의 **Applications** 섹션에서 **Launch**를 선택합니다.
3. **All entities** 드롭다운을 선택하고 Launch 작업이 속한 entity를 선택합니다.
4. Launch Application 페이지에서 축소 가능한 UI를 확장하여 해당 특정 대기열에 추가된 작업 목록을 봅니다.

{{% alert %}}
Launch 에이전트가 Launch 작업을 실행하면 run이 생성됩니다. 즉, 나열된 각 run은 해당 대기열에 추가된 특정 작업에 해당합니다.
{{% /alert %}}

예를 들어 다음 이미지는 `job-source-launch_demo-canonical`이라는 작업에서 생성된 두 개의 run을 보여줍니다. 이 작업은 `Start queue`라는 대기열에 추가되었습니다. 대기열에 나열된 첫 번째 run은 `resilient-snowball`이라고 하고 두 번째 run은 `earthy-energy-165`라고 합니다.

{{< img src="/images/launch/launch_jobs_status.png" alt="" >}}

W&B 앱 UI 내에서 Launch 작업에서 생성된 run에 대한 다음과 같은 추가 정보를 찾을 수 있습니다.
   - **Run**: 해당 작업에 할당된 W&B run의 이름입니다.
   - **Job ID**: 작업의 이름입니다.
   - **Project**: run이 속한 project의 이름입니다.
   - **Status**: 대기열에 있는 run의 상태입니다.
   - **Author**: run을 생성한 W&B entity입니다.
   - **Creation date**: 대기열이 생성된 타임스탬프입니다.
   - **Start time**: 작업이 시작된 타임스탬프입니다.
   - **Duration**: 작업을 완료하는 데 걸린 시간(초)입니다.

## 작업 나열

W&B CLI를 사용하여 project 내에 존재하는 작업 목록을 봅니다. W&B 작업 목록 코맨드를 사용하고 Launch 작업이 속한 project 및 entity의 이름을 각각 `--project` 및 `--entity` 플래그와 함께 제공합니다.

```bash
 wandb job list --entity your-entity --project project-name
```

## 작업 상태 확인

다음 표는 대기열에 있는 run이 가질 수 있는 상태를 정의합니다.

| 상태 | 설명 |
| --- | --- |
| **Idle** | run이 활성 에이전트가 없는 대기열에 있습니다. |
| **Queued** | run이 에이전트가 처리하기를 기다리는 대기열에 있습니다. |
| **Pending** | run이 에이전트에 의해 선택되었지만 아직 시작되지 않았습니다. 이는 클러스터에서 리소스를 사용할 수 없기 때문일 수 있습니다. |
| **Running** | run이 현재 실행 중입니다. |
| **Killed** | 사용자가 작업을 중단했습니다. |
| **Crashed** | run이 데이터 전송을 중단했거나 성공적으로 시작되지 않았습니다. |
| **Failed** | run이 0이 아닌 종료 코드로 종료되었거나 run을 시작하지 못했습니다. |
| **Finished** | 작업이 성공적으로 완료되었습니다. |
