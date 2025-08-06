---
title: 런치 작업 보기
menu:
  launch:
    identifier: ko-launch-create-and-deploy-jobs-launch-view-jobs
    parent: create-and-deploy-jobs
url: guides/launch/launch-view-jobs
---

다음 페이지에서는 큐에 추가된 launch job 에 대한 정보를 확인하는 방법을 안내합니다.

## Job 확인하기

W&B App 에서 큐에 추가된 job 들을 확인할 수 있습니다.

1. https://wandb.ai/home 에서 W&B App 으로 이동하세요.
2. 왼쪽 사이드바의 **Applications** 섹션에서 **Launch** 를 선택하세요.
3. **All entities** 드롭다운에서 launch job 이 속한 entity 를 선택하세요.
4. Launch Application 페이지에서 확장 가능한 UI를 펼쳐 해당 큐에 추가된 job 들의 목록을 확인하세요.

{{% alert %}}
launch agent 가 launch job 을 실행할 때 하나의 run 이 생성됩니다. 즉, 큐에 표시되는 각 run 은 해당 큐에 추가된 특정 job 에 해당합니다.
{{% /alert %}}

예를 들어, 아래 이미지는 `job-source-launch_demo-canonical` 이라는 job 에서 생성된 두 개의 run 을 보여줍니다. 이 job 은 `Start queue` 라는 큐에 추가되었습니다. 큐에서 첫 번째로 나오는 run 은 `resilient-snowball` 이고, 두 번째는 `earthy-energy-165` 입니다.

{{< img src="/images/launch/launch_jobs_status.png" alt="Launch jobs status view" >}}

W&B App UI 내에서 launch job 실행으로 생성된 run 에 대한 추가 정보를 확인할 수 있습니다. 예시는 다음과 같습니다:
   - **Run**: 해당 job 에 할당된 W&B run 의 이름
   - **Job ID**: job 의 이름
   - **Project**: run 이 속한 project 이름
   - **Status**: 큐에 있는 run 의 상태
   - **Author**: run 을 생성한 W&B entity
   - **Creation date**: 큐가 생성된 시점의 타임스탬프
   - **Start time**: job 시작 시점의 타임스탬프
   - **Duration**: job 의 run 이 완료되기까지 걸린 시간(초 단위)

## Job 리스트 확인
W&B CLI 를 이용해 project 내에 존재하는 job 목록을 확인할 수 있습니다. W&B job list 코맨드에 launch job 이 속한 project 와 entity 의 이름을 `--project` 및 `--entity` 플래그로 각각 지정하세요.

```bash
 wandb job list --entity your-entity --project project-name
```

## Job 상태 확인하기

다음 표는 큐에 들어간 run 이 가질 수 있는 상태를 설명합니다:

| Status | 설명 |
| --- | --- |
| **Idle** | run 이 활성 에이전트가 없는 큐에 있습니다. |
| **Queued** | run 이 에이전트가 처리하기를 기다리는 큐에 있습니다. |
| **Pending** | run 이 에이전트에 의해 할당되었으나 아직 시작되지 않았습니다. 클러스터의 리소스 부족으로 대기 중일 수 있습니다. |
| **Running** | run 이 현재 실행 중입니다. |
| **Killed** | job 이 사용자에 의해 중지되었습니다. |
| **Crashed** | run 이 데이터 전송을 중단했거나 정상적으로 시작되지 않았습니다. |
| **Failed** | run 이 0 이외의 종료 코드로 종료되었거나, 시작에 실패했습니다. |
| **Finished** | job 이 정상적으로 완료되었습니다. |