---
title: View launch jobs
displayed_sidebar: default
---

다음 페이지는 큐에 추가된 launch 작업에 대한 정보를 보는 방법을 설명합니다.

## 작업 보기

W&B App을 사용하여 큐에 추가된 작업을 확인하십시오.

1. https://wandb.ai/home에 접속하여 W&B App으로 이동합니다.
2. 왼쪽 사이드바의 **Applications** 섹션에서 **Launch**를 선택합니다.
3. **All entities** 드롭다운을 선택하고 launch 작업이 속한 entity를 선택합니다.
4. Launch Application 페이지에서 UI를 확장하여 해당 큐에 추가된 작업 목록을 봅니다.

:::info
런치는 launch 에이전트가 launch 작업을 실행할 때 생성됩니다. 즉, 나열된 각 런치는 해당 큐에 추가된 특정 작업에 해당합니다.
:::

예를 들어, 다음 이미지는 `job-source-launch_demo-canonical`이라는 작업에서 생성된 두 개의 런치를 보여줍니다. 이 작업은 `Start queue`라는 큐에 추가되었습니다. 큐에서 첫 번째로 나열된 런은 `resilient-snowball`이며 두 번째로 나열된 런은 `earthy-energy-165`입니다.

![](/images/launch/launch_jobs_status.png)

W&B App UI 내에서 launch 작업에서 생성된 런치에 대한 추가 정보를 찾을 수 있습니다:
   - **Run**: 해당 작업에 할당된 W&B 런의 이름입니다.
   - **Job ID**: 작업의 이름입니다.
   - **Project**: 런이 속한 프로젝트의 이름입니다.
   - **Status**: 대기 중인 런의 상태입니다.
   - **Author**: 런을 생성한 W&B entity입니다.
   - **Creation date**: 큐가 생성된 타임스탬프입니다.
   - **Start time**: 작업이 시작된 타임스탬프입니다.
   - **Duration**: 작업의 런이 완료되는데 걸린 시간(초)입니다.

## 작업 목록
W&B CLI로 프로젝트 내에 존재하는 작업 목록을 확인하세요. W&B job list 코맨드를 사용하고 `--project`와 `--entity` 플래그에 launch 작업이 속한 프로젝트와 entity의 이름을 제공합니다.

```bash
 wandb job list --entity your-entity --project project-name
```

## 작업 상태 확인

다음 표는 큐에 있는 런이 가질 수 있는 상태를 정의합니다:

| 상태 | 설명 |
| --- | --- |
| **Idle** | 런이 활성 에이전트 없이 큐에 있습니다. |
| **Queued** | 런이 에이전트의 프로세스를 기다리며 큐에 있습니다. |
| **Pending** | 런이 에이전트에 의해 선택되었지만 아직 시작되지 않았습니다. 이는 클러스터에서 리소스가 이용 가능하지 않은 경우일 수 있습니다. |
| **Running** | 런이 현재 실행되고 있습니다. |
| **Killed** | 작업이 사용자에 의해 중단되었습니다. |
| **Crashed** | 런이 데이터를 보내지 않거나 성공적으로 시작되지 않았습니다. |
| **Failed** | 런이 비정상 종료 코드로 종료되었거나 시작되지 않았습니다. |
| **Finished** | 작업이 성공적으로 완료되었습니다. |