---
displayed_sidebar: default
---

# 실행 job 보기

다음 페이지에서는 큐에 추가된 실행 job에 대한 정보를 보는 방법을 설명합니다.

## Job 보기

W&B App을 사용하여 큐에 추가된 job을 확인합니다.

1. https://wandb.ai/home에서 W&B App으로 이동합니다.
2. 왼쪽 사이드바의 **Applications** 섹션에서 **Launch**를 선택합니다.
3. **All entities** 드롭다운을 선택하고 실행 job이 속한 엔터티를 선택합니다.
4. Launch Application션 페이지에서 접을 수 있는 UI를 확장하여 해당 큐에 추가된 job 목록을 확인합니다.

:::info
런치 에이전트가 launch job을 실행할 때 run이 생성됩니다. 다시 말해, 목록에 나열된 각 run은 해당 큐에 추가된 특정 job에 해당합니다.
:::

예를 들어, 다음 이미지는 `job-source-launch_demo-canonical`이라는 job에서 생성된 두 개의 run을 보여줍니다. Job은 `Start queue`라는 큐에 추가되었습니다. 큐에서 나열된 첫 번째 run은 `resilient-snowball`이고 두 번째 run은 `earthy-energy-165`입니다.


![](/images/launch/launch_jobs_status.png)

W&B App UI에서 실행 job에서 생성된 run에 대한 추가 정보를 찾을 수 있습니다. 예를 들면:
   - **Run**: 해당 job에 할당된 W&B run의 이름.
   - **Job ID**: job의 이름.
   - **Project**: run이 속한 프로젝트의 이름.
   - **Status**: 큐에 있는 run의 상태.
   - **Author**: run을 생성한 W&B 엔터티.
   - **Creating date**: 큐가 생성된 타임스탬프.
   - **Creating time**: job이 시작된 타임스탬프.
   - **Duration**: job의 run을 완료하는 데 걸린 시간(초 단위).

## Job 목록
W&B CLI를 사용하여 프로젝트 내에 존재하는 job 목록을 확인합니다. Launch jov이 속한 프로젝트와 엔터티의 이름을 각각 `--project`와 `--entity` 플래그에 제공하고 W&B job 목록 코맨드를 사용합니다.

```bash
 wandb job list --entity your-entity --project project-name
```

## 작업의 상태 확인하기

다음 표는 큐에 있는 run이 가질 수 있는 상태를 정의합니다:


| 상태 | 설명 |
| --- | --- |
| **Idle** | run이 활성 에이전트 없이 큐에 있습니다. |
| **Queued** | run이 에이전트가 처리하기를 기다리며 큐에 있습니다. |
| **Pending** | run이 에이전트에 의해 선택되었지만 아직 시작하지 않았습니다. 이는 클러스터에서 리소스가 사용 불가능한 경우일 수 있습니다. |
| **Running** | run이 현재 실행 중입니다. |
| **Killed** | 사용자에 의해 job이 취소되었습니다. |
| **Crashed** | run이 데이터를 보내지 않거나 성공적으로 시작하지 못했습니다. |
| **Failed** | run이 0이 아닌 종료 코드로 종료되었거나 run이 시작에 실패했습니다. |
| **Finished** | job이 성공적으로 완료되었습니다. |