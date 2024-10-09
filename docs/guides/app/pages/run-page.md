---
title: Run page
description: 모델의 각 트레이닝 run은 더 큰 프로젝트 내에서 정리된 전용 페이지를 갖습니다.
displayed_sidebar: default
---

런 페이지를 사용하여 프로젝트 내의 특정 run에 대한 자세한 정보를 탐색하세요.

## Overview 탭
프로젝트 내 특정 run에 대해 알아보려면 Overview 탭을 사용하세요. 예를 들어:

* **Name**: run의 이름
* **Description**: run에 대한 설명
* **Author**: run을 생성한 W&B 엔티티
* **State**: [run의 상태](#run-states)
* **Start time**: run이 초기화된 타임스탬프
* **Duration**: run이 어떤 상태에 도달할 때까지 걸린 시간
* **Run path**: 고유한 run 경로 `<entity>/<project>/<run_id>`
* **OS**: run을 초기화한 운영 체제
* **Python version**: run을 생성한 Python 버전
* **Git repository**: [Git이 활성화된 경우](../settings-page/user-settings.md#personal-github-integration), run과 연관된 Git 리포지토리
* **Command**: run을 초기화한 코맨드
* **System hardware**: run이 실행된 하드웨어
* **Config**: [`wandb.config`](../../../guides/track/config.md)로 저장된 설정 파라미터 리스트
* **Summary**: [`wandb.log()`](../../../guides/track/log/intro.md)로 저장된 요약 파라미터 리스트, 기본적으로 마지막으로 로그된 값으로 설정

특정 run의 overview 페이지를 보려면:
1. 프로젝트 워크스페이스 내에서 특정 run을 클릭하세요.
2. 다음으로 좌측 패널에서 **Overview** 탭을 클릭하세요.

![W&B Dashboard run overview tab](/images/app_ui/wandb_run_overview_page.png)

### Run states
다음 표는 run이 가질 수 있는 가능한 상태를 설명합니다:

| State | Description |
| ----- | ----- |
| Finished| run이 종료되고 데이터가 완전히 동기화되었거나 `wandb.finish()`가 호출됨 |
| Failed | run이 비정상 종료 상태로 끝남 | 
| Crashed | 머신이 중지될 경우 발생할 수 있는 내부 프로세스에서 run이 하트비트를 보내지 않음 | 
| Running | run이 아직 실행 중이며 최근에 하트비트를 전송함 |

## Workspace 탭
Workspace 탭을 사용하여 시각화를 보고, 검색하고, 그룹화하고 정렬할 수 있습니다. 예를 들어:

* 검증 세트에 의해 생성된 예측값과 같은 자동 생성된 플롯
* 커스텀 플롯
* 시스템 메트릭 등

![](/images/app_ui/wandb-run-page-workspace-tab.png)

W&B 앱 UI를 통해 수동으로 또는 W&B Python SDK를 사용하여 프로그래밍 방식으로 차트를 생성하세요. 더 많은 정보는 [Log media and objects in Experiments](../../track/log/intro.md)를 참조하세요.

## System 탭
**System 탭**은 특정 run에 대해 추적된 시스템 메트릭을 보여줍니다. 예를 들어:

* CPU 사용률 시각화
* 시스템 메모리
* 디스크 I/O
* 네트워크 트래픽
* GPU 사용률
* GPU 온도
* 메모리 엑세스에 소비한 GPU 시간
* 할당된 GPU 메모리
* GPU 전력 소모

[여기에서 라이브 예제를 보세요](https://wandb.ai/stacey/deep-drive/runs/ki2biuqy/system?workspace=user-carey).

![](/images/app_ui/wandb_system_utilization.png)

["Tracking System Resource"](https://lambdalabs.com/blog/weights-and-bias-gpu-cpu-utilization/) 블로그를 통해 W&B 시스템 메트릭 사용법에 대한 자세한 정보를 확인하세요.

## Model 탭

**Model 탭**에서 모델의 레이어, 파라미터 수, 각 레이어의 출력 모양을 확인하세요.

[여기에서 라이브 예제를 보세요](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/model).

![](/images/app_ui/wandb_run_page_model_tab.png)

## Logs 탭
**Log 탭**은 커맨드라인에 출력된 stdout 및 stderr과 같은 출력을 보여줍니다. W&B는 마지막 10,000줄을 보여줍니다.

로그 파일을 다운로드하려면 오른쪽 상단의 **Download** 버튼을 클릭하세요.

[여기에서 라이브 예제를 보세요](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/logs).

![](/images/app_ui/wandb_run_page_log_tab.png)

## Files 탭
**Files 탭**을 사용하여 특정 run과 관련된 파일을 확인하세요. 모델 체크포인트, 검증 세트 예제 등을 포함하세요.

[여기에서 라이브 예제를 보세요](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/files/media/images).

![](/images/app_ui/wandb_run_page_files_tab.png)

:::tip
Runs의 입력 및 출력을 추적하려면 W&B [Artifacts](../../artifacts/intro.md)를 사용하세요. Artifacts 퀵스타트는 [여기](../../artifacts/artifacts-walkthrough.md)를 참조하세요.
:::

## Artifacts 탭
Artifacts 탭은 지정된 run의 입력 및 출력 [Artifacts](../../artifacts/intro.md)를 나열합니다.

[여기에서 라이브 예제를 보세요](https://wandb.ai/stacey/artifact_july_demo/runs/2cslp2rt/artifacts).

![](/images/app_ui/artifacts_tab.png)