---
description: Each training run of your model gets a dedicated page, organized within
  the larger project
displayed_sidebar: default
---

# Run 페이지

run 페이지를 사용하여 모델의 단일 버전에 대한 자세한 정보를 탐색하세요.

## Overview 탭

* Run 이름, 설명, 태그
* Run 상태
  * **finished**: 스크립트가 종료되고 데이터가 완전히 동기화되었거나 `wandb.finish()`가 호출됨
  * **failed**: 스크립트가 0이 아닌 종료 상태로 종료됨
  * **crashed**: 스크립트가 내부 프로세스에서 심장박동을 보내지 않아 중지됨, 이는 기계가 충돌했을 때 발생할 수 있음
  * **running**: 스크립트가 여전히 실행 중이며 최근에 심장박동을 보냄
* 호스트 이름, 운영 체제, Python 버전 및 run을 시작한 코맨드
* [`wandb.config`](../../../guides/track/config.md)로 저장된 설정 파라미터 목록
* [`wandb.log()`](../../../guides/track/log/intro.md)로 저장된 요약 파라미터 목록, 기본적으로 마지막에 로그된 값으로 설정됨

[실시간 예시 보기 →](https://app.wandb.ai/carey/pytorch-cnn-fashion/runs/munu5vvg/overview?workspace=user-carey)

![W&B 대시보드 run overview 탭](/images/app_ui/wandb_run_overview_page.png)

Python 세부 정보는 페이지 자체를 공개하더라도 비공개입니다. 여기 제 run 페이지의 익명 상태와 제 계정의 오른쪽 예입니다.

![](/images/app_ui/wandb_run_overview_page_2.png)

## 차트 탭

* 시각화 검색, 그룹화 및 배열
  * 검색창은 정규 표현식을 지원함
* 그래프에서 연필 아이콘 ✏️을 클릭하여 편집
  * x축, 메트릭, 범위 변경
  * 차트의 범례, 제목, 색상 편집
* 검증 세트에서 예시 예측값을 봄
* 이 차트들을 얻으려면 [`wandb.log()`](../../../guides/track/log/intro.md)로 데이터를 로그하세요

![](/images/app_ui/wandb-run-page-workspace-tab.png)

## 시스템 탭

* CPU 사용량, 시스템 메모리, 디스크 I/O, 네트워크 트래픽, GPU 사용량, GPU 온도, GPU 메모리 접근 시간 소요, GPU 메모리 할당량, GPU 전력 사용량 시각화
* Lambda Labs에서 W&B 시스템 메트릭 사용 방법을 [블로그 포스트 →](https://lambdalabs.com/blog/weights-and-bias-gpu-cpu-utilization/)에서 설명했습니다.

[실시간 예시 보기 →](https://wandb.ai/stacey/deep-drive/runs/ki2biuqy/system?workspace=user-carey)

![](/images/app_ui/wandb_system_utilization.png)

## 모델 탭

* 모델의 레이어, 파라미터 수, 각 레이어의 출력 모양을 확인

[실시간 예시 보기 →](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/model)

![](/images/app_ui/wandb_run_page_model_tab.png)

## 로그 탭

* 모델을 트레이닝하는 기계의 커맨드라인, stdout 및 stderr에 출력된 내용
* 마지막 1000줄을 보여줍니다. run이 종료된 후 전체 로그 파일을 다운로드하려면 오른쪽 상단의 다운로드 버튼을 클릭하세요.

[실시간 예시 보기 →](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/logs)

![](/images/app_ui/wandb_run_page_log_tab.png)

## 파일 탭

* [`wandb.save()`](../../track/save-restore.md)를 사용하여 run과 동기화할 파일 저장
* 모델 체크포인트, 검증 세트 예시 등 유지
* 코드의 정확한 버전을 [복원](../../track/save-restore.md)하기 위해 `diff.patch` 사용
  [실시간 예시 보기 →](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/files/media/images)

:::안내
W&B [Artifacts](../../artifacts/intro.md) 시스템은 데이터셋 및 모델과 같은 대용량 파일을 처리, 버전 관리, 중복 제거하기 위한 추가 기능을 제공합니다. run의 입력 및 출력을 추적하기 위해 `wandb.save` 대신 Artifacts를 사용하는 것이 좋습니다. 여기서 Artifacts 퀵스타트를 확인하세요 [여기](../../artifacts/artifacts-walkthrough.md).
:::

![](/images/app_ui/wandb_run_page_files_tab.png)

## Artifacts 탭

* 이 run의 입력 및 출력 [Artifacts](../../artifacts/intro.md)에 대한 검색 가능한 목록 제공
* 특정 아티팩트에 대한 정보를 보려면 행을 클릭하세요
* 웹 앱에서 아티팩트 뷰어를 탐색하고 사용하는 방법에 대한 자세한 내용은 [프로젝트](project-page.md) 수준 [Artifacts 탭](project-page.md#artifacts-tab) 참조를 확인하세요 [실시간 예시 보기 →](https://wandb.ai/stacey/artifact\_july\_demo/runs/2cslp2rt/artifacts)

![](/images/app_ui/artifacts_tab.png)