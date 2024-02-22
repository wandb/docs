---
description: Each training run of your model gets a dedicated page, organized within
  the larger project
displayed_sidebar: default
---

# 실행 페이지

실행 페이지를 사용하여 모델의 단일 버전에 대한 자세한 정보를 탐색하세요.

## Overview 탭

* 실행 이름, 설명, 태그
* 실행 상태
  * **완료됨**: 스크립트가 종료되고 전체 데이터가 동기화되었거나 `wandb.finish()`가 호출됨
  * **실패함**: 스크립트가 0이 아닌 종료 상태로 종료됨
  * **충돌함**: 내부 프로세스에서 스크립트가 하트비트를 보내지 않아 중단됨, 기계가 충돌하면 발생할 수 있음
  * **실행 중**: 스크립트가 여전히 실행 중이며 최근에 하트비트를 보냄
* 호스트 이름, 운영 체제, Python 버전, 실행을 시작한 명령
* [`wandb.config`](../../../guides/track/config.md)로 저장된 설정 파라미터 목록
* [`wandb.log()`](../../../guides/track/log/intro.md)로 저장된 요약 파라미터 목록, 기본적으로 로그된 마지막 값으로 설정됨

[실시간 예시 보기 →](https://app.wandb.ai/carey/pytorch-cnn-fashion/runs/munu5vvg/overview?workspace=user-carey)

![W&B 대시보드 실행 overview 탭](/images/app_ui/wandb_run_overview_page.png)

페이지 자체를 공개하더라도 Python 세부 정보는 비공개입니다. 여기 제 실행 페이지의 익명 탭과 제 계정의 오른쪽 예가 있습니다.

![](/images/app_ui/wandb_run_overview_page_2.png)

## 차트 탭

* 시각화 검색, 그룹화 및 배열
  * 검색창은 정규 표현식을 지원함
* 그래프에서 연필 아이콘 ✏️을 클릭하여 편집
  * x축, 메트릭 및 범위 변경
  * 차트의 범례, 제목 및 색상 편집
* 검증 세트에서 예시 예측값 보기
* 이러한 차트를 얻으려면 [`wandb.log()`](../../../guides/track/log/intro.md)로 데이터를 기록하세요.

![](/images/app_ui/wandb-run-page-workspace-tab.png)

## 시스템 탭

* CPU 사용량, 시스템 메모리, 디스크 I/O, 네트워크 트래픽, GPU 사용량, GPU 온도, GPU 메모리 액세스 시간, 할당된 GPU 메모리 및 GPU 전력 사용량 시각화
* Lambda Labs에서 W&B 시스템 메트릭 사용 방법을 [블로그 게시물 →](https://lambdalabs.com/blog/weights-and-bias-gpu-cpu-utilization/)에서 강조했습니다.

[실시간 예시 보기 →](https://wandb.ai/stacey/deep-drive/runs/ki2biuqy/system?workspace=user-carey)

![](/images/app_ui/wandb_system_utilization.png)

## 모델 탭

* 모델의 레이어, 파라미터 수 및 각 레이어의 출력 형태 확인

[실시간 예시 보기 →](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/model)

![](/images/app_ui/wandb_run_page_model_tab.png)

## 로그 탭

* 명령 줄에 출력된 내용, 모델을 학습하는 기계의 stdout 및 stderr
* 마지막 1000줄을 보여줍니다. 실행이 완료된 후 전체 로그 파일을 다운로드하려면 오른쪽 상단 모서리의 다운로드 버튼을 클릭하세요.

[실시간 예시 보기 →](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/logs)

![](/images/app_ui/wandb_run_page_log_tab.png)

## 파일 탭

* 실행과 동기화할 파일을 저장하기 위해 [`wandb.save()`](../../track/save-restore.md) 사용
* 모델 체크포인트, 검증 세트 예시 등 유지
* 정확한 코드 버전을 [복원](../../track/save-restore.md)하기 위해 `diff.patch` 사용
  [실시간 예시 보기 →](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/files/media/images)

:::안내
W&B [아티팩트](../../artifacts/intro.md) 시스템은 데이터세트와 모델과 같은 큰 파일을 처리, 버전 관리 및 중복 제거를 위한 추가 기능을 제공합니다. 실행의 입력 및 출력을 추적하기 위해 `wandb.save` 대신 아티팩트를 사용하는 것이 좋습니다. [여기](../../artifacts/artifacts-walkthrough.md)에서 아티팩트 퀵스타트를 확인하세요.
:::

![](/images/app_ui/wandb_run_page_files_tab.png)

## 아티팩트 탭

* 이 실행의 입력 및 출력 [아티팩트](../../artifacts/intro.md)에 대한 검색 가능한 목록 제공
* 특정 아티팩트에 대한 정보를 보려면 행을 클릭하세요
* 웹 앱에서 아티팩트 뷰어를 탐색하고 사용하는 방법에 대한 자세한 내용은 [프로젝트](project-page.md) 수준 [아티팩트 탭](project-page.md#artifacts-tab) 참조를 확인하세요 [실시간 예시 보기 →](https://wandb.ai/stacey/artifact\_july\_demo/runs/2cslp2rt/artifacts)

![](/images/app_ui/artifacts_tab.png)