---
title: Reports
description: 기계학습 프로젝트를 위한 프로젝트 관리 및 협업 툴
cascade:
- url: guides/reports/:filename
menu:
  default:
    identifier: ko-guides-core-reports-_index
    parent: core
url: guides/reports
weight: 3
---

{{< cta-button productLink="https://wandb.ai/stacey/deep-drive/reports/The-View-from-the-Driver-s-Seat--Vmlldzo1MTg5NQ?utm_source=fully_connected&utm_medium=blog&utm_campaign=view+from+the+drivers+seat" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb" >}}

W&B Reports를 사용하면 다음과 같은 작업이 가능합니다:
- Runs를 체계적으로 정리할 수 있습니다.
- 시각화를 임베드하고 자동화할 수 있습니다.
- 발견한 내용을 설명할 수 있습니다.
- LaTeX zip 파일 또는 PDF로 협업자와 업데이트를 공유할 수 있습니다.




아래 이미지는 트레이닝 과정에서 W&B에 기록된 메트릭을 기반으로 생성된 Report의 일부를 보여줍니다.

{{< img src="/images/reports/safe-lite-benchmark-with-comments.png" alt="W&B report with benchmark results" max-width="90%" >}}

위 이미지가 사용된 Report는 [여기](https://wandb.ai/stacey/saferlife/reports/SafeLife-Benchmark-Experiments--Vmlldzo0NjE4MzM)에서 확인할 수 있습니다.

## 작동 방식
몇 번의 클릭만으로 협업용 Report를 만들 수 있습니다.

1. W&B App에서 본인의 W&B Project Workspace로 이동하세요.
2. Workspace 오른쪽 상단에 있는 **Create report** 버튼을 클릭합니다.

{{< img src="/images/reports/create_a_report_button.png" alt="Create report button" max-width="90%">}}

3. **Create Report**라는 제목의 모달이 나타납니다. Report에 추가할 차트와 패널을 선택하세요. (차트와 패널은 나중에도 추가하거나 제거할 수 있습니다.)
4. **Create report**를 클릭합니다.
5. 원하는 대로 Report를 편집합니다.
6. **Publish to project**를 클릭합니다.
7. **Share** 버튼을 클릭해 협업자와 Report를 공유할 수 있습니다.

W&B Python SDK를 이용해 직접 또는 프로그래밍 방식으로 Report를 생성하는 방법은 [Create a report]({{< relref path="./create-a-report.md" lang="ko" >}}) 페이지에서 자세히 안내합니다.

## 시작 방법
본인의 유스 케이스에 따라 아래 리소스를 참고해 W&B Reports를 시작해 보세요:

* W&B Reports의 개요를 보고 싶다면 [비디오 데모](https://www.youtube.com/watch?v=2xeJIv_K_eI)를 확인해 보세요.
* 라이브 Report 예시를 보려면 [Reports gallery]({{< relref path="./reports-gallery.md" lang="ko" >}})를 둘러보세요.
* 워크스페이스 생성 및 커스터마이즈 방법은 [Programmatic Workspaces]({{< relref path="/tutorials/workspaces.md" lang="ko" >}}) 튜토리얼에서 확인할 수 있습니다.
* [W&B Fully Connected](https://wandb.me/fc)에서 엄선된 Report를 읽어보세요.

## 모범 사례 및 팁

Experiments와 로그 기록에 대한 모범 사례와 팁은 [Best Practices: Reports](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#reports)에서 확인하세요.