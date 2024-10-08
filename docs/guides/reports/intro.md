---
title: Reports
description: 기계학습 프로젝트를 위한 프로젝트 관리 및 협업 툴
slug: /guides/reports
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons productLink="https://wandb.ai/stacey/deep-drive/reports/The-View-from-the-Driver-s-Seat--Vmlldzo1MTg5NQ?utm_source=fully_connected&utm_medium=blog&utm_campaign=view+from+the+drivers+seat" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb"/>

W&B Reports를 사용하여 Runs를 정리하고, 시각화를 삽입 및 자동화하며, 발견한 내용을 설명하고, 협업자와 업데이트를 공유하세요. 보고서를 LaTeX zip 파일로 쉽게 내보내거나 PDF 파일로 변환할 수 있습니다.

다음 이미지는 트레이닝 과정에서 W&B에 로그된 메트릭으로 생성된 리포트의 섹션을 보여줍니다.

![](/images/reports/safe-lite-benchmark-with-comments.png)

위 이미지가 포함된 리포트는 [여기](https://wandb.ai/stacey/saferlife/reports/SafeLife-Benchmark-Experiments--Vmlldzo0NjE4MzM)에서 확인하세요.

## 작동 방식
몇 번의 클릭만으로 협업 가능한 리포트를 작성하세요.

1. W&B App에서 프로젝트 워크스페이스로 이동합니다.
2. 워크스페이스 오른쪽 상단에 있는 **Create report** 버튼을 클릭합니다.

![](/images/reports/create_a_report_button.png)

3. **Create Report**라는 제목의 모달이 나타납니다. 리포트에 추가할 차트와 패널을 선택하십시오. (차트와 패널은 나중에 추가하거나 제거할 수 있습니다).
4. **Create report**를 클릭합니다.
5. 원하는 상태로 리포트를 편집합니다.
6. **Publish to project**을 클릭합니다.
7. **Share** 버튼을 클릭하여 협업자와 리포트를 공유합니다.

리포트를 인터랙티브하고 프로그래밍 방식으로 작성하는 방법에 대한 자세한 내용은 [Create a report](./create-a-report.md) 페이지를 참조하세요.

## 시작 방법
귀하의 유스 케이스에 따라, W&B Reports를 시작하는 데 도움이 되는 다음 리소스를 탐색하세요:

* W&B Reports에 대한 개요를 얻으려면 [비디오 데모](https://www.youtube.com/watch?v=2xeJIv_K_eI)를 확인하세요.
* 라이브 보고서 예제를 위한 [리포트 갤러리](./reports-gallery.md)를 탐색하세요.
* 워크스페이스를 생성하고 사용자 지정하는 방법을 배우려면 [Programmatic Workspaces](../../tutorials/workspaces.md) 튜토리얼을 시도하세요.
* [W&B Fully Connected](http://wandb.me/fc)에서 큐레이션된 Reports를 읽어보세요.