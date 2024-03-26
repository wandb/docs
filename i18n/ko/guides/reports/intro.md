---
description: Project management and collaboration tools for machine learning projects
slug: /guides/reports
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# 협업 리포트

<CTAButtons productLink="https://wandb.ai/stacey/deep-drive/reports/The-View-from-the-Driver-s-Seat--Vmlldzo1MTg5NQ?utm_source=fully_connected&utm_medium=blog&utm_campaign=view+from+the+drivers+seat" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb"/>

W&B Reports를 사용래 run을 정리하고, 시각화를 내장 및 자동화하며, 발견한 내용을 설명하고, 공동 작업자와 업데이트를 공유하세요. 리포트를 LaTeX zip 파일로 쉽게 내보내거나 PDF 파일로 변환할 수 있습니다.

다음 이미지는 트레이닝 과정에서 W&B에 로깅된 메트릭으로 만든 리포트의 한 부분을 보여줍니다.

![](/images/reports/safe-lite-benchmark-with-comments.png)

위 이미지가 포함된 리포트는 [여기](https://wandb.ai/stacey/saferlife/reports/SafeLife-Benchmark-Experiments--Vmlldzo0NjE4MzM)에서 확인하세요.

## 작동 방식
단 몇 번의 클릭으로 협업 리포트를 생성하세요.

1. W&B App에서 W&B 프로젝트 워크스페이스로 이동합니다.
2. 워크스페이스 오른쪽 상단에 있는 **Create report** 버튼을 클릭합니다.

![](/images/reports/create_a_report_button.png)

3. **Create report**라는 제목의 모달이 나타납니다. 리포트에 추가하고 싶은 차트와 패널을 선택하세요. (나중에 차트와 패널을 추가하거나 제거할 수 있습니다).
4. **Create report**를 클릭합니다.
5. 원하는 상태로 리포트를 편집합니다.
6. **Publish to project**를 클릭합니다.
7. 리포트를 공동 작업자와 공유하려면 **Share** 버튼을 클릭합니다.

W&B Python SDK를 사용하여 리포트를 대화형 및 프로그래밍 방식으로 생성하는 방법에 대한 자세한 정보는 [리포트 생성](./create-a-report.md) 페이지를 참조하세요.

## 시작 방법
유스케이스에 따라 밑의 자료를 살펴보고 W&B Reports를 시작하세요:

* [데모 동영상](https://www.youtube.com/watch?v=2xeJIv_K_eI)을 통해 W&B Reports의 개요를 확인하세요.
* [리포트 갤러리](./reports-gallery.md)에서 라이브 리포트 예시를 둘러보세요.
* [W&B Fully Connected](http://wandb.me/fc)에서 큐레이트된 리포트를 읽어보세요.