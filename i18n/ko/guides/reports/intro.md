---
description: Project management and collaboration tools for machine learning projects
slug: /guides/reports
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# 협업 리포트

<CTAButtons productLink="https://wandb.ai/stacey/deep-drive/reports/The-View-from-the-Driver-s-Seat--Vmlldzo1MTg5NQ?utm_source=fully_connected&utm_medium=blog&utm_campaign=view+from+the+drivers+seat" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb"/>

W&B 리포트를 사용하여 실행을 구성하고, 시각화를 내장하고 자동화하며, 발견한 내용을 설명하고, 협업자와 업데이트를 공유하세요. 리포트를 LaTeX zip 파일로 쉽게 내보내거나 PDF 파일로 변환할 수 있습니다.

다음 이미지는 학습 과정에서 W&B에 로그된 메트릭을 기반으로 생성된 리포트의 섹션을 보여줍니다.

![](/images/reports/safe-lite-benchmark-with-comments.png)

위 이미지가 나온 리포트를 [여기](https://wandb.ai/stacey/saferlife/reports/SafeLife-Benchmark-Experiments--Vmlldzo0NjE4MzM)에서 확인하세요.

## 작동 방식
몇 번의 클릭으로 협업 리포트를 생성하세요.

1. W&B 앱에서 W&B 프로젝트 워크스페이스로 이동하세요.
2. 워크스페이스의 오른쪽 상단에 있는 **리포트 생성** 버튼을 클릭하세요.

![](/images/reports/create_a_report_button.png)

3. **리포트 생성**이라는 제목의 모달이 나타납니다. 리포트에 추가하고 싶은 차트와 패널을 선택하세요. (나중에 차트와 패널을 추가하거나 제거할 수 있습니다).
4. **리포트 생성**을 클릭하세요.
5. 원하는 상태로 리포트를 편집하세요.
6. **프로젝트에 게시**를 클릭하세요.
7. **공유** 버튼을 클릭하여 협업자와 리포트를 공유하세요.

W&B Python SDK를 사용하여 리포트를 대화형 및 프로그래밍 방식으로 생성하는 방법에 대한 자세한 내용은 [리포트 생성](./create-a-report.md) 페이지를 참조하세요.

## 시작 방법
사용 사례에 따라 W&B 리포트를 시작하기 위한 다음 리소스를 탐색하세요:

* W&B 리포트의 개요를 얻기 위한 [비디오 데모](https://www.youtube.com/watch?v=2xeJIv_K_eI)를 확인하세요.
* 실시간 리포트 예시를 위한 [리포트 갤러리](./reports-gallery.md)를 탐색하세요.
* [W&B Fully Connected](http://wandb.me/fc)에서 큐레이션된 리포트를 읽어보세요.