---
description: Hyperparameter search and model optimization with W&B Sweeps
slug: /guides/sweeps
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# 하이퍼파라미터 조정

<CTAButtons productLink="https://wandb.ai/stacey/deep-drive/workspace?workspace=user-lavanyashukla" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb"/>

<head>
  <title>스윕을 이용한 하이퍼파라미터 조정</title>
</head>

하이퍼파라미터 검색을 자동화하고 풍부하고 상호작용적인 실험 추적을 시각화하기 위해 W&B 스윕을 사용하세요. Bayesian, 그리드 검색, 무작위 검색과 같은 인기 있는 검색 방법 중에서 선택하여 하이퍼파라미터 공간을 검색합니다. 한 대 이상의 기계에서 스윕을 확장하고 병렬 처리합니다.

![대화형 대시보드를 통해 대규모 하이퍼파라미터 조정 실험에서 통찰을 얻습니다.](/images/sweeps/intro_what_it_is.png)

### 작동 방식
[W&B CLI](../../ref/cli/README.md) 명령어 두 가지로 스윕을 생성하세요:


1. 스윕 초기화

```bash
wandb sweep --project <프로젝트-이름> <구성 파일 경로>
```

2. 스윕 에이전트 시작

```bash
wandb agent <스윕-ID>
```

:::tip
이전 코드 조각과 이 페이지에 연결된 colab은 W&B CLI로 스윕을 초기화하고 생성하는 방법을 보여줍니다. 스윕 구성을 정의하고 스윕을 초기화하며 스윕을 시작하기 위해 사용할 W&B Python SDK 명령어에 대한 단계별 개요는 스윕 [가이드](./walkthrough.md)에서 확인하세요.
:::

### 시작 방법

사용 사례에 따라 W&B 스윕을 시작하기 위해 다음 자료를 탐색하세요:

* W&B 스윕을 처음 사용하는 경우, [Sweeps Colab 노트북](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb)을 통해 시작하는 것이 좋습니다.
* 스윕 구성을 정의하고 스윕을 초기화하며 스윕을 시작하기 위해 사용할 W&B Python SDK 명령어에 대한 단계별 개요는 [스윕 가이드](./walkthrough.md)를 읽어보세요.
* 이 장을 탐색하여 다음 방법을 배우세요:
  * [코드에 W&B 추가](./add-w-and-b-to-your-code.md)
  * [스윕 구성 정의](./define-sweep-configuration.md)
  * [스윕 초기화](./initialize-sweeps.md)
  * [스윕 에이전트 시작](./start-sweep-agents.md)
  * [스윕 결과 시각화](./visualize-sweep-results.md)
* W&B 스윕으로 하이퍼파라미터 최적화를 탐색하는 [선택된 스윕 실험 목록](./useful-resources.md)을 탐색하세요. 결과는 W&B 리포트에 저장됩니다.

단계별 비디오를 보려면: [W&B 스윕으로 쉽게 하이퍼파라미터 조정](https://www.youtube.com/watch?v=9zrmUIlScdY\&ab\_channel=Weights%26Biases).