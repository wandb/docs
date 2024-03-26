---
description: Hyperparameter search and model optimization with W&B Sweeps
slug: /guides/sweeps
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# 하이퍼파라미터 튜닝하기

<CTAButtons productLink="https://wandb.ai/stacey/deep-drive/workspace?workspace=user-lavanyashukla" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb"/>

<head>
  <title>Sweeps를 사용한 하이퍼파라미터 튜닝</title>
</head>

W&B Sweeps를 사용해 하이퍼파라미터 검색을 자동화하고 풍부한 실험 추적을 인터랙티브하게 시각화하세요. 베이지안, 그리드 검색, 랜덤 검색 등 널리 사용되는 검색 방법 중에서 선택하여 하이퍼파라미터 공간을 검색할 수 있습니다. 한 대 이상의 머신에서 스윕을 확장하고 병렬화할 수도 있습니다.

![인터랙티브한 대시보드를 통해 대규모 하이퍼파라미터 튜닝 실험에서 인사이트를 얻으세요.](/images/sweeps/intro_what_it_is.png)

### 작동 방식
두 개의 [W&B CLI](../../ref/cli/README.md) 커맨드를 사용하여 스윕을 생성하세요:


1. 스윕 초기화

```bash
wandb sweep --project <propject-name> <path-to-config file>
```

2. 스윕 에이전트 시작

```bash
wandb agent <sweep-ID>
```

:::tip
앞서 언급된 코드조각과 이 페이지에 연결된 colab 노트북은 W&B CLI를 사용하여 스윕을 초기화하고 생성하는 방법을 보여줍니다. 스윕 구성을 정의하고, 스윕을 초기화하고, 스윕을 시작하기 위해 사용할 W&B Python SDK 커맨드의 단계별 개요는 Sweeps [walkthrough](./walkthrough.md)를 참조하세요.
:::

### 시작 방법

유스 케이스에 따라 밑의 자료를 살펴보고 W&B Sweeps를 시작하세요:

* W&B Sweeps를 처음 사용하는 경우 [Sweeps Colab 노트북](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb)을 통해 시작하는 것이 좋습니다.
* 스윕 구성을 정의하고, 스윕을 초기화하고, 스윕을 시작하기 위해 사용할 W&B Python SDK 커맨드의 단계별 개요는 [Sweeps walkthrough](./walkthrough.md)를 참조하세요.
* 이 챕터에서 다음과 같은 방법을 배워보세요:
  * [코드에 W&B 추가하기](./add-w-and-b-to-your-code.md)
  * [스윕 구성 정의하기](./define-sweep-configuration.md)
  * [스윕 초기화하기](./initialize-sweeps.md)
  * [스윕 에이전트 시작하기](./start-sweep-agents.md)
  * [스윕 결과 시각화하기](./visualize-sweep-results.md)
* W&B Sweeps로 하이퍼파라미터 최적화를 탐색하는 [선별된 Sweeps 실험 목록](./useful-resources.md)을 살펴보세요. 결과는 W&B Reports에 저장됩니다.

단계별 비디오 가이드는 [W&B Sweeps로 쉽게 하이퍼파라미터 튜닝하기](https://www.youtube.com/watch?v=9zrmUIlScdY\&ab\_channel=Weights%26Biases)에서 보실 수 있습니다.