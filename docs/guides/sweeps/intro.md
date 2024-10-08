---
title: Sweeps
description: W&B Sweeps로 하이퍼파라미터 검색 및 모델 최적화
slug: /guides/sweeps
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons productLink="https://wandb.ai/stacey/deep-drive/workspace?workspace=user-lavanyashukla" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb"/>

W&B Sweeps를 사용하여 하이퍼파라미터 검색을 자동화하고 풍부하고 상호작용적인 실험 추적을 시각화하세요. Bayesian, 그리드 검색, 임의 검색 같은 인기 있는 검색 방법을 선택하여 하이퍼파라미터 공간을 탐색하세요. 하나 이상의 기계에서 스윕을 확장 및 병렬화합니다.

![대화형 대시보드를 통해 대규모 하이퍼파라미터 튜닝 실험에서 인사이트를 도출하세요.](/images/sweeps/intro_what_it_is.png)

### 작동 방식

두 개의 [W&B CLI](../../ref/cli/README.md) 코맨드를 사용하여 스윕을 생성하세요:

1. 스윕 초기화

```bash
wandb sweep --project <propject-name> <path-to-config file>
```

2. 스윕 에이전트 시작

```bash
wandb agent <sweep-ID>
```

:::tip
위의 코드조각과 이 페이지에 연결된 colab은 W&B CLI를 사용하여 스윕을 초기화하고 생성하는 방법을 보여줍니다. [Walkthrough](./walkthrough.md)를 참조하여 스윕 구성을 정의하고, 스윕을 초기화하고, 시작하는 단계별 W&B Python SDK 명령을 확인하세요.
:::

### 시작 방법

유스 케이스에 따라, W&B Sweeps를 시작하기 위해 다음 자료를 탐색하세요:

* 스윕 구성을 정의하고, 스윕을 초기화하고, 시작하는 단계별 W&B Python SDK 명령을 확인하려면 [sweeps walkthrough](./walkthrough.md)를 읽어보세요.
* 이 챕터를 탐색하여 다음을 배우세요:
  * [코드에 W&B 추가](./add-w-and-b-to-your-code.md)
  * [스윕 구성 정의](./define-sweep-configuration.md)
  * [스윕 초기화](./initialize-sweeps.md)
  * [스윕 에이전트 시작](./start-sweep-agents.md)
  * [스윕 결과 시각화](./visualize-sweep-results.md)
* W&B Sweeps로 하이퍼파라미터 최적화를 탐색하는 [큐레이트된 Sweep 실험 목록](./useful-resources.md)을 탐색하세요. 결과는 W&B Reports에 저장됩니다.

단계별 동영상을 보려면: [Tune Hyperparameters Easily with W&B Sweeps](https://www.youtube.com/watch?v=9zrmUIlScdY\&ab_channel=Weights%26Biases)를 참고하세요.