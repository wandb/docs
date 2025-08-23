---
title: Sweeps
description: W&B Sweeps 를 활용한 하이퍼파라미터 탐색과 모델 최적화
cascade:
- url: guides/sweeps/:filename
menu:
  default:
    identifier: ko-guides-models-sweeps-_index
    parent: w-b-models
url: guides/sweeps
weight: 2
---

{{< cta-button productLink="https://wandb.ai/stacey/deep-drive/workspace?workspace=user-lavanyashukla" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb" >}}

W&B Sweeps 를 사용하여 하이퍼파라미터 탐색을 자동화하고, 풍부하고 인터랙티브한 실험 추적을 시각화할 수 있습니다. Bayesian, 그리드 검색, 랜덤 등 인기 있는 탐색 방법 중에서 선택하여 하이퍼파라미터 공간을 탐색하세요. 하나 이상의 머신에서 스윕을 확장하고 병렬로 실행할 수 있습니다.

{{< img src="/images/sweeps/intro_what_it_is.png" alt="하이퍼파라미터 튜닝 인사이트" >}}

### 작동 방식
두 개의 [W&B CLI]({{< relref path="/ref/cli/" lang="ko" >}}) 코맨드를 사용하여 스윕을 생성하세요:

1. 스윕 초기화

```bash
wandb sweep --project <project-name> <path-to-config file>
```

2. 스윕 에이전트 시작

```bash
wandb agent <sweep-ID>
```

{{% alert %}}
위의 코드조각과 이 페이지에 연결된 colab 예제는 W&B CLI로 스윕을 초기화하고 생성하는 방법을 보여줍니다. 스윕 구성을 정의하고, 스윕을 초기화하며, 스윕을 시작하는 데 사용하는 W&B Python SDK 코맨드를 단계별로 안내하는 [Sweeps walkthrough]({{< relref path="./walkthrough.md" lang="ko" >}})를 참고하세요.
{{% /alert %}}

### 시작 방법

유스 케이스에 따라, 아래 리소스를 참고하여 W&B Sweeps 시작 방법을 알아보세요:

* 스윕 구성을 정의하고, 스윕을 초기화하며, 스윕을 시작하는 데 사용하는 W&B Python SDK 코맨드를 단계별로 안내하는 [sweeps walkthrough]({{< relref path="./walkthrough.md" lang="ko" >}})를 읽어보세요.
* 이 챕터를 탐색하여 다음과 같은 내용을 배울 수 있습니다:
  * [코드에 W&B 추가하기]({{< relref path="./add-w-and-b-to-your-code.md" lang="ko" >}})
  * [스윕 구성 정의하기]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ko" >}})
  * [스윕 초기화하기]({{< relref path="./initialize-sweeps.md" lang="ko" >}})
  * [스윕 에이전트 시작하기]({{< relref path="./start-sweep-agents.md" lang="ko" >}})
  * [스윕 결과 시각화하기]({{< relref path="./visualize-sweep-results.md" lang="ko" >}})
* [Sweep 실험 모음]({{< relref path="./useful-resources.md" lang="ko" >}})에서 W&B Sweeps 를 통한 하이퍼파라미터 최적화 예제를 확인하세요. 결과는 W&B Reports 에 저장됩니다.

단계별 영상이 필요하다면, [W&B Sweeps 로 하이퍼파라미터 쉽게 튜닝하기](https://www.youtube.com/watch?v=9zrmUIlScdY\&ab_channel=Weights%26Biases)를 참고하세요.