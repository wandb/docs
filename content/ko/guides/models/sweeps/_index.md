---
title: Sweeps
description: W&B Sweeps를 사용한 하이퍼파라미터 검색 및 모델 최적화
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

W&B Sweeps 를 사용하여 하이퍼파라미터 검색을 자동화하고 풍부하고 인터랙티브한 experiment 추적을 시각화하세요. Bayesian, 그리드 검색 및 random과 같은 인기 있는 검색 방법 중에서 선택하여 하이퍼파라미터 공간을 검색합니다. 하나 이상의 시스템에서 스윕을 확장하고 병렬화합니다.

{{< img src="/images/sweeps/intro_what_it_is.png" alt="인터랙티브한 대시보드를 통해 대규모 하이퍼파라미터 튜닝 Experiments에서 통찰력을 얻으세요." >}}

### 작동 방식
두 개의 [W&B CLI]({{< relref path="/ref/cli/" lang="ko" >}}) 명령으로 스윕을 생성합니다.

1. 스윕 초기화

```bash
wandb sweep --project <propject-name> <path-to-config file>
```

2. 스윕 에이전트 시작

```bash
wandb agent <sweep-ID>
```

{{% alert %}}
위의 코드 조각과 이 페이지에 링크된 colab은 W&B CLI로 스윕을 초기화하고 생성하는 방법을 보여줍니다. 스윕 구성을 정의하고 스윕을 초기화하고 스윕을 시작하는 데 사용할 W&B Python SDK 명령에 대한 단계별 개요는 Sweeps [워크스루]({{< relref path="./walkthrough.md" lang="ko" >}})를 참조하세요.
{{% /alert %}}

### 시작 방법

유스 케이스에 따라 다음 리소스를 탐색하여 W&B Sweeps를 시작하세요.

* 스윕 구성을 정의하고 스윕을 초기화하고 스윕을 시작하는 데 사용할 W&B Python SDK 명령에 대한 단계별 개요는 [스윕 워크스루]({{< relref path="./walkthrough.md" lang="ko" >}})를 읽어보세요.
* 다음 방법을 배우려면 이 챕터를 살펴보세요.
  * [W&B를 코드에 추가]({{< relref path="./add-w-and-b-to-your-code.md" lang="ko" >}})
  * [스윕 구성 정의]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ko" >}})
  * [스윕 초기화]({{< relref path="./initialize-sweeps.md" lang="ko" >}})
  * [스윕 에이전트 시작]({{< relref path="./start-sweep-agents.md" lang="ko" >}})
  * [스윕 결과 시각화]({{< relref path="./visualize-sweep-results.md" lang="ko" >}})
* W&B Sweeps를 사용한 하이퍼파라미터 최적화를 탐색하는 [선별된 스윕 Experiments 목록]({{< relref path="./useful-resources.md" lang="ko" >}})을 탐색합니다. 결과는 W&B Reports에 저장됩니다.

단계별 비디오는 [W&B Sweeps로 하이퍼파라미터를 쉽게 튜닝하세요](https://www.youtube.com/watch?v=9zrmUIlScdY\&ab_channel=Weights%26Biases)를 참조하세요.
