---
title: W&B Quickstart
description: W&B 퀵스타트
menu:
  default:
    identifier: ko-guides-quickstart
    parent: guides
url: quickstart
weight: 2
---

W&B를 설치하고 몇 분 안에 기계 학습 Experiments 트래킹을 시작하세요.

## 가입 및 API 키 생성

API 키는 사용자의 컴퓨터를 W&B에 인증합니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
더욱 간소화된 접근 방식을 위해 [https://wandb.ai/authorize](https://wandb.ai/authorize)로 직접 이동하여 API 키를 생성할 수 있습니다. 표시된 API 키를 복사하여 비밀번호 관리자와 같은 안전한 위치에 저장하십시오.
{{% /alert %}}

1. 오른쪽 상단 모서리에 있는 사용자 프로필 아이콘을 클릭합니다.
2. **User Settings**를 선택한 다음 **API Keys** 섹션으로 스크롤합니다.
3. **Reveal**을 클릭합니다. 표시된 API 키를 복사합니다. API 키를 숨기려면 페이지를 새로 고치세요.

## `wandb` 라이브러리 설치 및 로그인

`wandb` 라이브러리를 로컬에 설치하고 로그인하려면:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 API 키로 설정합니다.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` 라이브러리를 설치하고 로그인합니다.

    ```shell
    pip install wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## Run 시작 및 하이퍼파라미터 트래킹

[`wandb.init()`]({{< relref path="/ref/python/run.md" lang="ko" >}})을 사용하여 Python 스크립트 또는 노트북에서 W&B Run 오브젝트를 초기화하고 하이퍼파라미터 이름과 값의 키-값 쌍으로 이루어진 사전을 `config` 파라미터에 전달합니다.

```python
run = wandb.init(
    # 이 run이 기록될 프로젝트 설정
    project="my-awesome-project",
    # 하이퍼파라미터 및 run 메타데이터 트래킹
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```

[Run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})은 W&B의 기본 구성 요소입니다. 이를 사용하여 [메트릭 트래킹]({{< relref path="/guides/models/track/" lang="ko" >}}), [로그 생성]({{< relref path="/guides/core/artifacts/" lang="ko" >}}) 등을 자주 수행하게 됩니다.

## 모두 합치기

모든 것을 종합하면, 트레이닝 스크립트는 다음 코드 예제와 유사할 수 있습니다. 강조 표시된 코드는 W&B 관련 코드를 보여줍니다. 기계 학습 트레이닝을 모방하는 코드를 추가했습니다.

```python
# train.py
import wandb
import random  # 데모 스크립트용

# highlight-next-line
wandb.login()

epochs = 10
lr = 0.01

# highlight-start
run = wandb.init(
    # 이 run이 기록될 프로젝트 설정
    project="my-awesome-project",
    # 하이퍼파라미터 및 run 메타데이터 트래킹
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
)
# highlight-end

offset = random.random() / 5
print(f"lr: {lr}")

# 트레이닝 run 시뮬레이션
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    # highlight-next-line
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()
```

이것으로 끝입니다. [https://wandb.ai/home](https://wandb.ai/home)의 W&B 앱으로 이동하여 W&B로 기록한 메트릭 (정확도 및 손실)이 각 트레이닝 단계에서 어떻게 개선되었는지 확인하십시오.

{{< img src="/images/quickstart/quickstart_image.png" alt="위 스크립트를 실행할 때마다 추적된 손실 및 정확도를 보여줍니다." >}}

위의 이미지 (클릭하여 확대)는 스크립트를 실행할 때마다 추적된 손실 및 정확도를 보여줍니다. 생성된 각 run 오브젝트는 **Runs** 열 내에 표시됩니다. 각 run 이름은 임의로 생성됩니다.

## 다음 단계는 무엇일까요?

W&B 에코시스템의 나머지 부분을 탐색해보세요.

1. [W&B Integrations]({{< relref path="guides/integrations/" lang="ko" >}})를 확인하여 PyTorch와 같은 ML framework, Hugging Face와 같은 ML library 또는 SageMaker와 같은 ML service와 W&B를 통합하는 방법을 알아보세요.
2. [W&B Reports]({{< relref path="/guides/core/reports/" lang="ko" >}})를 사용하여 Runs를 구성하고, 시각화를 포함 및 자동화하고, 발견한 내용을 설명하고, 협업자와 업데이트를 공유하세요.
3. [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})를 생성하여 machine learning 파이프라인의 각 단계를 통해 데이터셋, Models, 종속성 및 결과를 추적하세요.
4. [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}})를 사용하여 하이퍼파라미터 검색을 자동화하고 가능한 Models의 공간을 탐색해보세요.
5. [중앙 대시보드]({{< relref path="/guides/core/tables/" lang="ko" >}})에서 데이터셋을 이해하고, Model 예측을 시각화하고, 통찰력을 공유하세요.
6. W&B AI Academy로 이동하여 실습 [코스](https://wandb.me/courses)에서 LLM, MLOps 및 W&B Models에 대해 알아보세요.
