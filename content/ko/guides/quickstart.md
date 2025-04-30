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

W&B를 설치하여 모든 규모의 기계 학습 Experiments를 추적, 시각화 및 관리하세요.

## 가입하고 API 키 생성하기

W&B로 사용자의 머신을 인증하려면 사용자 프로필 또는 [wandb.ai/authorize](https://wandb.ai/authorize)에서 API 키를 생성하세요. API 키를 복사하여 안전하게 보관하세요.

## `wandb` 라이브러리 설치 및 로그인

{{< tabpane text=true >}}
{{% tab header="커맨드라인" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 설정합니다.

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

{{% tab header="Python 노트북" value="notebook" %}}

```notebook
!pip install wandb
import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## run 시작 및 하이퍼파라미터 추적

Python 스크립트 또는 노트북에서 [`wandb.init()`]({{< relref path="/ref/python/run.md" lang="ko" >}})으로 W&B run 오브젝트를 초기화합니다. `config` 파라미터에 사전을 사용하여 하이퍼파라미터 이름과 값을 지정합니다.

```python
run = wandb.init(
    project="my-awesome-project",  # 프로젝트 지정
    config={                        # 하이퍼파라미터 및 메타데이터 추적
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```

[Run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})은 W&B의 핵심 요소로, [메트릭 추적]({{< relref path="/guides/models/track/" lang="ko" >}}), [로그 생성]({{< relref path="/guides/models/track/log/" lang="ko" >}}) 등에 사용됩니다.

## 컴포넌트 조립

이 모의 트레이닝 스크립트는 시뮬레이션된 정확도 및 손실 메트릭을 W&B에 기록합니다.

```python
# train.py
import wandb
import random

wandb.login()

epochs = 10
lr = 0.01

run = wandb.init(
    project="my-awesome-project",    # 프로젝트 지정
    config={                         # 하이퍼파라미터 및 메타데이터 추적
        "learning_rate": lr,
        "epochs": epochs,
    },
)

offset = random.random() / 5
print(f"lr: {lr}")

# 트레이닝 run 시뮬레이션
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()
```

[wandb.ai/home](https://wandb.ai/home)을 방문하여 정확도 및 손실과 같이 기록된 메트릭과 각 트레이닝 단계에서 어떻게 변경되었는지 확인하세요. 다음 이미지는 각 run에서 추적된 손실 및 정확도를 보여줍니다. 각 run 오브젝트는 생성된 이름과 함께 **Runs** 열에 나타납니다.

{{< img src="/images/quickstart/quickstart_image.png" alt="Shows loss and accuracy tracked from each run." >}}

## 다음 단계

W&B ecosystem의 더 많은 기능을 탐색해 보세요.

1. PyTorch와 같은 프레임워크, Hugging Face와 같은 라이브러리, SageMaker와 같은 서비스와 W&B를 결합하는 [W&B Integration 튜토리얼]({{< relref path="guides/integrations/" lang="ko" >}})을 읽어보세요.
2. [W&B Reports]({{< relref path="/guides/core/reports/" lang="ko" >}})를 사용하여 runs을 구성하고, 시각화를 자동화하고, 발견한 내용을 요약하고, 협업자와 업데이트를 공유합니다.
3. [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})를 생성하여 기계 학습 파이프라인 전체에서 데이터셋, Models, 종속성 및 결과를 추적합니다.
4. [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}})로 하이퍼파라미터 검색을 자동화하고 Models을 최적화합니다.
5. [중앙 대시보드]({{< relref path="/guides/models/tables/" lang="ko" >}})에서 runs을 분석하고, 모델 예측값을 시각화하고, 인사이트를 공유합니다.
6. 핸즈온 코스를 통해 LLM, MLOps 및 W&B Models에 대해 배우려면 [W&B AI Academy](https://wandb.ai/site/courses/)를 방문하세요.
