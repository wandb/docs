---
title: W&B 퀵스타트
description: W&B 퀵스타트
menu:
  default:
    identifier: ko-guides-quickstart
    parent: guides
url: quickstart
weight: 2
---

W&B를 설치하여 어떤 규모의 기계학습 실험도 추적, 시각화, 관리하세요.

{{% alert %}}
W&B Weave에 대한 정보를 찾고 계신가요? [Weave Python SDK 퀵스타트](https://weave-docs.wandb.ai/quickstart) 또는 [Weave TypeScript SDK 퀵스타트](https://weave-docs.wandb.ai/reference/generated_typescript_docs/intro-notebook)를 참고하세요.
{{% /alert %}}

## 회원가입 및 API 키 생성

W&B에 기기를 인증하려면, 사용자 프로필 또는 [wandb.ai/authorize](https://wandb.ai/authorize)에서 API 키를 생성하세요. API 키를 복사해 안전하게 보관하시기 바랍니다.

## `wandb` 라이브러리 설치 및 로그인

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 설정하세요.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` 라이브러리를 설치하고 로그인하세요.

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

## run을 시작하고 하이퍼파라미터 추적하기

Python 스크립트나 노트북에서 [`wandb.init()`]({{< relref path="/ref/python/sdk/classes/run.md" lang="ko" >}})로 W&B run 오브젝트를 초기화하세요. 하이퍼파라미터 이름과 값을 지정하려면 `config` 파라미터에 사전(dictionary)을 사용하세요.

```python
run = wandb.init(
    project="my-awesome-project",  # 프로젝트 지정
    config={                        # 하이퍼파라미터 및 메타데이터 추적
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```

[run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})은 W&B의 핵심 요소로, [메트릭 추적]({{< relref path="/guides/models/track/" lang="ko" >}}), [로그 생성]({{< relref path="/guides/models/track/log/" lang="ko" >}}) 등에 활용됩니다.

## 구성 요소 통합하기

이 예시 트레이닝 스크립트는 W&B에 시뮬레이션된 accuracy와 loss 메트릭을 로그합니다:

```python
import wandb
import random

wandb.login()

# run이 기록될 프로젝트
project = "my-awesome-project"

# 하이퍼파라미터가 담긴 사전
config = {
    'epochs' : 10,
    'lr' : 0.01
}

with wandb.init(project=project, config=config) as run:
    offset = random.random() / 5
    print(f"lr: {config['lr']}")
    
    # 트레이닝 run 시뮬레이션
    for epoch in range(2, config['epochs']):
        acc = 1 - 2**-config['epochs'] - random.random() / config['epochs'] - offset
        loss = 2**-config['epochs'] + random.random() / config['epochs'] + offset
        print(f"epoch={config['epochs']}, accuracy={acc}, loss={loss}")
        run.log({"accuracy": acc, "loss": loss})
```

[wandb.ai/home](https://wandb.ai/home)에서 accuracy, loss 등 기록된 메트릭과 트레이닝 단계별 변화를 확인할 수 있습니다. 아래 이미지는 각 run에서 추적된 loss와 accuracy를 보여줍니다. 각각의 run 오브젝트는 **Runs** 컬럼에 생성된 이름으로 표시됩니다.

{{< img src="/images/quickstart/quickstart_image.png" alt="각 run에서 추적된 loss 및 accuracy를 보여줍니다." >}}

## 다음 단계

W&B 에코시스템의 다양한 기능을 경험해보세요:

1. PyTorch 같은 프레임워크, Hugging Face 같은 라이브러리, SageMaker와 같은 서비스를 W&B와 함께 사용하는 [W&B 인테그레이션 튜토리얼]({{< relref path="guides/integrations/" lang="ko" >}})을 읽어보세요.
2. [W&B Reports]({{< relref path="/guides/core/reports/" lang="ko" >}})를 활용해 run을 정리하고, 시각화를 자동화하고, 발견한 내용을 요약하여 협업자와 공유할 수 있습니다.
3. [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})로 데이터셋, 모델, 의존성, 그리고 기계학습 파이프라인 전반의 결과를 추적해보세요.
4. [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}})로 하이퍼파라미터 탐색을 자동화하고 모델을 최적화하세요.
5. [중앙 대시보드]({{< relref path="/guides/models/tables/" lang="ko" >}})에서 run을 분석하고, 모델 예측값을 시각화하며, 인사이트를 공유하세요.
6. [W&B AI Academy](https://wandb.ai/site/courses/)에서 LLM, MLOps, W&B Models를 실습 중심 코스를 통해 배울 수 있습니다.
7. [weave-docs.wandb.ai](https://weave-docs.wandb.ai/)에서 Weave를 활용해 LLM 기반 애플리케이션을 추적, 실험, 평가, 배포 및 개선하는 방법을 알아보세요.