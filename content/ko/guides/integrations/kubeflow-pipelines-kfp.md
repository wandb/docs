---
title: Kubeflow Pipelines (kfp)
description: W&B를 Kubeflow 파이프라인과 통합하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-kubeflow-pipelines-kfp
    parent: integrations
weight: 170
---

[Kubeflow Pipelines (kfp) ](https://www.kubeflow.org/docs/components/pipelines/overview/)는 Docker 컨테이너를 기반으로 확장 가능한 휴대용 기계학습 (ML) 워크플로우를 구축하고 배포하기 위한 플랫폼입니다.

이 통합을 통해 사용자는 kfp python 기능 컴포넌트에 데코레이터를 적용하여 파라미터와 Artifacts를 자동으로 W&B에 로그할 수 있습니다.

이 기능은 `wandb==0.12.11`에서 활성화되었으며 `kfp<2.0.0`이 필요합니다.

## 가입하고 API 키를 생성하세요

API 키는 사용자의 머신을 W&B에 인증합니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
보다 간소화된 접근 방식을 위해 [https://wandb.ai/authorize](https://wandb.ai/authorize)로 직접 이동하여 API 키를 생성할 수 있습니다. 표시된 API 키를 복사하여 비밀번호 관리자와 같은 안전한 위치에 저장하세요.
{{% /alert %}}

1. 오른쪽 상단 모서리에 있는 사용자 프로필 아이콘을 클릭합니다.
2. **User Settings**를 선택한 다음 **API Keys** 섹션으로 스크롤합니다.
3. **Reveal**을 클릭합니다. 표시된 API 키를 복사합니다. API 키를 숨기려면 페이지를 새로 고칩니다.

## `wandb` 라이브러리를 설치하고 로그인합니다

`wandb` 라이브러리를 로컬에 설치하고 로그인하려면 다음을 수행하세요.

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 사용자의 API 키로 설정합니다.

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

## 컴포넌트 데코레이트하기

`@wandb_log` 데코레이터를 추가하고 평소처럼 컴포넌트를 생성합니다. 이렇게 하면 파이프라인을 실행할 때마다 입력/출력 파라미터와 Artifacts가 자동으로 W&B에 기록됩니다.

```python
from kfp import components
from wandb.integration.kfp import wandb_log


@wandb_log
def add(a: float, b: float) -> float:
    return a + b


add = components.create_component_from_func(add)
```

## 컨테이너에 환경 변수 전달하기

컨테이너에 [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 명시적으로 전달해야 할 수 있습니다. 양방향 연결을 위해 환경 변수 `WANDB_KUBEFLOW_URL`을 Kubeflow Pipelines 인스턴스의 기본 URL로 설정해야 합니다. 예를 들어 `https://kubeflow.mysite.com`과 같습니다.

```python
import os
from kubernetes.client.models import V1EnvVar


def add_wandb_env_variables(op):
    env = {
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        "WANDB_BASE_URL": os.getenv("WANDB_BASE_URL"),
    }

    for name, value in env.items():
        op = op.add_env_variable(V1EnvVar(name, value))
    return op


@dsl.pipeline(name="example-pipeline")
def example_pipeline(param1: str, param2: int):
    conf = dsl.get_pipeline_conf()
    conf.add_op_transformer(add_wandb_env_variables)
```

## 프로그래밍 방식으로 데이터에 엑세스하기

### Kubeflow Pipelines UI를 통해

W&B에 기록된 Kubeflow Pipelines UI에서 Run을 클릭합니다.

* `Input/Output` 및 `ML Metadata` 탭에서 입력 및 출력에 대한 세부 정보를 찾습니다.
* `Visualizations` 탭에서 W&B 웹 앱을 봅니다.

{{< img src="/images/integrations/kubeflow_app_pipelines_ui.png" alt="Kubeflow UI에서 W&B 보기" >}}

### 웹 앱 UI를 통해

웹 앱 UI는 Kubeflow Pipelines의 `Visualizations` 탭과 동일한 내용을 가지고 있지만 더 넓은 공간을 제공합니다. [웹 앱 UI에 대해 자세히 알아보려면 여기를 참조하세요]({{< relref path="/guides/models/app" lang="ko" >}}).

{{< img src="/images/integrations/kubeflow_pipelines.png" alt="특정 run에 대한 세부 정보 보기 (그리고 Kubeflow UI로 다시 연결)" >}}

{{< img src="/images/integrations/kubeflow_via_app.png" alt="파이프라인의 각 단계에서 입력 및 출력의 전체 DAG를 확인하세요" >}}

### Public API를 통해 (프로그래밍 방식 엑세스)

* 프로그래밍 방식 엑세스의 경우 [Public API를 참조하세요]({{< relref path="/ref/python/public-api" lang="ko" >}}).

### Kubeflow Pipelines에서 W&B로의 개념 매핑

다음은 Kubeflow Pipelines 개념과 W&B의 매핑입니다.

| Kubeflow Pipelines | W&B | W&B의 위치 |
| ------------------ | --- | --------------- |
| Input Scalar | [`config`]({{< relref path="/guides/models/track/config" lang="ko" >}}) | [Overview 탭]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ko" >}}) |
| Output Scalar | [`summary`]({{< relref path="/guides/models/track/log" lang="ko" >}}) | [Overview 탭]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ko" >}}) |
| Input Artifact | Input Artifact | [Artifacts 탭]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ko" >}}) |
| Output Artifact | Output Artifact | [Artifacts 탭]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ko" >}}) |

## 세분화된 로깅

로깅을 더 세밀하게 제어하려면 컴포넌트에 `wandb.log` 및 `wandb.log_artifact` 호출을 추가할 수 있습니다.

### 명시적인 `wandb.log_artifacts` 호출 사용

아래 예에서는 모델을 트레이닝하고 있습니다. `@wandb_log` 데코레이터는 관련 입력 및 출력을 자동으로 추적합니다. 트레이닝 프로세스를 기록하려면 다음과 같이 명시적으로 로깅을 추가할 수 있습니다.

```python
@wandb_log
def train_model(
    train_dataloader_path: components.InputPath("dataloader"),
    test_dataloader_path: components.InputPath("dataloader"),
    model_path: components.OutputPath("pytorch_model"),
):
    ...
    for epoch in epochs:
        for batch_idx, (data, target) in enumerate(train_dataloader):
            ...
            if batch_idx % log_interval == 0:
                wandb.log(
                    {"epoch": epoch, "step": batch_idx * len(data), "loss": loss.item()}
                )
        ...
        wandb.log_artifact(model_artifact)
```

### 암시적 wandb 인테그레이션 사용

[지원하는 프레임워크 인테그레이션]({{< relref path="/guides/integrations/" lang="ko" >}})을 사용하는 경우 콜백을 직접 전달할 수도 있습니다.

```python
@wandb_log
def train_model(
    train_dataloader_path: components.InputPath("dataloader"),
    test_dataloader_path: components.InputPath("dataloader"),
    model_path: components.OutputPath("pytorch_model"),
):
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning import Trainer

    trainer = Trainer(logger=WandbLogger())
    ...  # do training
```
