---
title: Kubeflow 파이프라인(kfp)
description: W&B 를 Kubeflow 파이프라인과 통합하는 방법
menu:
  default:
    identifier: ko-guides-integrations-kubeflow-pipelines-kfp
    parent: integrations
weight: 170
---

[Kubeflow Pipelines (kfp)](https://www.kubeflow.org/docs/components/pipelines/overview/)는 Docker 컨테이너 기반으로 이식 가능하고 확장 가능한 머신러닝(ML) 워크플로우를 구축하고 배포할 수 있는 플랫폼입니다.

이 인테그레이션을 통해 사용자는 kfp의 파이썬 함수형 컴포넌트에 데코레이터를 적용하여 파라미터와 Artifacts를 W&B에 자동으로 로그할 수 있습니다.

이 기능은 `wandb==0.12.11` 버전부터 지원되며, `kfp<2.0.0` 환경이 필요합니다.

## 회원가입 및 API 키 생성

API 키는 사용자의 머신을 W&B에 인증할 때 사용됩니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
더 쉽게 진행하려면, [W&B 인증 페이지](https://wandb.ai/authorize)에서 즉시 API 키를 생성할 수 있습니다. 표시된 API 키를 복사하여 패스워드 매니저 등 안전한 곳에 저장하세요.
{{% /alert %}}

1. 우측 상단에서 사용자 프로필 아이콘을 클릭합니다.
1. **User Settings**를 선택한 뒤 아래로 내려가 **API Keys** 섹션을 찾습니다.
1. **Reveal**을 클릭하고, 표시된 API 키를 복사합니다. 키를 숨기려면 페이지를 새로고침하세요.

## `wandb` 라이브러리 설치 및 로그인

다음과 같이 로컬 환경에 `wandb`를 설치하고 로그인할 수 있습니다.

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 본인의 API 키 값으로 설정합니다.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` 라이브러리를 설치하고 로그인합니다.

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

## 컴포넌트에 데코레이터 추가하기

`@wandb_log` 데코레이터를 추가하고 평소처럼 컴포넌트를 생성하세요. 이렇게 하면 파이프라인 실행마다 입력/출력 파라미터와 Artifacts가 W&B에 자동 로그됩니다.

```python
from kfp import components
from wandb.integration.kfp import wandb_log

@wandb_log
def add(a: float, b: float) -> float:
    return a + b

add = components.create_component_from_func(add)
```

## 컨테이너에 환경 변수 전달하기

[환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 컨테이너에 직접 전달해야 할 수 있습니다. 양방향 링크(two-way linking)를 위해 `WANDB_KUBEFLOW_URL` 환경 변수를 Kubeflow Pipelines 인스턴스의 기본 URL로 지정해야 합니다. 예: `https://kubeflow.mysite.com`.

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

## 프로그래밍적으로 데이터에 엑세스하기

### Kubeflow Pipelines UI를 통한 엑세스

W&B 로그가 남겨진 Run을 Kubeflow Pipelines UI에서 클릭하세요.

* `Input/Output` 및 `ML Metadata` 탭에서 입력과 출력 정보가 보입니다.
* `Visualizations` 탭에서 W&B 웹 앱을 열 수 있습니다.

{{< img src="/images/integrations/kubeflow_app_pipelines_ui.png" alt="W&B in Kubeflow UI" >}}

### 웹 앱 UI를 통한 엑세스

웹 앱 UI를 이용하면, Kubeflow Pipelines의 `Visualizations` 탭과 같은 내용을 더 넓은 화면에서 확인할 수 있습니다. [웹 앱 UI 자세히 살펴보기]({{< relref path="/guides/models/app" lang="ko" >}}).

{{< img src="/images/integrations/kubeflow_pipelines.png" alt="Run details" >}}

{{< img src="/images/integrations/kubeflow_via_app.png" alt="Pipeline DAG" >}}

### Public API를 통한 엑세스(프로그래밍 방식)

* 프로그래밍적으로 접근하려면 [Public API 문서]({{< relref path="/ref/python/public-api/index.md" lang="ko" >}})를 참고하세요.

### Kubeflow Pipelines와 W&B 컨셉 대응

Kubeflow Pipelines의 개념이 W&B에서 어떻게 매핑되는지 아래 표를 참고하세요:

| Kubeflow Pipelines | W&B | W&B 내 위치 |
| ------------------ | --- | ----------- |
| Input Scalar | [`config`]({{< relref path="/guides/models/track/config" lang="ko" >}}) | [Overview 탭]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ko" >}}) |
| Output Scalar | [`summary`]({{< relref path="/guides/models/track/log" lang="ko" >}}) | [Overview 탭]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ko" >}}) |
| Input Artifact | Input Artifact | [Artifacts 탭]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ko" >}}) |
| Output Artifact | Output Artifact | [Artifacts 탭]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ko" >}}) |

## 세부 로그 남기기

보다 자세한 로그를 남기고 싶다면 컴포넌트 내부에서 `wandb.log` 및 `wandb.log_artifact`를 직접 호출할 수 있습니다.

### 명시적 `wandb.log_artifacts` 호출 예시

예를 들어 아래처럼 모델 트레이닝 과정 전체를 기록할 수 있습니다. `@wandb_log` 데코레이터로 입력/출력이 자동 추적되며, 훈련 과정에 대한 로그를 남기고 싶다면 아래와 같이 사용하면 됩니다:

```python
@wandb_log
def train_model(
    train_dataloader_path: components.InputPath("dataloader"),
    test_dataloader_path: components.InputPath("dataloader"),
    model_path: components.OutputPath("pytorch_model"),
):
    with wandb.init() as run:
        ...
        for epoch in epochs:
            for batch_idx, (data, target) in enumerate(train_dataloader):
                ...
                if batch_idx % log_interval == 0:
                    run.log(
                        {"epoch": epoch, "step": batch_idx * len(data), "loss": loss.item()}
                    )
            ...
            run.log_artifact(model_artifact)
```

### wandb 인테그레이션 암시적으로 사용하기

[지원 프레임워크 인테그레이션]({{< relref path="/guides/integrations/" lang="ko" >}})를 활용하면, 콜백을 직접 넘기는 방식도 가능합니다:

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
    ...  # 트레이닝 진행
```