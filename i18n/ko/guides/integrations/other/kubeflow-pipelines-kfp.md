---
description: How to integrate W&B with Kubeflow Pipelines.
slug: /guides/integrations/kubeflow-pipelines-kfp
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Kubeflow Pipelines (kfp)

## 개요

[Kubeflow Pipelines (kfp)](https://www.kubeflow.org/docs/components/pipelines/introduction/)는 Docker 컨테이너를 기반으로 한 이식 가능하고 확장 가능한 기계학습(ML) 워크플로우를 구축 및 배포하기 위한 플랫폼입니다.

이 인테그레이션은 사용자가 kfp 파이썬 함수 컴포넌트에 데코레이터를 적용하여 파라미터와 아티팩트를 W&B에 자동으로 로그할 수 있게 합니다.

이 기능은 `wandb==0.12.11`에서 활성화되었으며 `kfp<2.0.0`이 필요합니다.

## 퀵스타트

### W&B 설치 및 로그인

<Tabs
  defaultValue="notebook"
  values={[
    {label: '노트북', value: 'notebook'},
    {label: '커맨드라인', value: 'cli'},
  ]}>
  <TabItem value="notebook">

```python
!pip install kfp wandb

import wandb
wandb.login()
```

  </TabItem>
  <TabItem value="cli">

```
pip install kfp wandb
wandb login
```

  </TabItem>
</Tabs>

### 컴포넌트 꾸미기

`@wandb_log` 데코레이터를 추가하고 평소와 같이 컴포넌트를 생성하세요. 이렇게 하면 파이프라인을 실행할 때마다 입력/출력 파라미터와 아티팩트를 W&B에 자동으로 로그합니다.

```python
from kfp import components
from wandb.integration.kfp import wandb_log

@wandb_log
def add(a: float, b: float) -> float:
    return a + b

add = components.create_component_from_func(add)
```

### 컨테이너에 환경 변수 전달하기

컨테이너에 [WANDB 환경 변수](../../track/environment-variables.md)를 명시적으로 전달해야 할 수도 있습니다. 양방향 연결을 위해, Kubeflow Pipelines 인스턴스의 기본 URL을 `WANDB_KUBEFLOW_URL` 환경 변수로 설정해야 합니다(예: https://kubeflow.mysite.com)

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
def example_pipeline(...):
    conf = dsl.get_pipeline_conf()
    conf.add_op_transformer(add_wandb_env_variables)
    ...
```

## 내 데이터는 어디에 있나요? 프로그래매틱하게 엑세스할 수 있나요?

### Kubeflow Pipelines UI를 통해

Kubeflow Pipelines UI에서 W&B로 로그된 모든 Run을 클릭하세요.

* 입력 및 출력은 `Input/Output` 및 `ML Metadata` 탭에서 추적됩니다
* `시각화` 탭에서 W&B 웹 앱을 볼 수도 있습니다.

![Kubeflow UI에서 W&B 보기](/images/integrations/kubeflow_app_pipelines_ui.png)

### 웹 앱 UI를 통해

웹 앱 UI는 Kubeflow Pipelines의 `시각화` 탭과 동일한 내용을 가지고 있지만 더 많은 공간이 있습니다! [웹 앱 UI에 대해 더 알아보기](https://docs.wandb.ai/ref/app).

![특정 런에 대한 세부 정보 보기(및 Kubeflow UI로 다시 링크)](/images/integrations/kubeflow_pipelines.png)

![파이프라인의 각 단계에서 입력 및 출력의 전체 DAG 보기](/images/integrations/kubeflow_via_app.png)

### 프로그래매틱 엑세스를 위한 공개 API (Public API)

* 프로그래매틱 엑세스를 위해, [우리의 공개 API를 참조하세요](https://docs.wandb.ai/ref/python/public-api).

### Kubeflow Pipelines에서 W&B로의 개념 매핑

다음은 Kubeflow Pipelines 개념과 W&B의 매핑입니다.

| Kubeflow Pipelines | W&B                                                      | W&B에서의 위치                                                                                  |
| ------------------ | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| 입력 스칼라          | ``[`config`](https://docs.wandb.ai/guides/track/config)`` | [Overview 탭](https://docs.wandb.ai/ref/app/pages/run-page#overview-tab)                         |
| 출력 스칼라          | ``[`summary`](https://docs.wandb.ai/guides/track/log)``   | [Overview 탭](https://docs.wandb.ai/ref/app/pages/run-page#overview-tab)                         |
| 입력 아티팩트       | 입력 아티팩트                                           | [Artifacts 탭](https://docs.wandb.ai/ref/app/pages/run-page#artifacts-tab)                       |
| 출력 아티팩트       | 출력 아티팩트                                           | [Artifacts 탭](https://docs.wandb.ai/ref/app/pages/run-page#artifacts-tab) |

## 세밀한 로깅

로깅을 더 세밀하게 제어하고 싶다면, 컴포넌트에서 `wandb.log`와 `wandb.log_artifact` 호출을 추가할 수 있습니다.

### 명시적인 wandb 로깅 호출로

아래 예에서, 우리는 모델을 트레이닝하고 있습니다. `@wandb_log` 데코레이터는 관련 입력 및 출력을 자동으로 추적합니다. 트레이닝 프로세스를 로그하려면 다음과 같이 명시적으로 로깅을 추가할 수 있습니다:

```python
@wandb_log
def train_model(
    train_dataloader_path: components.InputPath("dataloader"),
    test_dataloader_path: components.InputPath("dataloader"),
    model_path: components.OutputPath("pytorch_model")
):
    ...
    for epoch in epochs:
        for batch_idx, (data, target) in enumerate(train_dataloader):
            ...
            if batch_idx % log_interval == 0:
                wandb.log({
                    "epoch": epoch,
                    "step": batch_idx * len(data),
                    "loss": loss.item()
                })
        ...
        wandb.log_artifact(model_artifact)
```

### 암시적인 wandb 인테그레이션으로

우리가 지원하는 [프레임워크 인테그레이션을 사용하고 있다면](https://docs.wandb.ai/guides/integrations), 콜백을 직접 전달할 수도 있습니다:

```python
@wandb_log
def train_model(
    train_dataloader_path: components.InputPath("dataloader"),
    test_dataloader_path: components.InputPath("dataloader"),
    model_path: components.OutputPath("pytorch_model")
):
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning import Trainer
    
    trainer = Trainer(logger=WandbLogger())
    ...  # do training
```