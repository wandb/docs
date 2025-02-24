---
title: Define a sweep configuration
description: 스윕을 위한 설정 파일을 만드는 방법을 알아보세요.
menu:
  default:
    identifier: ko-guides-models-sweeps-define-sweep-configuration-_index
    parent: sweeps
url: guides/sweeps/define-sweep-configuration
weight: 3
---

W&B Sweep은 하이퍼파라미터 값을 탐색하는 전략과 이를 평가하는 코드를 결합합니다. 이 전략은 모든 옵션을 시도하는 것만큼 간단할 수도 있고, 베이지안 최적화 및 Hyperband ([BOHB](https://arxiv.org/abs/1807.01774))만큼 복잡할 수도 있습니다.

[Python 사전](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) 또는 [YAML](https://yaml.org/) 파일에서 스윕 구성을 정의합니다. 스윕을 관리하려는 방식에 따라 스윕 구성 정의 방법이 달라집니다.

{{% alert %}}
커맨드 라인에서 스윕을 초기화하고 스윕 에이전트를 시작하려면 YAML 파일에서 스윕 구성을 정의하십시오. Python 스크립트 또는 Jupyter notebook 내에서 스윕을 초기화하고 시작하는 경우 Python 사전에서 스윕을 정의하십시오.
{{% /alert %}}

다음 가이드에서는 스윕 구성 형식을 지정하는 방법을 설명합니다. 최상위 스윕 구성 키의 전체 목록은 [스윕 구성 옵션]({{< relref path="./sweep-config-keys.md" lang="ko" >}})을 참조하십시오.

## 기본 구조

두 가지 스윕 구성 형식 옵션(YAML 및 Python 사전) 모두 키-값 쌍과 중첩 구조를 활용합니다.

스윕 이름([`name`]({{< relref path="./sweep-config-keys.md" lang="ko" >}}) 키), 검색할 파라미터([`parameters`]({{< relref path="./sweep-config-keys.md#parameters" lang="ko" >}}) 키), 파라미터 공간 검색 방법([`method`]({{< relref path="./sweep-config-keys.md#method" lang="ko" >}}) 키) 등과 같은 스윕 검색의 품질을 정의하려면 스윕 구성 내에서 최상위 키를 사용하십시오.

예를 들어, 다음 코드 조각은 YAML 파일과 Python 사전 내에서 정의된 동일한 스윕 구성을 보여줍니다. 스윕 구성 내에는 `program`, `name`, `method`, `metric` 및 `parameters`의 5가지 최상위 키가 지정되어 있습니다.

{{< tabpane  text=true >}}
  {{% tab header="CLI" %}}
커맨드 라인 (CLI)에서 스윕을 대화식으로 관리하려면 YAML 파일에서 스윕 구성을 정의하십시오.

```yaml title="config.yaml"
program: train.py
name: sweepdemo
method: bayes
metric:
  goal: minimize
  name: validation_loss
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
  batch_size:
    values: [16, 32, 64]
  epochs:
    values: [5, 10, 15]
  optimizer:
    values: ["adam", "sgd"]
```
  {{% /tab %}}
  {{% tab header="Python script or Jupyter notebook" %}}
Python 스크립트 또는 Jupyter notebook에서 트레이닝 알고리즘을 정의하는 경우 Python 사전 데이터 구조에서 스윕을 정의합니다.

다음 코드 조각은 스윕 구성을 `sweep_configuration`이라는 변수에 저장합니다.

```python title="train.py"
sweep_configuration = {
    "name": "sweepdemo",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "validation_loss"},
    "parameters": {
        "learning_rate": {"min": 0.0001, "max": 0.1},
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [5, 10, 15]},
        "optimizer": {"values": ["adam", "sgd"]},
    },
}
```  
  {{% /tab %}}
{{< /tabpane >}}

최상위 `parameters` 키 내에는 `learning_rate`, `batch_size`, `epoch` 및 `optimizer` 키가 중첩되어 있습니다. 지정하는 각 중첩 키에 대해 하나 이상의 값, 분포, 확률 등을 제공할 수 있습니다. 자세한 내용은 [스윕 구성 옵션]({{< relref path="./sweep-config-keys.md" lang="ko" >}})의 [parameters]({{< relref path="./sweep-config-keys.md#parameters" lang="ko" >}}) 섹션을 참조하십시오.

## 이중 중첩 파라미터

스윕 구성은 중첩된 파라미터를 지원합니다. 중첩된 파라미터를 구분하려면 최상위 파라미터 이름 아래에 추가 `parameters` 키를 사용하십시오. 스윕 구성은 다단계 중첩을 지원합니다.

베이지안 또는 랜덤 하이퍼파라미터 검색을 사용하는 경우 랜덤 변수에 대한 확률 분포를 지정하십시오. 각 하이퍼파라미터에 대해 다음을 수행하십시오.

1. 스윕 구성에서 최상위 `parameters` 키를 만듭니다.
2. `parameters` 키 내에서 다음을 중첩합니다.
   1. 최적화하려는 하이퍼파라미터의 이름을 지정합니다.
   2. `distribution` 키에 사용할 분포를 지정합니다. 하이퍼파라미터 이름 아래에 `distribution` 키-값 쌍을 중첩합니다.
   3. 탐색할 하나 이상의 값을 지정합니다. 값은 분포 키와 함께 있어야 합니다.
      1. (선택 사항) 최상위 파라미터 이름 아래에 추가 파라미터 키를 사용하여 중첩된 파라미터를 구분합니다.

{{% alert color="secondary" %}}
스윕 구성에 정의된 중첩된 파라미터는 W&B run 구성에 지정된 키를 덮어씁니다.

예를 들어, `train.py` Python 스크립트에서 다음 구성으로 W&B run을 초기화한다고 가정합니다(1-2행 참조). 다음으로 `sweep_configuration`이라는 사전에서 스윕 구성을 정의합니다(4-13행 참조). 그런 다음 스윕 구성 사전을 `wandb.sweep`에 전달하여 스윕 구성을 초기화합니다(16행 참조).

```python title="train.py" showLineNumbers
def main():
    run = wandb.init(config={"nested_param": {"manual_key": 1}})


sweep_configuration = {
    "top_level_param": 0,
    "nested_param": {
        "learning_rate": 0.01,
        "double_nested_param": {"x": 0.9, "y": 0.8},
    },
}

# Initialize sweep by passing in config.
sweep_id = wandb.sweep(sweep=sweep_configuration, project="<project>")

# Start sweep job.
wandb.agent(sweep_id, function=main, count=4)
```
W&B run이 초기화될 때 전달되는 `nested_param.manual_key`(2행)에는 액세스할 수 없습니다. `run.config`는 스윕 구성 사전에 정의된 키-값 쌍만 가지고 있습니다(4-13행).
{{% /alert %}}

## 스윕 구성 템플릿

다음 템플릿은 파라미터를 구성하고 검색 제약 조건을 지정하는 방법을 보여줍니다. `<>`로 묶인 모든 값과 함께 `hyperparameter_name`을 하이퍼파라미터 이름으로 바꿉니다.

```yaml title="config.yaml"
program: <insert>
method: <insert>
parameter:
  hyperparameter_name0:
    value: 0  
  hyperparameter_name1: 
    values: [0, 0, 0]
  hyperparameter_name: 
    distribution: <insert>
    value: <insert>
  hyperparameter_name2:  
    distribution: <insert>
    min: <insert>
    max: <insert>
    q: <insert>
  hyperparameter_name3: 
    distribution: <insert>
    values:
      - <list_of_values>
      - <list_of_values>
      - <list_of_values>
early_terminate:
  type: hyperband
  s: 0
  eta: 0
  max_iter: 0
command:
- ${Command macro}
- ${Command macro}
- ${Command macro}
- ${Command macro}      
```

## 스윕 구성 예제

{{< tabpane text=true >}}
  {{% tab header="CLI" %}}

```yaml title="config.yaml" 
program: train.py
method: random
metric:
  goal: minimize
  name: loss
parameters:
  batch_size:
    distribution: q_log_uniform_values
    max: 256 
    min: 32
    q: 8
  dropout: 
    values: [0.3, 0.4, 0.5]
  epochs:
    value: 1
  fc_layer_size: 
    values: [128, 256, 512]
  learning_rate:
    distribution: uniform
    max: 0.1
    min: 0
  optimizer:
    values: ["adam", "sgd"]
```

  {{% /tab %}}
  {{% tab header="Python script or Jupyter notebook" %}}

```python title="train.py" 
sweep_config = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "batch_size": {
            "distribution": "q_log_uniform_values",
            "max": 256,
            "min": 32,
            "q": 8,
        },
        "dropout": {"values": [0.3, 0.4, 0.5]},
        "epochs": {"value": 1},
        "fc_layer_size": {"values": [128, 256, 512]},
        "learning_rate": {"distribution": "uniform", "max": 0.1, "min": 0},
        "optimizer": {"values": ["adam", "sgd"]},
    },
}
```  

  {{% /tab %}}
{{< /tabpane >}}

### Bayes hyperband 예제

```yaml
program: train.py
method: bayes
metric:
  goal: minimize
  name: val_loss
parameters:
  dropout:
    values: [0.15, 0.2, 0.25, 0.3, 0.4]
  hidden_layer_size:
    values: [96, 128, 148]
  layer_1_size:
    values: [10, 12, 14, 16, 18, 20]
  layer_2_size:
    values: [24, 28, 32, 36, 40, 44]
  learn_rate:
    values: [0.001, 0.01, 0.003]
  decay:
    values: [1e-5, 1e-6, 1e-7]
  momentum:
    values: [0.8, 0.9, 0.95]
  epochs:
    value: 27
early_terminate:
  type: hyperband
  s: 2
  eta: 3
  max_iter: 27
```

다음 탭은 `early_terminate`에 대한 최소 또는 최대 반복 횟수를 지정하는 방법을 보여줍니다.

{{% tabpane text=true %}}
  {{% tab header="Maximum number of iterations specified" %}}

```yaml
early_terminate:
  type: hyperband
  min_iter: 3
```

이 예제의 괄호는 `[3, 3*eta, 3*eta*eta, 3*eta*eta*eta]`이며, 이는 `[3, 9, 27, 81]`과 같습니다.
  {{% /tab %}}
  {{% tab header="Minimum number of iterations specified" %}}

```yaml
early_terminate:
  type: hyperband
  max_iter: 27
  s: 2
```

이 예제의 괄호는 `[27/eta, 27/eta/eta]`이며, 이는 `[9, 3]`과 같습니다.

  {{% /tab %}}
{{% /tabpane %}}

### Command 예제

```yaml
program: main.py
metric:
  name: val_loss
  goal: minimize

method: bayes
parameters:
  optimizer.config.learning_rate:
    min: !!float 1e-5
    max: 0.1
  experiment:
    values: [expt001, expt002]
  optimizer:
    values: [sgd, adagrad, adam]

command:
- ${env}
- ${interpreter}
- ${program}
- ${args_no_hyphens}
```

{{< tabpane text=true >}}
  {{% tab header="Unix" %}}

```bash
/usr/bin/env python train.py --param1=value1 --param2=value2
```  

  {{% /tab %}}
  {{% tab header="Windows" %}}

```bash
python train.py --param1=value1 --param2=value2

```  
  {{% /tab %}}
{{< /tabpane >}}

다음 탭은 일반적인 커맨드 매크로를 지정하는 방법을 보여줍니다.

{{< tabpane text=true >}}
  {{% tab header="Set Python interpreter" %}}

`{$interpreter}` 매크로를 제거하고 Python 인터프리터를 하드 코딩하기 위해 값을 명시적으로 제공하십시오. 예를 들어, 다음 코드 조각은 이를 수행하는 방법을 보여줍니다.

```yaml
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```

  {{% /tab %}}
  {{% tab header="Add extra parameters" %}}

다음은 스윕 구성 파라미터로 지정되지 않은 추가 커맨드 라인 인수를 추가하는 방법을 보여줍니다.

```yaml
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - "--config"
  - "your-training-config.json"
  - ${args}
```

  {{% /tab %}}
  {{% tab header="Omit arguments" %}}

프로그램에서 인수 구문 분석을 사용하지 않는 경우 인수를 모두 전달하지 않고 `wandb.init`가 스윕 파라미터를 자동으로 `wandb.config`에 포함하는 것을 활용할 수 있습니다.

```yaml
command:
  - ${env}
  - ${interpreter}
  - ${program}
```  

  {{% /tab %}}
  {{% tab header="Hydra" %}}

[Hydra](https://hydra.cc)와 같은 툴이 예상하는 방식으로 인수를 전달하도록 커맨드를 변경할 수 있습니다. 자세한 내용은 [W&B와 함께 Hydra]({{< relref path="/guides/integrations/hydra.md" lang="ko" >}})를 참조하십시오.

```yaml
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - ${args_no_hyphens}
```

  {{% /tab %}}
{{< /tabpane >}}
