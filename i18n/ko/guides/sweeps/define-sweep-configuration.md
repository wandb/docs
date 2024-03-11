---
description: Learn how to create configuration files for sweeps.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 스윕 구성 구조

<head>
  <title>하이퍼파라미터 튜닝을 위한 스윕 구성 정의.</title>
</head>

W&B 스윕은 하이퍼파라미터 값을 탐색하는 전략과 이를 평가하는 코드를 결합합니다. 전략은 모든 옵션을 시도하는 것처럼 단순할 수도 있고 베이지안 최적화 및 Hyperband ([BOHB](https://arxiv.org/abs/1807.01774))와 같이 복잡할 수도 있습니다.

스윕 구성을 [Python 사전](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)이나 [YAML](https://yaml.org/) 파일로 정의합니다. 스윕 구성을 정의하는 방법은 스윕을 관리하고자 하는 방법에 따라 달라집니다.

:::info
커맨드라인에서 스윕을 초기화하고 스윕 에이전트를 시작하려는 경우 YAML 파일에 스윕 구성을 정의하십시오. 파이썬 스크립트나 Jupyter 노트북 내에서 스윕을 초기화하고 시작하려는 경우 파이썬 사전에 스윕을 정의하십시오.
:::

다음 가이드에서는 스윕 구성을 어떻게 포맷하는지 설명합니다. 스윕 구성 옵션의 상위 수준 스윕 구성 키에 대한 전체 목록은 [스윕 구성 옵션](./sweep-config-keys.md)을 참조하십시오.

## 기본 구조

스윕 구성 형식 옵션(YAML 및 Python 사전) 모두 키-값 쌍과 중첩 구조를 사용합니다.

스윕 구성 내의 상위 수준 키를 사용하여 스윕 검색의 특성을 정의하십시오. 예를 들어 스윕의 이름([`name`](./sweep-config-keys.md#name) 키), 검색할 파라미터([`parameters`](./sweep-config-keys.md#parameters) 키), 파라미터 공간을 검색하는 방법론([`method`](./sweep-config-keys.md#method) 키) 등입니다.

예를 들어, 다음 코드 조각은 YAML 파일과 Python 사전 내에서 동일한 스윕 구성을 보여줍니다. 스윕 구성에는 `program`, `name`, `method`, `metric`, `parameters`의 다섯 가지 상위 수준 키가 지정됩니다.


<Tabs
  defaultValue="cli"
  values={[    
    {label: 'CLI', value: 'cli'},
    {label: 'Python 스크립트 또는 Jupyter 노트북', value: 'script'},
  ]}>
  <TabItem value="script">

Python 스크립트나 Jupyter 노트북에서 트레이닝 알고리즘을 정의하는 경우 파이썬 사전 데이터 구조에서 스윕을 정의하십시오.

다음 코드 조각은 `sweep_configuration`이라는 변수에 스윕 구성을 저장합니다:

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
  </TabItem>
  <TabItem value="cli">
커맨드라인(CLI)에서 스윕을 대화식으로 관리하려는 경우 YAML 파일에서 스윕 구성을 정의하십시오.

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
  </TabItem>
</Tabs>

상위 수준 `parameters` 키 내에서 다음 키가 중첩됩니다: `learning_rate`, `batch_size`, `epoch`, `optimizer`. 중첩된 각 키에 대해 하나 이상의 값을, 분포를, 확률 등을 지정할 수 있습니다. 자세한 내용은 [스윕 구성 옵션](./sweep-config-keys.md) 내의 [파라미터](./sweep-config-keys.md#parameters) 섹션을 참조하십시오.

## 이중 중첩 파라미터

스윕 구성은 중첩 파라미터를 지원합니다. 중첩 파라미터를 구분하기 위해 상위 수준 파라미터 이름 아래에 추가 `parameters` 키를 사용하십시오. 스윕 구성은 다중 레벨 중첩을 지원합니다.

베이지안 또는 랜덤 하이퍼파라미터 검색을 사용하는 경우 랜덤 변수에 대한 확률 분포를 지정하십시오. 각 하이퍼파라미터에 대해:

1. 스윕 구성에 상위 수준 `parameters` 키를 생성하십시오.
2. `parameters` 키 내부에 다음을 중첩하십시오:
   1. 최적화하려는 하이퍼파라미터의 이름을 지정하십시오.
   2. `distribution` 키에 사용할 분포를 지정하십시오. 하이퍼파라미터 이름 아래에 `distribution` 키-값 쌍을 중첩하십시오.
   3. 탐색할 하나 이상의 값을 지정하십시오. 값(또는 값들)은 분포 키와 일치해야 합니다.
      1. (선택사항) 상위 수준 파라미터 이름 아래에 추가 파라미터 키를 사용하여 중첩 파라미터를 구분하십시오.








:::caution
스윕 구성에서 정의된 중첩 파라미터는 W&B run 구성에서 지정된 키를 덮어씁니다.

예를 들어, `train.py` 파이썬 스크립트에서 다음 구성으로 W&B run을 초기화한다고 가정합니다(1-2줄 참조). 다음으로, `sweep_configuration`이라는 사전에서 스윕 구성을 정의합니다(4-13줄 참조). 그런 다음 스윕 구성 사전을 `wandb.sweep`에 전달하여 스윕 구성을 초기화합니다(16줄 참조).


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

# 스윕을 초기화하기 위해 config를 전달합니다.
sweep_id = wandb.sweep(sweep=sweep_configuration, project="<project>")

# 스윕 작업을 시작합니다.
wandb.agent(sweep_id, function=main, count=4)
```
W&B run이 초기화될 때 전달된 `nested_param.manual_key`(2줄)는 접근할 수 없습니다. `run.config`는 스윕 구성 사전(4-13줄)에 정의된 키-값 쌍만 가집니다.
:::

## 스윕 구성 템플릿


다음 템플릿은 파라미터를 구성하고 검색 제약 조건을 지정하는 방법을 보여줍니다. `hyperparameter_name`을 하이퍼파라미터의 이름으로 바꾸고 `<>`로 묶인 값을 교체하십시오.

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

## 스윕 구성 예시

<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'Python 스크립트 또는 Jupyter 노트북', value: 'notebook'},
  ]}>
  <TabItem value="cli">


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

  </TabItem>
  <TabItem value="notebook">

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

  </TabItem>
</Tabs>

### 베이지안 하이퍼밴드 예시
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

다음 탭은 `early_terminate`에 대해 최소 또는 최대 반복 횟수를 지정하는 방법을 보여줍니다:

<Tabs
  defaultValue="min_iter"
  values={[
    {label: '지정된 최소 반복 횟수', value: 'min_iter'},
    {label: '지정된 최대 반복 횟수', value: 'max_iter'},
  ]}>
  <TabItem value="min_iter">

```yaml
early_terminate:
  type: hyperband
  min_iter: 3
```

이 예시의 괄호는 `[3, 3*eta, 3*eta*eta, 3*eta*eta*eta]`로, `[3, 9, 27, 81]`과 같습니다.
  </TabItem>
  <TabItem value="max_iter">

```yaml
early_terminate:
  type: hyperband
  max_iter: 27
  s: 2
```

이 예시의 괄호는 `[27/eta, 27/eta/eta]`로, `[9, 3]`과 같습니다.
  </TabItem>
</Tabs>

### 코맨드 예시
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

<Tabs
  defaultValue="unix"
  values={[
    {label: 'Unix', value: 'unix'},
    {label: 'Windows', value: 'windows'},
  ]}>
  <TabItem value="unix">

```bash
/usr/bin/env python train.py --param1=value1 --param2=value2
```
  </TabItem>
  <TabItem value="windows">

```bash
python train.py --param1=value1 --param2=value2
```
  </TabItem>
</Tabs>

다음 탭은 일반적인 코맨드 매크로를 지정하는 방법을 보여줍니다:

<Tabs
  defaultValue="python"
  values={[
    {label: '파이썬 인터프리터 설정', value: 'python'},
    {label: '추가 파라미터 추가', value: 'parameters'},
    {label: '인수 생략', value: 'omit'},
    {label: 'Hydra', value: 'hydra'}
  ]}>
  <TabItem value="python">

`{$interpreter}` 매크로를 제거하고 파이썬 인터프리터를 명시적으로 제공하여 하드코딩하십시오. 예를 들어, 다음 코드 조각은 이를 수행하는 방법을 보여줍니다:

```yaml
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```
  </TabItem>
  <TabItem value="parameters">

스윕 구성 파라미터에 의해 지정되지 않은 추가 커맨드라인 인수를 추가하는 방법은 다음과 같습니다:

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - "--config"
  - "your-training-config.json"
  - ${args}
```

  </TabItem>
  <TabItem value="omit">

프로그램이 인수 분석을 사용하지 않는 경우 모든 인수를 전달하지 않고 `wandb.init`이 자동으로 스윕 파라미터를 `wandb.config`로 가져오는 것을 활용할 수 있습니다:

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
```
  </TabItem>
  <TabItem value="hydra">

[Hydra](https://hydra.cc)와 같은 툴이 기대하는 방식으로 인수를 전달하도록 코맨드를 변경할 수 있습니다. 자세한 내용은 [Hydra와 W&B](../integrations/other/hydra.md)를 참조하십시오.

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - ${args_no_hyphens}
```
  </TabItem>
