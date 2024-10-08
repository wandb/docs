---
title: Configure sweeps
description: 스윕을 위한 설정 파일을 만드는 방법을 배우세요.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B Sweep는 하이퍼파라미터 값을 탐색하는 전략과 이를 평가하는 코드를 결합합니다. 전략은 모든 옵션을 시도하는 간단한 방법부터 베이지안 최적화 및 하이퍼밴드 ([BOHB](https://arxiv.org/abs/1807.01774))와 같은 복잡한 방법까지 다양할 수 있습니다.

스윕 구성을 [Python 사전](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) 또는 [YAML](https://yaml.org/) 파일로 정의합니다. 스윕 구성을 정의하는 방법은 스윕을 관리하고자 하는 방식에 따라 다릅니다.

:::info
커맨드라인에서 스윕을 초기화하고 스윕 에이전트를 시작하려면 스윕 구성을 YAML 파일로 정의하십시오. Python 스크립트나 Jupyter 노트북 내에서 스윕을 완전히 초기화하고 시작하려면 스윕을 Python 사전으로 정의하십시오.
:::

다음 가이드는 스윕 구성을 포맷하는 방법을 설명합니다. 상위 레벨 스윕 구성 키의 종합 목록은 [Sweep 구성 옵션](./sweep-config-keys.md)을 참조하십시오.

## 기본 구조

스윕 구성의 두 가지 포맷 옵션(YAML 및 Python 사전)은 키-값 쌍과 중첩 구조를 사용합니다.

스윕 구성 내의 상위 레벨 키를 사용하여 스윕 검색의 특성을 정의합니다. 예를 들어 스윕의 이름 ([`name`](./sweep-config-keys.md) 키), 검색할 파라미터 ([`parameters`](./sweep-config-keys.md#parameters) 키), 파라미터 공간을 검색하는 방법론 ([`method`](./sweep-config-keys.md#method) 키) 등을 정의합니다.

예를 들어, 다음 코드조각은 동일한 스윕 구성이 YAML 파일과 Python 사전 내에서 각각 정의된 예시를 보여줍니다. 스윕 구성 내에는 지정된 다섯 개의 상위 레벨 키가 있습니다: `program`, `name`, `method`, `metric` 및 `parameters`.

<Tabs
  defaultValue="cli"
  values={[    
    {label: 'CLI', value: 'cli'},
    {label: 'Python script or Jupyter notebook', value: 'script'},
  ]}>
  <TabItem value="script">

Python 스크립트 또는 Jupyter 노트북 내에서 트레이닝 알고리즘을 정의하면 Python 사전 데이터 구조로 스윕을 정의하십시오.

다음 코드조각은 `sweep_configuration`라는 변수에 스윕 구성을 저장합니다:

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
스윕 구성을 YAML 파일로 정의하여 커맨드라인(CLI)에서 대화형으로 스윕을 관리하십시오.

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

상위 레벨 `parameters` 키 내에 `learning_rate`, `batch_size`, `epoch` 및 `optimizer`와 같은 키가 중첩되어 있습니다. 지정한 각 중첩 키에 대해 하나 이상의 값, 분포, 확률 등을 제공할 수 있습니다. 자세한 내용은 [Sweep 구성 옵션](./sweep-config-keys.md) 내의 [parameters](./sweep-config-keys.md#parameters) 섹션을 참조하십시오.

## 이중 중첩 파라미터

스윕 구성은 중첩 파라미터를 지원합니다. 중첩 파라미터를 구분하기 위해 상위 레벨 파라미터 이름 아래에 추가 `parameters` 키를 사용하세요. 스윕 구성은 다단계 중첩을 지원합니다.

랜덤 또는 베이지안 하이퍼파라미터 검색을 사용하는 경우 확률 분포를 지정하십시오. 각 하이퍼파라미터에 대해 다음을 수행하십시오:

1. 스윕 구성에서 상위 레벨 `parameters` 키를 만드세요. 
2. `parameters` 키 내에서 다음을 중첩하세요:
   1. 최적화하려는 하이퍼파라미터의 이름을 지정합니다.
   2. `distribution` 키에 사용할 분포를 지정합니다. `distribution` 키-값 쌍은 하이퍼파라미터 이름 아래에 중첩시킵니다.
   3. 탐색할 하나 이상의 값을 지정합니다. 값(또는 값들)은 분포 키와 일치해야 합니다.  
      1. (선택 사항) 중첩 파라미터를 구분하기 위해 상위 레벨 파라미터 이름 아래에 추가 파라미터 키를 사용하세요.

:::caution
스윕 구성에서 정의된 중첩 파라미터는 W&B run 구성에 명시된 키를 덮어씁니다.

예를 들어, `train.py` Python 스크립트에서 다음 설정으로 W&B run을 초기화한다고 가정합니다(Lines 1-2 참조). 다음으로, `sweep_configuration`이라는 사전에서 스윕 구성을 정의합니다(Lines 4-13 참조). 그런 다음 `wandb.sweep`에 스윕 구성 사전을 전달하여 스윕 구성을 초기화합니다(Line 16 참조).

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
W&B run 초기화 시 전달된 `nested_param.manual_key` (2행 참조)는 엑세스할 수 없습니다. `run.config`는 스윕 구성 사전(4-13행)에서 정의된 키-값 쌍만을 가지고 있습니다.
:::

## 스윕 구성 템플릿

다음 템플릿은 파라미터를 구성하고 검색 제약 조건을 지정하는 방법을 보여줍니다. `hyperparameter_name`을 하이퍼파라미터의 이름으로, `<>`로 둘러싸인 값을 원하는 값으로 바꾸십시오.

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
    {label: 'Python script or Jupyter notebook', value: 'notebook'},
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

다음 탭은 `early_terminate`의 최소 또는 최대 반복 횟수를 지정하는 방법을 보여줍니다:

<Tabs
  defaultValue="min_iter"
  values={[
    {label: 'Minimum number of iterations specified', value: 'min_iter'},
    {label: 'Maximum number of iterations specified', value: 'max_iter'},
  ]}>
  <TabItem value="min_iter">

```yaml
early_terminate:
  type: hyperband
  min_iter: 3
```

이 예의 브라켓은: `[3, 3*eta, 3*eta*eta, 3*eta*eta*eta]`이며, 이는 `[3, 9, 27, 81]`와 같습니다.
  </TabItem>
  <TabItem value="max_iter">

```yaml
early_terminate:
  type: hyperband
  max_iter: 27
  s: 2
```

이 예의 브라켓은 `[27/eta, 27/eta/eta]`이며, 이는 `[9, 3]`와 같습니다.
  </TabItem>
</Tabs>

### 커맨드 예시
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

다음 탭은 일반적인 커맨드 매크로를 지정하는 방법을 보여줍니다:

<Tabs
  defaultValue="python"
  values={[
    {label: 'Set python interpreter', value: 'python'},
    {label: 'Add extra parameters', value: 'parameters'},
    {label: 'Omit arguments', value: 'omit'},
    {label: 'Hydra', value: 'hydra'}
  ]}>
  <TabItem value="python">

`{$interpreter}` 매크로를 제거하고 Python 인터프리터를 하드코딩하기 위해 값을 명시적으로 제공하십시오. 예를 들어, 다음 코드조각은 이를 수행하는 방법을 보여줍니다:

```yaml
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```
  </TabItem>
  <TabItem value="parameters">

스윕 구성 파라미터에 명시되지 않은 추가 커맨드라인 파라미터를 추가하는 방법은 다음과 같습니다:

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

프로그램이 인수 파싱을 사용하지 않는 경우 인수 전달을 모두 피하고 `wandb.init`이 스윕 파라미터를 자동으로 `wandb.config`로 가져가는 이점을 활용할 수 있습니다:

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
```
  </TabItem>
  <TabItem value="hydra">

커맨드를 [Hydra](https://hydra.cc)와 같은 툴이 예상하는 방식으로 인수를 전달하도록 변경할 수 있습니다. 자세한 내용은 [Hydra와 W&B](../integrations/other/hydra.md)를 참조하십시오.

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - ${args_no_hyphens}
```
  </TabItem>
</Tabs>