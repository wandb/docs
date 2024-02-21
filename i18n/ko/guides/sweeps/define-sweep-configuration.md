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

W&B 스윕은 하이퍼파라미터 값을 탐색하는 전략과 이를 평가하는 코드를 결합합니다. 전략은 모든 옵션을 시도하는 것처럼 단순할 수도 있고, 베이지안 최적화와 하이퍼밴드([BOHB](https://arxiv.org/abs/1807.01774))처럼 복잡할 수도 있습니다.

스윕 구성은 [Python 사전](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)이나 [YAML](https://yaml.org/) 파일 중 하나에서 정의할 수 있습니다. 스윕 구성을 어떻게 정의하느냐는 스윕을 어떻게 관리하고 싶은지에 따라 달라집니다.

:::info
명령 줄에서 스윕을 초기화하고 스윕 에이전트를 시작하려면 YAML 파일에서 스윕 구성을 정의하세요. Python 스크립트나 Jupyter 노트북 내에서 전적으로 스윕을 초기화하고 시작하려면 Python 사전에서 스윕을 정의하세요.
:::

다음 가이드는 스윕 구성을 형식화하는 방법을 설명합니다. 전체 스윕 구성 키 목록은 [스윕 구성 옵션](./sweep-config-keys.md)을 참조하세요.

## 기본 구조

스윕 구성 형식 옵션(YAML 및 Python 사전)은 모두 키-값 쌍과 중첩된 구조를 사용합니다.

스윕 구성 내의 최상위 키를 사용하여 스윕 검색의 특성을 정의하세요. 예를 들어 스윕의 이름([`name`](./sweep-config-keys.md#name) 키), 탐색할 파라미터([`parameters`](./sweep-config-keys.md#parameters) 키), 파라미터 공간 탐색 방법론([`method`](./sweep-config-keys.md#method) 키) 등을 정의할 수 있습니다.

예를 들어, 다음 코드 조각은 YAML 파일 내와 Python 사전 내에서 동일한 스윕 구성을 보여줍니다. 스윕 구성 내에는 다음과 같은 다섯 개의 최상위 키가 지정되어 있습니다: `program`, `name`, `method`, `metric` 및 `parameters`.

<Tabs
  defaultValue="cli"
  values={[    
    {label: 'CLI', value: 'cli'},
    {label: 'Python 스크립트 또는 Jupyter 노트북', value: 'script'},
  ]}>
  <TabItem value="script">

Python 스크립트나 Jupyter 노트북에서 학습 알고리즘을 정의하는 경우 Python 사전 데이터 구조에서 스윕을 정의하세요.

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
명령 줄(CLI)에서 상호 작용적으로 스윕을 관리하려면 YAML 파일에서 스윕 구성을 정의하세요.

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

최상위 `parameters` 키 내에는 `learning_rate`, `batch_size`, `epoch`, 및 `optimizer`와 같은 다음 키가 중첩되어 있습니다. 중첩된 각 키에 대해 하나 이상의 값을, 분포, 확률 등을 지정할 수 있습니다. 자세한 정보는 [스윕 구성 옵션](./sweep-config-keys.md)의 [parameters](./sweep-config-keys.md#parameters) 섹션을 참조하세요.

## 이중 중첩 파라미터

스윕 구성은 중첩 파라미터를 지원합니다. 중첩 파라미터를 구분하려면 최상위 파라미터 이름 아래에 추가 `parameters` 키를 사용하세요. 스윕 구성은 다중 레벨 중첩을 지원합니다.

베이지안 또는 랜덤 하이퍼파라미터 검색을 사용하는 경우 무작위 변수에 대한 확률 분포를 지정하세요. 각 하이퍼파라미터에 대해:

1. 스윕 구성에 최상위 `parameters` 키를 생성합니다.
2. `parameters`키 내부에 다음을 중첩합니다:
   1. 최적화하려는 하이퍼파라미터의 이름을 지정합니다.
   2. `distribution` 키에 사용할 분포를 지정합니다. `distribution` 키-값 쌍을 하이퍼파라미터 이름 아래에 중첩합니다.
   3. 탐색할 하나 이상의 값을 지정합니다. 값(또는 값들)은 분포 키와 일치해야 합니다.
      1. (옵션) 중첩 파라미터를 구분하기 위해 최상위 파라미터 이름 아래에 추가 파라미터 키를 사용합니다.

:::caution
스윕 구성에 정의된 중첩 파라미터는 W&B 실행 구성에서 지정된 키를 덮어씁니다.

예를 들어, 다음과 같은 구성으로 W&B 실행을 초기화하는 `train.py` Python 스크립트를 작성한다고 가정해 보세요(1-2행 참조). 다음으로, `sweep_configuration`이라는 사전에서 스윕 구성을 정의합니다(4-13행 참조). 그런 다음 스윕 구성 사전을 `wandb.sweep`에 전달하여 스윕 구성을 초기화합니다(16행 참조).

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

# 스윕 구성을 전달하여 스윕 초기화.
sweep_id = wandb.sweep(sweep=sweep_configuration, project="<project>")

# 스윕 작업 시작.
wandb.agent(sweep_id, function=main, count=4)
```
W&B 실행이 초기화될 때 전달된 `nested_param.manual_key`(2행)는 접근할 수 없습니다. `run.config`는 스윕 구성 사전(4-13행)에 정의된 키-값 쌍만을 가지고 있습니다.
:::

## 스윕 구성 템플릿

다음 템플릿은 파라미터를 구성하고 검색 제약 조건을 지정하는 방법을 보여줍니다. `hyperparameter_name`을 하이퍼파라미터의 이름으로 대체하고 `<>`로 둘러싸인 모든 값을 대체하세요.

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

### 명령 예시
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

다음 탭은 일반적인 명령 매크로를 지정하는 방법을 보여줍니다:

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python 해석기 설정', value: 'python'},
    {label: '추가 파라미터 추가', value: 'parameters'},
    {label: '인수 생략', value: 'omit'},
    {label: 'Hydra', value: 'hydra'}
  ]}>
  <TabItem value="python">

`{$interpreter}` 매크로를 제거하고 값으로 명시적으로 제공하여 Python 해석기를 하드코딩하는 방법은 다음 코드 조각에서 보여줍니다:

```yaml
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```
  </TabItem>
  <TabItem value="parameters">

스윕 구성 파라미터로 지정되지 않은 추가 명령 줄 인수를 추가하는 방법은 다음과 같습니다:

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

프로그램이 인수 분석을 사용하지 않는 경우 `wandb.init`이 자동으로 스윕 파라미터를 `wandb.config`에 가져오는 기능을 활용하여 전체적으로 인수 전달을 피할 수 있습니다:

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
```
  </TabItem>
  <TabItem value="hydra">

[Hydra](https://hydra.cc)와 같은 도구가 예상하는 방식으로 인수를 전달하도록 명령을 변경할 수 있습니다. 자세한 정보는 [W&B와 Hydra](../integrations/other/hydra.md)를 참조하세요.

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - ${args_no_hyphens}
```
  </TabItem>
</Tabs>