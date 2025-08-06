---
title: 스윕 구성을 정의하기
description: 스윕을 위한 설정 파일을 만드는 방법을 알아보세요.
menu:
  default:
    identifier: ko-guides-models-sweeps-define-sweep-configuration-_index
    parent: sweeps
url: guides/sweeps/define-sweep-configuration
weight: 3
---

W&B Sweep 은 하이퍼파라미터 값 탐색을 위한 전략과, 해당 값을 평가하는 코드를 결합한 것입니다. 전략은 모든 옵션을 하나씩 시도하는 간단한 방식부터, 복잡한 베이지안 최적화 및 Hyperband ([BOHB](https://arxiv.org/abs/1807.01774)) 방식까지 다양하게 선택할 수 있습니다.

스윕 구성(Sweep configuration)은 [Python 사전](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)이나 [YAML](https://yaml.org/) 파일로 정의할 수 있습니다. 스윕 구성을 어떻게 정의할지는 스윕 관리 방식에 따라 달라집니다.

{{% alert %}}
커맨드라인에서 스윕을 초기화하고 스윕 에이전트를 실행하려면 YAML 파일에 스윕 구성을 작성하세요. Python 스크립트나 노트북 내에서 스윕을 전부 실행하려면 Python 사전에 스윕을 정의하면 됩니다.
{{% /alert %}}

아래 가이드는 스윕 구성을 포맷하는 방법을 안내합니다. 스윕 구성 키 전체 목록은 [스윕 구성 옵션]({{< relref path="./sweep-config-keys.md" lang="ko" >}})을 참고하세요.

## 기본 구조

스윕 구성은 YAML, Python 사전 두 가지 포맷 모두 key-value(키-값) 쌍과 중첩 구조를 사용합니다.  

스윕 구성에서는 상위 키를 사용해 스윕 검색의 특징(예: Sweep 이름([`name`]({{< relref path="./sweep-config-keys.md" lang="ko" >}}) 키), 탐색할 파라미터([`parameters`]({{< relref path="./sweep-config-keys.md#parameters" lang="ko" >}}) 키), 파라미터 검색 방법([`method`]({{< relref path="./sweep-config-keys.md#method" lang="ko" >}}) 키) 등)을 정의합니다.  

예시로, 아래 코드조각은 동일한 스윕 구성을 YAML 파일과 Python 사전으로 각각 정의한 것입니다. 스윕 구성에는 5개의 상위 키(`program`, `name`, `method`, `metric`, `parameters`)가 들어갑니다.  

{{< tabpane  text=true >}}
  {{% tab header="CLI" %}}
커맨드라인(CLI)에서 Sweeps 를 인터랙티브하게 관리하고 싶다면 YAML 파일로 스윕 구성을 작성하세요.

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
  {{% tab header="Python script or notebook" %}}
Python 스크립트나 노트북에서 트레이닝 알고리즘을 정의한다면, 파이썬 사전 구조에 스윕을 작성할 수 있습니다.

아래 코드조각은 스윕 구성을 `sweep_configuration` 변수에 저장합니다:

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

상위 `parameters` 키 아래에는 `learning_rate`, `batch_size`, `epoch`, `optimizer` 등이 중첩되어 있습니다. 각각의 중첩 키에는 하나 이상의 값, 분포, 확률 등을 지정할 수 있습니다. 자세한 내용은 [parameters]({{< relref path="./sweep-config-keys.md#parameters" lang="ko" >}}) 섹션과 [스윕 구성 옵션]({{< relref path="./sweep-config-keys.md" lang="ko" >}})을 참고하세요.

## 이중 중첩 파라미터(Double nested parameters)

스윕 구성은 중첩 파라미터를 지원합니다. 중첩된 파라미터를 구분하려면 상위 파라미터 명 아래에 추가 `parameters` 키를 사용하세요. 스윕 구성은 다단계 중첩이 가능합니다.

베이지안 또는 랜덤 하이퍼파라미터 탐색을 사용할 경우, 랜덤 변수에 대한 분포를 지정할 수 있습니다. 각 하이퍼파라미터에 대해 다음과 같이 설정하세요.

1. 스윕 구성에서 상위 `parameters` 키를 생성합니다.
2. `parameters` 키 내에 다음 정보를 중첩합니다:
   1. 최적화하고자 하는 하이퍼파라미터 이름을 지정합니다.
   2. 사용할 분포를 `distribution` 키에 지정합니다. `distribution` 키-값 쌍은 하이퍼파라미터 이름 아래에 중첩됩니다.
   3. 탐색할 값(들)을 지정합니다. 값 또는 값들은 분포 키와 동일 레벨에 위치해야 합니다.
      1. (선택사항) 상위 파라미터 아래 추가로 `parameters` 키를 사용해 중첩 파라미터를 구분할 수 있습니다.

{{% alert color="secondary" %}}
스윕 구성에 정의된 중첩 파라미터는 W&B run 구성에서 지정한 키를 덮어씁니다.

예시로, 아래처럼 Python 스크립트(예: `train.py`)에서 W&B run 을 다음과 같이 초기화했다고 가정합니다(1~2라인). 그 다음, 딕셔너리 `sweep_configuration`에 스윕 구성을 정의합니다(4~13라인). 마지막으로 이 스윕 구성 딕셔너리를 `wandb.sweep` 에 전달하여 스윕을 초기화합니다(16라인).

```python title="train.py" 
def main():
    run = wandb.init(config={"nested_param": {"manual_key": 1}})


sweep_configuration = {
    "top_level_param": 0,
    "nested_param": {
        "learning_rate": 0.01,
        "double_nested_param": {"x": 0.9, "y": 0.8},
    },
}

# config 를 넘겨 스윕 초기화
sweep_id = wandb.sweep(sweep=sweep_configuration, project="<project>")

# 스윕 작업 시작
wandb.agent(sweep_id, function=main, count=4)
```
W&B run 을 초기화할 때 전달한 `nested_param.manual_key` 는 엑세스할 수 없습니다. `wandb.Run.config` 에는 오직 스윕 구성 딕셔너리에 정의된 키-값 쌍만 존재합니다.
{{% /alert %}}

## 스윕 구성 템플릿

아래 템플릿은 파라미터를 설정하고 탐색 제약 조건을 지정하는 예시입니다. `hyperparameter_name` 자리는 하이퍼파라미터 이름으로, 꺾쇠괄호(`< >`)로 감싸진 값들은 원하는 값으로 바꿔주세요.

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

숫자 값을 과학적 표기법으로 표현할 경우, YAML `!!float` 연산자를 붙여주면 됩니다. 이는 해당 값을 실수형으로 변환합니다. 예: `min: !!float 1e-5`. [커맨드 예시]({{< relref path="#command-example" lang="ko" >}}) 참고.

## 스윕 구성 예시

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
  {{% tab header="Python script or notebook" %}}

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

### 베이즈 하이퍼밴드 예시

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

아래 탭에서는 `early_terminate` 에 대해 최소 또는 최대 반복 횟수를 지정하는 법을 설명합니다.

{{< tabpane  text=true >}}
  {{% tab header="Maximum number of iterations" %}}

이 예시에서 반복 구간(brackets)은 `[3, 3*eta, 3*eta*eta, 3*eta*eta*eta]`로, 결과적으로 `[3, 9, 27, 81]` 입니다.  

```yaml
early_terminate:
  type: hyperband
  min_iter: 3
```

  {{% /tab %}}
  {{% tab header="Minimum number of iterations" %}}

이 예시에서 반복 구간(brackets)은 `[27/eta, 27/eta/eta]`, 즉 `[9, 3]` 입니다.

```yaml
early_terminate:
  type: hyperband
  max_iter: 27
  s: 2
```

  {{% /tab %}}
{{< /tabpane >}}

### 매크로 및 커스텀 커맨드 ARG 예시

복잡한 커맨드라인 인수가 필요한 경우, 매크로를 활용해 환경 변수, 파이썬 인터프리터, 추가 인수 등을 전달할 수 있습니다. [W&B는 사전 정의된 매크로]({{< relref path="./sweep-config-keys.md#command-macros" lang="ko" >}})와 스윕 구성에서 지정 가능한 커스텀 커맨드라인 인수 모두를 지원합니다.

예를 들어, 아래 스윕 구성(`sweep.yaml`)은 Python 스크립트(`run.py`)를 실행하도록 커맨드를 정의하고 있습니다. Sweeps 실행 시 `${env}`, `${interpreter}`, `${program}` 매크로는 각각 상황에 맞는 값으로 치환됩니다.

그리고 `--batch_size=${batch_size}`, `--test=True`, `--optimizer=${optimizer}` 인수에서는 스윕 설정에 정의된 `batch_size`, `test`, `optimizer` 파라미터 값이 매크로로 전달됩니다.

```yaml title="sweep.yaml"
program: run.py
method: random
metric:
  name: validation_loss
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - "--batch_size=${batch_size}"
  - "--optimizer=${optimizer}"
  - "--test=True"
```
연관된 Python 스크립트(`run.py`)에서는 `argparse` 모듈을 사용해 이 커맨드라인 인수들을 파싱할 수 있습니다.

```python title="run.py"
# run.py  
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], required=True)
parser.add_argument('--test', type=str2bool, default=False)
args = parser.parse_args()

# W&B Run 초기화
with wandb.init('test-project') as run:
    run.log({'validation_loss':1})
```

스윕 구성에 사용 가능한 사전 정의된 매크로 목록은 [스윕 구성 옵션]({{< relref path="./sweep-config-keys.md" lang="ko" >}})의 [Command macros]({{< relref path="./sweep-config-keys.md#command-macros" lang="ko" >}}) 섹션을 확인하세요.

#### 불리언 인수 처리

`argparse` 모듈은 기본적으로 불리언 타입의 인수를 직접 지원하지 않습니다. 불리언 인수를 정의하려면 [`action`](https://docs.python.org/3/library/argparse.html#action) 파라미터를 활용하거나, 문자열 표현을 불리언 타입으로 변환하는 커스텀 함수를 사용해야 합니다.

예를 들어 다음 코드조각처럼 `store_true` 또는 `store_false` 값을 `ArgumentParser`에 인수로 넘겨서 불리언 인수를 정의할 수 있습니다.

```python
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

args.test  # --test가 전달되면 True, 없으면 False가 됨
```

또는 문자열 표현을 불리언 형태로 변환하는 커스텀 함수를 만들 수도 있습니다. 예시는 아래와 같습니다.

```python
def str2bool(v: str) -> bool:
  """문자열을 부울값으로 변환합니다.
  argparse 기본값만으로는 불리언 처리가 안됩니다.
  """
  if isinstance(v, bool):
      return v
  return v.lower() in ('yes', 'true', 't', '1')
```