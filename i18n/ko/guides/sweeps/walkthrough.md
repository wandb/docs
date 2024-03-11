---
description: Sweeps quickstart shows how to define, initialize, and run a sweep. There
  are four main steps
displayed_sidebar: default
---

# 워크스루

<head>
  <title>Sweeps 워크스루</title>
</head>

이 페이지는 스윕을 정의, 초기화 및 실행하는 방법을 보여줍니다. 네 가지 주요 단계가 있습니다:

1. [트레이닝 코드 설정하기](#트레이닝-코드-설정하기)
2. [스윕 구성으로 검색 공간 정의하기](#스윕-구성으로-검색-공간-정의하기)
3. [스윕 초기화하기](#스윕-초기화하기)
4. [스윕 에이전트 시작하기](#스윕-에이전트-시작하기)

다음 코드를 Jupyter Notebook이나 Python 스크립트에 복사하여 붙여넣으세요:

```python 
# W&B Python 라이브러리를 임포트하고 W&B에 로그인하기
import wandb

wandb.login()

# 1: 목표/트레이닝 함수 정의하기
def objective(config):
    score = config.x**3 + config.y
    return score

def main():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})

# 2: 검색 공간 정의하기
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

# 3: 스윕 시작하기
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
```

다음 섹션에서는 코드 샘플의 각 단계를 상세히 설명합니다.

## 트레이닝 코드 설정하기
`wandb.config`에서 하이퍼파라미터 값을 입력으로 받아 모델을 트레이닝하고 메트릭을 반환하는 트레이닝 함수를 정의합니다.

W&B Run의 출력을 저장하고자 하는 프로젝트의 이름을 선택적으로 제공할 수 있습니다(`wandb.init`의 프로젝트 파라미터). 프로젝트가 지정되지 않으면 run은 "Uncategorized" 프로젝트에 저장됩니다.

:::tip
스윕과 run은 같은 프로젝트에 있어야 합니다. 따라서, W&B를 초기화할 때 제공하는 이름은 스윕을 초기화할 때 제공하는 프로젝트의 이름과 일치해야 합니다.
:::

```python
# 1: 목표/트레이닝 함수 정의하기
def objective(config):
    score = config.x**3 + config.y
    return score

def main():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})
```

## 스윕 구성으로 검색 공간 정의하기
사전 내에서 스윕할 하이퍼파라미터를 지정합니다. 구성 옵션에 대한 자세한 정보는 [스윕 구성 정의하기](./define-sweep-configuration.md)를 참조하세요.

다음 예제는 배치 크기, 에포크 및 학습률에 대한 구성에 나열된 무작위 값을 무작위로 선택하는 랜덤 검색(`'method':'random'`)을 사용하는 스윕 구성을 보여줍니다.

스윕 전체에서 W&B는 메트릭 키(`metric`)에 지정된 메트릭을 최대화합니다. 다음 예에서 W&B는 검증 정확도(`'val_acc'`)를 최대화(`'goal':'maximize'`)합니다.


```python
# 2: 검색 공간 정의하기
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}
```

## 스윕 초기화하기

W&B는 클라우드(표준) 또는 하나 이상의 기계에서 로컬로 스윕을 관리하기 위해 _Sweep 컨트롤러_를 사용합니다. Sweep 컨트롤러에 대한 자세한 정보는 [로컬에서 검색 및 정지 알고리즘](./local-controller.md)을 참조하세요.

스윕을 초기화하면 스윕 식별 번호가 반환됩니다:

```python
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
```

스윕 초기화에 대한 자세한 정보는 [스윕 초기화하기](./initialize-sweeps.md)를 참조하세요.

## 스윕 시작하기

[`wandb.agent`](../../ref/python/agent.md) API 호출을 사용하여 스윕을 시작합니다.

```python
wandb.agent(sweep_id, function=main, count=10)
```

## 결과 시각화하기 (선택 사항)

W&B App 대시보드에서 실시간 결과를 확인하세요. 몇 번의 클릭만으로 [평행 좌표 플롯](../app/features/panels/parallel-coordinates.md), [파라미터 중요도 분석](../app/features/panels/parameter-importance.md), [그리고 더](../app/features/panels/intro.md)와 같은 풍부하고 인터랙티브한 차트를 생성할 수 있습니다.

![Sweeps 대시보드 예시](/images/sweeps/quickstart_dashboard_example.png)

결과 시각화에 대한 자세한 정보는 [스윕 결과 시각화하기](./visualize-sweep-results.md)를 참조하세요. 예시 대시보드는 이 샘플 [Sweeps 프로젝트](https://wandb.ai/anmolmann/pytorch-cnn-fashion/sweeps/pmqye6u3)를 확인하세요.

## 에이전트 중지하기 (선택 사항)

터미널에서 `Ctrl+c`를 눌러 현재 스윕 에이전트가 실행중인 run을 중지하세요. run이 중지된 후 다시 `Ctrl+c`를 누르면 에이전트가 종료됩니다.