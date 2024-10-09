---
title: Tutorial: Define, initialize, and run a sweep
description: Sweeps 퀵스타트는 스윕을 정의하고, 초기화하고, 실행하는 방법을 보여줍니다. 주요 단계를 네 가지로 나눌 수 있습니다.
displayed_sidebar: default
---

이 페이지에서는 스윕을 정의하고 초기화하며 실행하는 방법을 설명합니다. 네 가지 주요 단계가 있습니다:

1. [트레이닝 코드 설정](#set-up-your-training-code)
2. [스윕 구성을 통해 검색 공간 정의](#define-the-search-space-with-a-sweep-configuration)
3. [스윕 초기화](#initialize-the-sweep)
4. [스윕 에이전트 시작](#start-the-sweep)

다음 코드를 Jupyter 노트북 또는 Python 스크립트에 복사하여 붙여넣으세요:

```python
# W&B Python 라이브러리를 가져오고 W&B에 로그인합니다.
import wandb

wandb.login()

# 1: 목적/트레이닝 함수 정의
def objective(config):
    score = config.x**3 + config.y
    return score

def main():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})

# 2: 검색 공간 정의
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

# 3: 스윕 시작
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
```

다음 섹션에서는 코드 샘플의 각 단계를 설명합니다.

## 트레이닝 코드 설정

`wandb.config`에서 하이퍼파라미터 값을 가져와서 모델을 트레이닝하고 메트릭을 반환하는 트레이닝 함수를 정의하세요.

출력을 저장할 W&B Run의 프로젝트 이름을 선택적으로 지정할 수 있습니다 (프로젝트 파라미터는 [`wandb.init`](../../ref/python/init.md)에서). 프로젝트가 지정되지 않으면, run은 "미분류" 프로젝트에 저장됩니다.

:::tip
스윕과 run은 동일한 프로젝트 내에 있어야 합니다. 따라서 W&B를 초기화할 때 제공하는 이름은 스윕을 초기화할 때 제공하는 프로젝트 이름과 일치해야 합니다.
:::

```python
# 1: 목적/트레이닝 함수 정의
def objective(config):
    score = config.x**3 + config.y
    return score


def main():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})
```

## 스윕 구성을 통해 검색 공간 정의

사전 내에서 스윕할 하이퍼파라미터를 지정하세요. 구성 옵션에 대한 자세한 내용은 [스윕 구성 정의](./define-sweep-configuration.md)를 참조하세요.

다음 예제는 랜덤 검색 (`'method':'random'`)을 사용하는 스윕 구성을 보여줍니다. 스윕은 구성에 나열된 배치 크기, 에포크, 학습률에 대한 무작위 값 세트를 무작위로 선택합니다.

스윕 동안, W&B는 메트릭 키 (`metric`)에 지정된 메트릭을 최대화합니다. 다음 예제에서는 W&B가 검증 정확도 (`'val_acc'`)를 최대화합니다 (`'goal':'maximize'`).

```python
# 2: 검색 공간 정의
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}
```

## 스윕 초기화

W&B는 클라우드(기본)에서, 또는 로컬에서 하나 이상의 머신을 통해 스윕을 관리하기 위해 _Sweep Controller_ 를 사용합니다. Sweep Controller에 대한 정보는 [Search and stop algorithms locally](./local-controller.md)를 참조하세요.

스윕을 초기화할 때 스윕 식별 번호가 반환됩니다:

```python
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
```

스윕 초기화에 대한 자세한 정보는 [스윕 초기화](./initialize-sweeps.md)를 참조하세요.

## 스윕 시작

[`wandb.agent`](../../ref/python/agent.md) API 호출을 사용하여 스윕을 시작하세요.

```python
wandb.agent(sweep_id, function=main, count=10)
```

## 결과 시각화 (선택 사항)

프로젝트를 열어 W&B 앱 대시보드에서 실시간 결과를 확인하세요. 몇 번의 클릭만으로 [평행 좌표 플롯](../app/features/panels/parallel-coordinates.md), [파라미터 중요도 분석](../app/features/panels/parameter-importance.md) 등의 풍부하고 인터랙티브한 차트를 구성할 수 있습니다.

![Sweeps Dashboard example](/images/sweeps/quickstart_dashboard_example.png)

결과를 시각화하는 방법에 대한 자세한 정보는 [스윕 결과 시각화](./visualize-sweep-results.md)를 참조하세요. 예시 대시보드는 이 샘플 [Sweeps Project](https://wandb.ai/anmolmann/pytorch-cnn-fashion/sweeps/pmqye6u3)를 참조하세요.

## 에이전트 중지 (선택 사항)

터미널에서 `Ctrl+c`를 눌러 스윕 에이전트가 현재 실행 중인 run을 중지하세요. 에이전트를 종료하려면, run이 중지된 후 `Ctrl+c`를 다시 누르세요.