---
title: 'Tutorial: Define, initialize, and run a sweep'
description: Sweeps 퀵스타트 는 스윕을 정의, 초기화 및 실행하는 방법을 보여줍니다. 주요 단계는 다음과 같습니다.
menu:
  default:
    identifier: ko-guides-models-sweeps-walkthrough
    parent: sweeps
weight: 1
---

이 페이지에서는 스윕을 정의하고 초기화하고 실행하는 방법을 보여줍니다. 주요 단계는 네 가지입니다.

1. [트레이닝 코드 설정]({{< relref path="#set-up-your-training-code" lang="ko" >}})
2. [스윕 구성으로 검색 공간 정의]({{< relref path="#define-the-search-space-with-a-sweep-configuration" lang="ko" >}})
3. [스윕 초기화]({{< relref path="#initialize-the-sweep" lang="ko" >}})
4. [스윕 에이전트 시작]({{< relref path="#start-the-sweep" lang="ko" >}})

다음 코드를 Jupyter Notebook 또는 Python 스크립트에 복사하여 붙여넣으세요.

```python
# W&B Python 라이브러리를 임포트하고 W&B에 로그인합니다
import wandb

wandb.login()

# 1: objective/트레이닝 함수 정의
def objective(config):
    score = config.x**3 + config.y
    return score

def train():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})

if __name__ == '__main__':
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

wandb.agent(sweep_id, function=train, count=10)
```

다음 섹션에서는 코드 샘플의 각 단계를 분석하고 설명합니다.

## 트레이닝 코드 설정
`wandb.config`에서 하이퍼파라미터 값을 가져와 모델을 트레이닝하고 메트릭을 반환하는 트레이닝 함수를 정의합니다.

선택적으로 W&B Run의 출력을 저장할 프로젝트 이름을 제공합니다([`wandb.init`]({{< relref path="/ref/python/init.md" lang="ko" >}})의 project 파라미터). 프로젝트가 지정되지 않은 경우 run은 "Uncategorized" 프로젝트에 배치됩니다.

{{% alert %}}
스윕과 run은 모두 동일한 project에 있어야 합니다. 따라서 W&B를 초기화할 때 제공하는 이름은 스윕을 초기화할 때 제공하는 project 이름과 일치해야 합니다.
{{% /alert %}}

```python
# 1: objective/트레이닝 함수 정의
def objective(config):
    score = config.x**3 + config.y
    return score


def train():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})
```

## 스윕 구성으로 검색 공간 정의

사전에서 스윕할 하이퍼파라미터를 지정합니다. 구성 옵션은 [스윕 구성 정의]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ko" >}})을 참조하세요.

다음 예제는 랜덤 검색(`'method':'random'`)을 사용하는 스윕 구성을 보여줍니다. 스윕은 배치 크기, 에포크 및 학습률에 대해 구성에 나열된 임의의 값 집합을 무작위로 선택합니다.

W&B는 `metric` 키에 `“goal": "minimize"`가 연결되어 있을 때 지정된 메트릭을 최소화합니다. 이 경우 W&B는 메트릭 `score`(`"name": "score"`)를 최소화하도록 최적화합니다.

```python
if __name__ == '__main__':
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

W&B는 클라우드(표준), 로컬(로컬)에서 하나 이상의 머신에서 스윕을 관리하기 위해 _Sweep Controller_ 를 사용합니다. Sweep Controller에 대한 자세한 내용은 [로컬에서 검색 및 중지 알고리즘]({{< relref path="./local-controller.md" lang="ko" >}})을 참조하세요.

스윕을 초기화하면 스윕 식별 번호가 반환됩니다.

```python
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
```

스윕 초기화에 대한 자세한 내용은 [스윕 초기화]({{< relref path="./initialize-sweeps.md" lang="ko" >}})을 참조하세요.

## 스윕 시작

[`wandb.agent`]({{< relref path="/ref/python/agent.md" lang="ko" >}}) API 호출을 사용하여 스윕을 시작합니다.

```python
    wandb.agent(sweep_id, function=train, count=10)
```

## 결과 시각화 (선택 사항)

프로젝트를 열어 W&B App 대시보드에서 실시간 결과를 확인하세요. 몇 번의 클릭만으로 [평행 좌표 플롯]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ko" >}}), [파라미터 중요도 분석]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ko" >}}) 및 [기타]({{< relref path="/guides/models/app/features/panels/" lang="ko" >}})와 같은 풍부한 인터랙티브 차트를 구성합니다.

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="Sweeps Dashboard example" >}}

결과 시각화 방법에 대한 자세한 내용은 [스윕 결과 시각화]({{< relref path="./visualize-sweep-results.md" lang="ko" >}})를 참조하세요. 대시보드 예제는 샘플 [Sweeps Project](https://wandb.ai/anmolmann/pytorch-cnn-fashion/sweeps/pmqye6u3)를 참조하세요.

## 에이전트 중지 (선택 사항)

터미널에서 `Ctrl+C`를 눌러 현재 run을 중지합니다. 다시 누르면 에이전트가 종료됩니다.
