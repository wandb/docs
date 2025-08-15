---
title: '튜토리얼: 스윕 정의, 초기화, 실행하기'
description: Sweeps 퀵스타트에서는 스윕을 정의하고, 초기화하며, 실행하는 방법을 안내합니다. 주요 단계는 네 가지입니다.
menu:
  default:
    identifier: ko-guides-models-sweeps-walkthrough
    parent: sweeps
weight: 1
---

이 페이지에서는 스윕을 정의, 초기화, 실행하는 방법을 안내합니다. 주요 단계는 네 가지입니다.

1. [트레이닝 코드 준비하기]({{< relref path="#set-up-your-training-code" lang="ko" >}})
2. [스윕 구성으로 탐색 공간 정의하기]({{< relref path="#define-the-search-space-with-a-sweep-configuration" lang="ko" >}})
3. [스윕 초기화하기]({{< relref path="#initialize-the-sweep" lang="ko" >}})
4. [스윕 에이전트 시작하기]({{< relref path="#start-the-sweep" lang="ko" >}})

아래 코드를 Jupyter Notebook이나 Python 스크립트에 복사하여 붙여넣으세요.

```python 
# W&B Python 라이브러리 임포트 및 W&B 로그인
import wandb

# 1: objective/트레이닝 함수 정의
def objective(config):
    score = config.x**3 + config.y
    return score

def main():
    with wandb.init(project="my-first-sweep") as run:
        score = objective(run.config)
        run.log({"score": score})

# 2: 탐색 공간 정의
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

아래 섹션에서는 코드 예시의 각 단계를 자세히 설명합니다.

## 트레이닝 코드 준비하기
`wandb.Run.config`에서 하이퍼파라미터 값을 받아 모델을 학습하고, 메트릭을 반환하는 트레이닝 함수를 정의하세요.

원한다면 W&B Run의 결과가 저장될 프로젝트 이름을 지정할 수 있습니다([`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}})의 project 파라미터 이용). 프로젝트를 설정하지 않으면 해당 run은 "Uncategorized" 프로젝트에 저장됩니다.

{{% alert %}}
스윕과 run은 반드시 같은 프로젝트에서 실행되어야 합니다. 따라서 W&B를 초기화할 때 지정한 이름과 스윕을 초기화할 때 지정한 프로젝트 이름이 일치해야 합니다.
{{% /alert %}}

```python
# 1: objective/트레이닝 함수 정의
def objective(config):
    score = config.x**3 + config.y
    return score


def main():
    with wandb.init(project="my-first-sweep") as run:
        score = objective(run.config)
        run.log({"score": score})
```

## 스윕 구성으로 탐색 공간 정의하기

스윕할 하이퍼파라미터들을 딕셔너리 형태로 지정합니다. 더 많은 설정 옵션은 [스윕 구성 정의하기]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ko" >}})를 참고하세요.

아래 예시는 'random' 메소드를 활용한 스윕 구성입니다. 구성에 지정된 배치 크기, 에포크, 러닝레이트 값을 무작위로 선택해 실험합니다.

`"goal": "minimize"`가 지정된 경우, W&B는 `metric` 키에 지정된 메트릭을 최소화하도록 최적화합니다. 여기서는 `score`(`"name": "score"`)를 최소화하는 것이 목표입니다.

```python
# 2: 탐색 공간 정의
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

W&B는 클라우드(standard), 로컬(local) 등 다양한 환경에서 스윕을 관리하기 위해 _Sweep Controller_를 사용합니다. Sweep Controller에 대한 자세한 내용은 [로컬에서 탐색 및 정지 알고리즘 실행]({{< relref path="./local-controller.md" lang="ko" >}})을 참고하세요.

스윕을 초기화하면 고유한 스윕 식별 번호가 반환됩니다.

```python
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
```

스윕 초기화에 대한 자세한 내용은 [스윕 초기화하기]({{< relref path="./initialize-sweeps.md" lang="ko" >}})에서 확인하세요.

## 스윕 에이전트 시작하기

[`wandb.agent`]({{< relref path="/ref/python/sdk/functions/agent.md" lang="ko" >}}) API를 사용해 스윕을 시작하세요.

```python
wandb.agent(sweep_id, function=main, count=10)
```

## 결과 시각화하기 (선택 사항)

프로젝트를 열면 W&B 앱 대시보드에서 실시간 결과를 확인할 수 있습니다. 몇 번의 클릭만으로 [평행 좌표 플롯]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ko" >}}), [파라미터 중요도 분석]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ko" >}}), [그 외 다양한 차트 유형]({{< relref path="/guides/models/app/features/panels/" lang="ko" >}}) 등 다양한 인터랙티브 차트를 쉽게 만들 수 있습니다.

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="Sweeps Dashboard example" >}}

결과 시각화 방법에 대한 자세한 내용은 [스윕 결과 시각화하기]({{< relref path="./visualize-sweep-results.md" lang="ko" >}})를 참고하세요. 대시보드 예시는 이 샘플 [Sweeps Project](https://wandb.ai/anmolmann/pytorch-cnn-fashion/sweeps/pmqye6u3)를 확인해보세요.

## 에이전트 중지하기 (선택 사항)

터미널에서 `Ctrl+C`를 눌러 현재 run을 중지하세요. 한 번 더 누르면 에이전트가 완전히 종료됩니다.

```