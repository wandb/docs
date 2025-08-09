---
title: XGBoost Sweeps
menu:
  tutorials:
    identifier: ko-tutorials-integration-tutorials-xgboost_sweeps
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W&B_Sweeps_with_XGBoost.ipynb" >}}
W&B를 사용해서 기계학습 실험 추적, 데이터셋 버전 관리, 프로젝트 협업을 간편하게 진행하세요.

{{< img src="/images/tutorials/huggingface-why.png" alt="Benefits of using W&B" >}}

트리 기반 모델에서 최고의 성능을 이끌어내기 위해서는
[적절한 하이퍼파라미터 선택](https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f)이 필요합니다.
`early_stopping_rounds`는 얼마나 설정해야 할까요? 트리의 `max_depth` 값은 얼마가 적당할까요?

고차원 하이퍼파라미터 공간에서 최적의 모델을 찾으려면, 파라미터 조합이 급격히 증가해 관리가 어렵습니다.
하이퍼파라미터 스윕(Sweeps)은 다양한 모델을 체계적이고 효율적으로 비교하여 최고의 모델을 찾는 방법을 제공합니다.
스윕은 하이퍼파라미터 값 조합을 자동으로 탐색하여 최적의 값을 찾아내도록 도와줍니다.

이 튜토리얼에서는 W&B를 사용해 XGBoost 모델에 대해 복잡한 하이퍼파라미터 스윕을 3단계로 실행하는 방법을 소개합니다.

아래 그래프에서 미리 결과를 확인해보세요:

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost.png" alt="sweeps_xgboost" >}}

## Sweeps: 개요

W&B에서 하이퍼파라미터 스윕을 실행하는 것은 매우 간단합니다. 3단계만 따라하면 됩니다:

1. **스윕 정의:** 사전(dictionary) 형태의 오브젝트를 생성해서 어떤 파라미터를 탐색할지, 어떤 탐색 전략을 사용할지, 어떤 메트릭을 최적화할지 지정합니다.

2. **스윕 초기화:** 코드 한 줄로 스윕을 초기화하면서 스윕의 설정 정보를 전달합니다:  
`sweep_id = wandb.sweep(sweep_config)`

3. **스윕 에이전트 실행:** 마찬가지로 코드 한 줄로 `wandb.agent()`를 실행하면서 `sweep_id`와 모델 아키텍처를 정의하고 트레이닝하는 함수를 전달합니다:  
`wandb.agent(sweep_id, function=train)`

이렇게 하면 하이퍼파라미터 스윕을 손쉽게 실행할 수 있습니다.

아래 노트북에서 이 세 단계를 더 자세히 다뤄봅니다.

이 노트북을 포크해서 직접 파라미터를 수정해 보거나, 본인만의 데이터셋으로 모델을 실험해보시길 적극 추천합니다.

### 참고 자료
- [Sweeps 문서 →]({{< relref path="/guides/models/sweeps/" lang="ko" >}})
- [커맨드라인에서 실행하기 →](https://www.wandb.com/articles/hyperparameter-tuning-as-easy-as-1-2-3)



```python
!pip install wandb -qU
```


```python

import wandb
wandb.login()
```

## 1. 스윕 정의하기

W&B Sweeps는 아주 적은 코드만으로 원하는 대로 스윕을 세밀하게 구성할 수 있게 해줍니다. 스윕 설정(config)은
[사전(dict) 또는 YAML 파일 형태]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ko" >}})로 정의할 수 있습니다.

함께 주요 옵션을 살펴보겠습니다:
*   **Metric(메트릭):** 스윕에서 최적화하려는 메트릭입니다. name(트레이닝 스크립트에서 반드시 로그되어야 함)과 goal(`maximize` 또는 `minimize`)을 지정할 수 있습니다.
*   **Search Strategy(탐색 전략):** `"method"` 키로 지정합니다. W&B 스윕에서는 다양한 탐색 전략을 지원합니다.
  *   **Grid Search(그리드 검색):** 모든 하이퍼파라미터 조합을 전부 시도합니다.
  *   **Random Search(랜덤 검색):** 무작위로 선택된 하이퍼파라미터 조합을 반복적으로 평가합니다.
  *   **Bayesian Search(베이지안 탐색):** 하이퍼파라미터와 메트릭 점수의 확률적 모델을 만들어, 메트릭이 개선될 확률이 높은 파라미터 조합을 우선적으로 선택합니다. 베이지안 최적화의 목적은 하이퍼파라미터 값을 신중하게 선택하면서도, 실제로 시도해보는 조합의 수는 줄이는 것입니다.
*   **Parameters(파라미터):** 하이퍼파라미터 이름과 해당 값들의 리스트, 범위, 또는 분포를 담고 있는 사전입니다. 각 반복마다 여기서 값을 뽑아서 사용합니다.

자세한 내용은 [스윕 구성 옵션 전체 목록]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ko" >}})을 참고하세요.


```python
sweep_config = {
    "method": "random", # grid 또는 random을 시도해보세요
    "metric": {
      "name": "accuracy",
      "goal": "maximize"   
    },
    "parameters": {
        "booster": {
            "values": ["gbtree","gblinear"]
        },
        "max_depth": {
            "values": [3, 6, 9, 12]
        },
        "learning_rate": {
            "values": [0.1, 0.05, 0.2]
        },
        "subsample": {
            "values": [1, 0.5, 0.3]
        }
    }
}
```

## 2. 스윕 초기화하기

`wandb.sweep`를 호출하면 스윕 컨트롤러(Sweep Controller)가 시작됩니다.
이 컨트롤러는 `parameters` 값을 질의하는 모든 에이전트에게 해당 설정을 전달하고,
에이전트는 `wandb` 로그를 통해 `metrics` 성능을 다시 보고하게 됩니다.


```python
sweep_id = wandb.sweep(sweep_config, project="XGBoost-sweeps")
```

### 트레이닝 프로세스 정의
스윕을 실행하기 전에, 하이퍼파라미터 값을 받아서 모델을 생성하고 트레이닝하는 함수를 정의해야 합니다.

또한, `wandb`를 우리의 스크립트에 통합해야 하며,
주요 컴포넌트는 다음과 같습니다:
*   `wandb.init()`: 새로운 W&B Run을 초기화합니다. 각 run은 트레이닝 스크립트의 한 번의 실행입니다.
*   `run.config`: 모든 하이퍼파라미터를 config 오브젝트에 저장합니다. 이를 통해 [앱](https://wandb.ai)에서 하이퍼파라미터 값별로 run을 정렬하고 비교할 수 있습니다.
*   `run.log()`: 메트릭 및 이미지, 영상, 오디오, HTML, 플롯, 포인트 클라우드 등 커스텀 오브젝트를 로그합니다.

데이터도 다운로드해야 합니다:


```python
!wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
```


```python
# Pima Indians 데이터셋을 위한 XGBoost 모델
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 불러오기
def train():
  config_defaults = {
    "booster": "gbtree",
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 1,
    "seed": 117,
    "test_size": 0.33,
  }

  with wandb.init(config=config_defaults)  as run: # 스윕 중에는 디폴트 값이 오버라이드됩니다
    config = run.config

    # 데이터 불러오고 입력(X), 타겟(y)으로 분리
    dataset = loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
    X, Y = dataset[:, :8], dataset[:, 8]

    # 트레인/테스트셋으로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=config.test_size,
                                                        random_state=config.seed)

    # 트레인셋으로 모델 학습
    model = XGBClassifier(booster=config.booster, max_depth=config.max_depth,
                          learning_rate=config.learning_rate, subsample=config.subsample)
    model.fit(X_train, y_train)

    # 테스트셋에서 예측
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # 예측 평가
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.0%}")
    run.log({"accuracy": accuracy})
```

## 3. 에이전트로 스윕 실행하기

이제 `wandb.agent`를 호출해서 스윕을 시작할 수 있습니다.

`wandb.agent`는 다음 조건을 만족하는 어떤 머신에서도 실행할 수 있습니다:
- `sweep_id`가 준비되어 있고,
- 데이터셋과 `train` 함수가 있으며,

해당 머신이 스윕에 참여하게 됩니다.

> _참고_: `random` 스윕은 기본적으로 무제한 실행됩니다.
계속해서 새로운 파라미터 조합을 시도하므로,
[앱 UI에서 스윕을 직접 종료하지 않는 한]({{< relref path="/guides/models/sweeps/sweeps-ui" lang="ko" >}})
계속 실행됩니다.
완료할 run의 총 `count` 값을 지정하여
에이전트가 완료할 run 수를 제한할 수 있습니다.


```python
wandb.agent(sweep_id, train, count=25)
```

## 결과 시각화

스윕이 종료되면, 이제 결과를 살펴볼 시간입니다.

W&B에서는 다양한 유용한 플롯이 자동으로 생성됩니다.

### 평행좌표 플롯(Parallel coordinates plot)

이 플롯은 하이퍼파라미터 값과 모델 메트릭 간의 관계를 시각화합니다.  
최고의 모델 성능으로 이어지는 하이퍼파라미터 조합을 찾을 때 도움이 됩니다.

이 플롯을 보면, 트리 기반 학습자를 선택하는 것이
선형 모델을 사용한 경우보다 약간 더 성능이 좋은 것으로 나타납니다.

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost2.png" alt="sweeps_xgboost" >}}

### 하이퍼파라미터 중요도 플롯

하이퍼파라미터 중요도 플롯에서는 어떤 하이퍼파라미터 값이
메트릭에 더 큰 영향을 주었는지 확인할 수 있습니다.

상관관계(선형 예측 변수로 가정)와 feature importance(결과에 랜덤 포레스트를 학습했을 때)를 모두 제공하므로,
어떤 파라미터가 가장 영향을 끼쳤는지,
그리고 그 영향이 긍정적인지, 부정적인지 직관적으로 파악할 수 있습니다.

이 차트를 보면, 위의 평행좌표 플롯에서 확인한 경향이
수치적으로도 뒷받침됨을 알 수 있습니다:
최대의 검증 정확도(Validation Accuracy)에 영향을 준 것은 학습기(learner)의 선택이었으며,
`gblinear` 학습자가 `gbtree`에 비해 대체로 성능이 떨어졌습니다.

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost3.png" alt="sweeps_xgboost" >}}

이런 시각화는 하이퍼파라미터 최적화에 드는 시간과 자원을 절약하면서,
정말 중요한 파라미터(및 파라미터 값 구간)를 빠르게 파악하고
더 깊게 탐색할 대상을 선정하는 데 큰 도움이 됩니다.