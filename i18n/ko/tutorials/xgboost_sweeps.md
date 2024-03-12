
# XGBoost Sweeps

[**여기에서 Colab 노트북으로 시도해 보세요 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W&B_Sweeps_with_XGBoost.ipynb)

기계학습 실험 추적, 데이터셋 버전 관리 및 프로젝트 협업을 위해 Weights & Biases를 사용하세요.

<img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

트리 기반 모델의 최상의 성능을 달성하려면 [올바른 하이퍼파라미터를 선택하는 것](https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f)이 필요합니다. 얼마나 많은 `early_stopping_rounds`? 트리의 `max_depth`는 어떻게 되어야 할까요?

최적의 모델을 찾기 위해 고차원 하이퍼파라미터 공간을 탐색하는 것은 매우 번거로울 수 있습니다. 하이퍼파라미터 탐색은 하이퍼파라미터 값의 조합을 자동으로 탐색하여 가장 최적의 값을 찾는 조직적이고 효율적인 방법을 제공하여 모델의 왕좌를 결정합니다.

이 튜토리얼에서는 Weights and Biases를 사용하여 XGBoost 모델에서 3단계로 하이퍼파라미터 탐색을 실행하는 방법을 살펴봅니다.

미리보기를 위해 아래의 그래프를 확인하세요:

![sweeps_xgboost](/images/tutorials/xgboost_sweeps/sweeps_xgboost.png)

## Sweeps: 개요

Weights & Biases를 사용하여 하이퍼파라미터 탐색을 실행하는 것은 매우 쉽습니다. 단지 3단계만 있습니다:

1. **탐색 정의:** 탐색을 지정하는 사전과 같은 오브젝트를 생성하여 어떤 파라미터를 탐색할지, 어떤 검색 전략을 사용할지, 어떤 메트릭을 최적화할지를 지정합니다.

2. **탐색 초기화:** 한 줄의 코드로 탐색을 초기화하고 탐색 구성의 사전을 전달합니다:
`sweep_id = wandb.sweep(sweep_config)`

3. **탐색 에이전트 실행:** 한 줄의 코드로 `wandb.agent()`를 호출하고 `sweep_id`와 모델 아키텍처를 정의하고 트레이닝하는 함수를 전달합니다:
`wandb.agent(sweep_id, function=train)`

그게 다입니다! 하이퍼파라미터 탐색을 실행하는 것은 이게 전부입니다!

아래 노트북에서, 이 3단계를 더 자세히 살펴보겠습니다.

이 노트북을 포크하고, 파라미터를 조정하거나 자신의 데이터셋으로 모델을 시도해 보세요!

### 자료
- [Sweeps 문서 →](https://docs.wandb.com/library/sweeps)
- [커맨드라인에서 실행하기 →](https://www.wandb.com/articles/hyperparameter-tuning-as-easy-as-1-2-3)

```python
!pip install wandb -qU
```

```python
import wandb
wandb.login()
```

## 1. 탐색 정의

Weights & Biases 탐색은 단 몇 줄의 코드로 원하는 대로 탐색을 구성할 수 있는 강력한 기능을 제공합니다. 탐색 구성은 [사전 또는 YAML 파일로 정의될 수 있습니다](https://docs.wandb.ai/guides/sweeps/configuration).

함께 살펴봅시다:
*   **메트릭** – 탐색이 최적화하려는 메트릭입니다. 메트릭은 `name`(이 메트릭은 트레이닝 스크립트에 의해 로그되어야 함)과 `goal`(`maximize` 또는 `minimize`)을 가질 수 있습니다.
*   **검색 전략** – `"method"` 키를 사용하여 지정됩니다. 우리는 탐색과 함께 몇 가지 다른 검색 전략을 지원합니다.
  *   **그리드 검색** – 하이퍼파라미터 값의 모든 조합을 반복합니다.
  *   **랜덤 검색** – 하이퍼파라미터 값의 임의로 선택된 조합을 반복합니다.
  *   **베이지안 탐색** – 하이퍼파라미터를 메트릭 점수의 확률에 매핑하는 확률적 모델을 생성하고 메트릭을 개선할 가능성이 높은 파라미터를 선택합니다. 베이지안 최적화의 목적은 하이퍼파라미터 값을 선택하는 데 더 많은 시간을 소비하지만, 그렇게 함으로써 더 적은 하이퍼파라미터 값을 시도하는 것입니다.
*   **파라미터** – 하이퍼파라미터 이름과 이산 값, 범위 또는 분포를 포함하는 사전으로, 각 반복에서 그 값을 끌어옵니다.

모든 구성 옵션의 목록은 [여기서 찾을 수 있습니다](https://docs.wandb.com/library/sweeps/configuration).

```python
sweep_config = {
    "method": "random", # grid 또는 random을 시도해 보세요
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

## 2. 탐색 초기화

`wandb.sweep`을 호출하면 Sweep 컨트롤러가 시작됩니다 -- 
`parameters`의 설정을 제공하고 `wandb` 로깅을 통해 `metrics`에 대한 성능을 기대하는 중앙 집중식 프로세스입니다.

```python
sweep_id = wandb.sweep(sweep_config, project="XGBoost-sweeps")
```

### 트레이닝 프로세스 정의
탐색을 실행하기 전에,
하이퍼파라미터 값을 입력으로 받아 메트릭을 출력하는 모델을 생성하고 트레이닝하는 함수를 정의해야 합니다.

스크립트에 `wandb`를 통합해야 합니다.
세 가지 주요 구성 요소가 있습니다:
*   `wandb.init()` – 새로운 W&B run을 초기화합니다. 각 run은 트레이닝 스크립트의 단일 실행입니다.
*   `wandb.config` – 모든 하이퍼파라미터를 config 객체에 저장합니다. 이를 통해 [우리 앱](https://wandb.ai)에서 하이퍼파라미터 값으로 run을 정렬하고 비교할 수 있습니다.
*   `wandb.log()` – 메트릭과 사용자 정의 객체를 로그합니다 – 이들은 이미지, 비디오, 오디오 파일, HTML, 플롯, 포인트 클라우드 등일 수 있습니다.

데이터를 다운로드할 필요도 있습니다:

```python
!wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
```

```python
# Pima Indians 데이터셋을 위한 XGBoost 모델
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 로드
def train():
  config_defaults = {
    "booster": "gbtree",
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 1,
    "seed": 117,
    "test_size": 0.33,
  }

  wandb.init(config=config_defaults)  # 탐색 중에 기본값이 덮어쓰여집니다
  config = wandb.config

  # 데이터를 로드하고 예측 변수와 목표 변수로 분할
  dataset = loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
  X, Y = dataset[:, :8], dataset[:, 8]

  # 데이터를 트레이닝 세트와 테스트 세트로 분할
  X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                      test_size=config.test_size,
                                                      random_state=config.seed)

  # 트레이닝 세트에서 모델 학습
  model = XGBClassifier(booster=config.booster, max_depth=config.max_depth,
                        learning_rate=config.learning_rate, subsample=config.subsample)
  model.fit(X_train, y_train)

  # 테스트에서 예측
  y_pred = model.predict(X_test)
  predictions = [round(value) for value in y_pred]

  # 예측 평가
  accuracy = accuracy_score(y_test, predictions)
  print(f"정확도: {accuracy:.0%}")
  wandb.log({"accuracy": accuracy})
```

## 3. 에이전트와 함께 탐색 실행

이제 `wandb.agent`를 호출하여 탐색을 시작합니다.

- `sweep_id`,
- 데이터셋 및 `train` 함수가 있는

모든 기기에서 `wandb.agent`를 호출할 수 있으며 해당 기기는 탐색에 참여할 것입니다!

> _참고_: 기본적으로 `random` 탐색은 계속 실행되며, 새로운 파라미터 조합을 시도합니다 -- 소가 집에 돌아올 때까지 -- 또는 [앱 UI에서 탐색을 끄기](https://docs.wandb.ai/ref/app/features/sweeps)까지입니다. `agent`가 완료해야 할 총 `count`를 제공함으로써 이를 방지할 수 있습니다.

```python
wandb.agent(sweep_id, train, count=25)
```

## 결과 시각화

이제 탐색이 완료되었으니 결과를 살펴볼 시간입니다.

Weights & Biases는 자동으로 여러 유용한 플롯을 생성할 것입니다.

### 병렬 좌표 플롯

이 플롯은 하이퍼파라미터 값과 모델 메트릭을 매핑합니다. 최상의 모델 성능으로 이어진 하이퍼파라미터 조합에 초점을 맞추는 데 유용합니다.

이 플롯은 트리를 학습자로 사용하는 것이 단순한 선형 모델을 학습자로 사용하는 것보다 약간,
하지만 결정적으로,
성능이 더 뛰어나다는 것을 시사합니다.

![sweeps_xgboost](/images/tutorials/xgboost_sweeps/sweeps_xgboost2.png)

### 하이퍼파라미터 중요도 플롯

하이퍼파라미터 중요도 플롯은 메트릭에 가장 큰 영향을 준 하이퍼파라미터 값을 보여줍니다.

우리는 상관 관계(선형 예측 변수로 취급)와 특성 중요도(결과에 대한 랜덤 포레스트를 훈련한 후)를 보고하여 어떤 파라미터가 가장 큰 영향을 미쳤는지,
그리고 그 영향이 긍정적이었는지 부정적이었는지를 볼 수 있습니다.

이 차트를 읽으면서 위의 병렬 좌표 차트에서 알아챈 경향을 정량적으로 확인할 수 있습니다:
검증 정확도에 가장 큰 영향을 미친 것은 학습자의 선택이었으며, `gblinear` 학습자는 일반적으로 `gbtree` 학습자보다 나빴습니다.

![sweeps_xgboost](/images/tutorials/xgboost_sweeps/sweeps_xgboost3.png)

이 시각화들은 고가의 하이퍼파라미터 최적화를 실행하는 데 드는 시간과 자원을 절약하는 데 도움이 될 수 있으며, 가장 중요하고 따라서 추가 탐색할 가치가 있는 파라미터(및 값 범위)에 초점을 맞추는 데 도움이 됩니다.