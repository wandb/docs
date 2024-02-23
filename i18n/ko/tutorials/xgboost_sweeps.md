
# XGBoost 스윕

[**Colab 노트북에서 시도해보세요 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W&B_Sweeps_with_XGBoost.ipynb)

Weights & Biases를 사용하여 머신 러닝 실험 추적, 데이터세트 버전 관리 및 프로젝트 협업을 수행하세요.

<img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

트리 기반 모델의 최적 성능을 얻기 위해서는 [올바른 하이퍼파라미터 선택](https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f)이 필요합니다. `early_stopping_rounds`는 얼마나 많아야 할까요? 트리의 `max_depth`는 어떻게 되어야 할까요?

최적의 모델을 찾기 위해 고차원 하이퍼파라미터 공간을 탐색하는 것은 매우 번거로울 수 있습니다. 하이퍼파라미터 스윕은 모델의 왕좌를 놓고 경쟁하는 정리된 효율적인 방법을 제공합니다. 하이퍼파라미터 값의 조합을 자동으로 탐색하여 가장 최적의 값들을 찾음으로써 이를 가능하게 합니다.

이 튜토리얼에서는 Weights and Biases를 사용하여 XGBoost 모델에서 세 가지 쉬운 단계로 고급 하이퍼파라미터 스윕을 실행하는 방법을 살펴볼 것입니다.

티저로 아래의 플롯을 확인해보세요:

![sweeps_xgboost](/images/tutorials/xgboost_sweeps/sweeps_xgboost.png)

## 스윕: 개요

Weights & Biases에서 하이퍼파라미터 스윕을 실행하는 것은 매우 쉽습니다. 단지 3단계만 있습니다:

1. **스윕 정의:** 스윕을 지정하는 사전 같은 개체를 생성하여 어떤 파라미터를 탐색할지, 어떤 검색 전략을 사용할지, 어떤 메트릭을 최적화할지를 지정합니다.

2. **스윕 초기화:** 한 줄의 코드로 스윕을 초기화하고 스윕 구성의 사전을 전달합니다:
`sweep_id = wandb.sweep(sweep_config)`

3. **스윕 에이전트 실행:** 또한 한 줄의 코드로, 모델 아키텍처를 정의하고 학습하는 함수와 함께 `sweep_id`를 전달하여 `wandb.agent()`를 호출합니다:
`wandb.agent(sweep_id, function=train)`

그리고 끝! 하이퍼파라미터 스윕을 실행하는 것이 전부입니다!

아래 노트북에서 이 3단계를 더 자세히 살펴보겠습니다.

이 노트북을 포크하고 파라미터를 조정하거나 자신의 데이터세트로 모델을 시도해보시기를 권장합니다!

### 자료
- [스윕 문서 →](https://docs.wandb.com/library/sweeps)
- [명령 줄에서 실행하기 →](https://www.wandb.com/articles/hyperparameter-tuning-as-easy-as-1-2-3)

```python
!pip install wandb -qU
```

```python
import wandb
wandb.login()
```

## 1. 스윕 정의

Weights & Biases 스윕은 몇 줄의 코드로 원하는 대로 스윕을 정확하게 구성할 수 있는 강력한 레버를 제공합니다. 스윕 구성은 [사전이나 YAML 파일로 정의](https://docs.wandb.ai/guides/sweeps/configuration)될 수 있습니다.

함께 살펴볼 몇 가지가 있습니다:
*   **메트릭** – 스윕이 최적화하려고 시도하는 메트릭입니다. 메트릭은 `name`(이 메트릭은 학습 스크립트에 의해 로그되어야 함)과 `goal`(`maximize` 또는 `minimize`)을 가질 수 있습니다.
*   **검색 전략** – `"method"` 키를 사용하여 지정됩니다. 스윕에서 여러 가지 검색 전략을 지원합니다.
  *   **그리드 검색** – 하이퍼파라미터 값의 모든 조합을 반복합니다.
  *   **랜덤 검색** – 하이퍼파라미터 값의 임의로 선택된 조합을 반복합니다.
  *   **베이지안 검색** – 하이퍼파라미터를 메트릭 점수의 확률로 매핑하는 확률적 모델을 생성하고 메트릭을 개선할 가능성이 높은 매개 변수를 선택합니다. 베이지안 최적화의 목표는 하이퍼파라미터 값을 선택하는 데 더 많은 시간을 소비하는 것이지만 그렇게 함으로써 더 적은 하이퍼파라미터 값을 시도하는 것입니다.
*   **파라미터** – 하이퍼파라미터 이름과 각 반복에서 값의 이산 값, 범위 또는 분포를 당기는 사전입니다.

모든 구성 옵션 목록은 [여기](https://docs.wandb.com/library/sweeps/configuration)에서 찾을 수 있습니다.

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

## 2. 스윕 초기화

`wandb.sweep`을 호출하면 스윕 컨트롤러가 시작됩니다 --
`parameters`의 설정을 제공하고 `metrics`에 대한 성능을 `wandb` 로깅을 통해 기대하는 중앙 집중식 프로세스입니다.

```python
sweep_id = wandb.sweep(sweep_config, project="XGBoost-sweeps")
```

### 학습 프로세스 정의
스윕을 실행하기 전에,
하이퍼파라미터 값이 주어지고 메트릭이 출력되는 함수를 정의해야 합니다 -- 모델을 생성하고 학습하는 함수입니다.

스크립트에 `wandb`도 통합해야 합니다.
세 가지 주요 구성 요소가 있습니다:
*   `wandb.init()` – 새로운 W&B 실행을 초기화합니다. 각 실행은 학습 스크립트의 단일 실행입니다.
*   `wandb.config` – 모든 하이퍼파라미터를 config 객체에 저장합니다. 이렇게 하면 [우리 앱](https://wandb.ai)을 사용하여 하이퍼파라미터 값에 따라 실행을 정렬하고 비교할 수 있습니다.
*   `wandb.log()` – 메트릭과 사용자 정의 객체를 로그합니다 – 이들은 이미지, 비디오, 오디오 파일, HTML, 플롯, 포인트 클라우드 등일 수 있습니다.

데이터도 다운로드해야 합니다:

```python
!wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
```

```python
# Pima Indians 데이터세트용 XGBoost 모델
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

  wandb.init(config=config_defaults)  # 스윕 중에 기본값이 덮어씌워집니다
  config = wandb.config

  # 데이터를 로드하고 예측기와 목표로 분할
  dataset = loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
  X, Y = dataset[:, :8], dataset[:, 8]

  # 데이터를 학습 및 테스트 세트로 분할
  X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                      test_size=config.test_size,
                                                      random_state=config.seed)

  # 학습 데이터에 모델 적합
  model = XGBClassifier(booster=config.booster, max_depth=config.max_depth,
                        learning_rate=config.learning_rate, subsample=config.subsample)
  model.fit(X_train, y_train)

  # 테스트에 대한 예측 수행
  y_pred = model.predict(X_test)
  predictions = [round(value) for value in y_pred]

  # 예측 평가
  accuracy = accuracy_score(y_test, predictions)
  print(f"Accuracy: {accuracy:.0%}")
  wandb.log({"accuracy": accuracy})
```

## 3. 에이전트와 스윕 실행

이제, 스윕을 시작하기 위해 `wandb.agent`를 호출합니다.

- `sweep_id`,
- 데이터세트와 `train` 함수가 있는

어떤 기계에서든 W&B에 로그인되어 있고 그 기계가 스윕에 참여할 것입니다!

> _참고_: `random` 스윕은 기본적으로 영원히 실행되며, 새로운 파라미터 조합을 시도하거나 [앱 UI에서 스윕을 끄기](https://docs.wandb.ai/ref/app/features/sweeps) 전까지 계속됩니다. `agent`가 완료해야 할 총 `count`를 제공함으로써 이를 방지할 수 있습니다.

```python
wandb.agent(sweep_id, train, count=25)
```

## 결과 시각화

스윕이 완료되었으니 결과를 살펴볼 시간입니다.

Weights & Biases는 자동으로 여러 유용한 플롯을 생성해줍니다.

### 병렬 좌표 플롯

이 플롯은 하이퍼파라미터 값을 모델 메트릭에 매핑합니다. 최적의 모델 성능으로 이어지는 하이퍼파라미터 조합에 집중하는 데 유용합니다.

이 플롯은 우리의 학습자로 트리를 사용하는 것이 단순한 선형 모델을 학습자로 사용하는 것보다 약간,
하지만 현저하게,
성능이 우수함을 나타냅니다.

![sweeps_xgboost](/images/tutorials/xgboost_sweeps/sweeps_xgboost2.png)

### 하이퍼파라미터 중요도 플롯

하이퍼파라미터 중요도 플롯은 메트릭에 가장 큰 영향을 미친 하이퍼파라미터 값을 보여줍니다.

우리는 상관 관계(선형 예측 변수로 처리)와 특징 중요도(결과에 대해 랜덤 포레스트를 학습한 후)를 보고하므로 어떤 매개 변수가 가장 큰 영향을 미쳤는지,
그리고 그 영향이 긍정적이었는지 부정적이었는지 볼 수 있습니다.

이 차트를 읽으면서 위의 병렬 좌표 차트에서 주목한 경향을 정량적으로 확인할 수 있습니다:
검증 정확도에 가장 큰 영향을 미친 것은 학습자의 선택이었으며, `gblinear` 학습자는 일반적으로 `gbtree` 학습자보다 나빴습니다.

![sweeps_xgboost](/images/tutorials/xgboost_sweeps/sweeps_xgboost3.png)

이 시각화는 비용이 많이 드는 하이퍼파라미터 최적화를 실행하는 데 시간과 자원을 절약하는 데 도움이 될 수 있으며, 가장 중요하고 따라서 추가 탐색할 가치가 있는 매개 변수(및 값 범위)에 집중함으로써 그렇게 합니다.