---
title: XGBoost Sweeps
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W&B_Sweeps_with_XGBoost.ipynb'/>

Weights & Biases를 사용하여 기계학습 실험 추적, 데이터셋 버전 관리, 프로젝트 협업을 진행하세요.

![](/images/tutorials/huggingface-why.png)

트리 기반 모델의 최상의 성능을 얻으려면
[올바른 하이퍼파라미터를 선택](https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f)해야 합니다.
얼마나 많은 `early_stopping_rounds`가 필요할까요? 트리의 `max_depth`는 얼마로 설정해야 할까요?

고성능 모델을 찾기 위해 고차원 하이퍼파라미터 공간을 탐색하는 것은 매우 빠르게 감당하기 어려워질 수 있습니다.
하이퍼파라미터 탐색은 모델들의 전투로얄을 조직적이고 효율적으로 수행하여 승자를 가리는 방법을 제공합니다.
이를 통해 하이퍼파라미터 값의 조합을 자동으로 탐색하여 최적의 값을 찾을 수 있습니다.

이 튜토리얼에서는 Weights & Biases를 사용하여 XGBoost 모델에서 고급 하이퍼파라미터 탐색을 3단계로 수행하는 방법을 살펴보겠습니다.

아래의 플롯을 먼저 확인해보세요:

![sweeps_xgboost](/images/tutorials/xgboost_sweeps/sweeps_xgboost.png)

## Sweeps: 개요

Weights & Biases를 사용하여 하이퍼파라미터 탐색을 실행하는 것은 매우 간단합니다. 단 3단계만 있습니다:

1. **스윕 정의:** 스윕을 정의하는 사전과 같은 오브젝트를 생성합니다: 탐색할 파라미터, 사용할 탐색 전략, 최적화할 메트릭을 지정합니다.

2. **스윕 초기화:** 한 줄의 코드로 스윕을 초기화하고 스윕 설정의 사전을 전달합니다:
   `sweep_id = wandb.sweep(sweep_config)`

3. **스윕 에이전트 실행:** 역시 한 줄의 코드로 완료됩니다. `wandb.agent()`를 호출하고, 모델 아키텍처를 정의하고 트레이닝하는 함수를 전달합니다:
   `wandb.agent(sweep_id, function=train)`

그리고 보세요! 하이퍼파라미터 탐색을 실행하는 것은 그것이 전부입니다!

아래의 노트북에서 이 3단계를 더 자세히 알아보겠습니다.

이 노트북을 포크하고 파라미터를 조정하거나 자신의 데이터셋으로 모델을 시도해보는 것을 적극 권장합니다!

### 리소스
- [Sweeps 문서 →](/guides/sweeps)
- [커맨드라인에서 시작하기 →](https://www.wandb.com/articles/hyperparameter-tuning-as-easy-as-1-2-3)

```python
!pip install wandb -qU
```

```python
import wandb
wandb.login()
```

## 1. 스윕 정의

Weights & Biases의 sweeps는 몇 줄의 코드만으로 원하는 방식으로 스윕을 구성할 수 있는 강력한 레버를 제공합니다. sweeps 설정은
[사전이나 YAML 파일](/guides/sweeps/define-sweep-configuration)로 정의할 수 있습니다.

몇 가지를 함께 살펴보겠습니다:
*   **메트릭** – sweeps가 최적화하려고 시도하는 메트릭입니다. 메트릭은 `name` (이 메트릭은 트레이닝 스크립트에서 로그되어야 함)과 `goal` (`maximize` 또는 `minimize`)을 가질 수 있습니다.
*   **탐색 전략** – `"method"` 키를 사용하여 지정합니다. Sweeps는 여러 가지 탐색 전략을 지원합니다.
  *   **그리드 검색** – 하이퍼파라미터 값의 모든 조합을 반복합니다.
  *   **랜덤 검색** – 무작위로 선택된 하이퍼파라미터 값의 조합을 반복합니다.
  *   **베이지안 탐색** – 하이퍼파라미터를 메트릭 점수의 확률로 맵핑하는 확률 모델을 생성하고, 메트릭을 개선할 확률이 높은 파라미터를 선택합니다. 베이지안 최적화의 목표는 하이퍼파라미터 값을 선택하는 데 더 많은 시간을 들이는 것이지만, 그렇게 함으로써 더 적은 하이퍼파라미터 값을 시도하는 것입니다.
*   **파라미터** – 각 반복에서 해당 값을 추출할 하이퍼파라미터 이름과 이산 값, 범위 또는 분포를 포함하는 사전.

세부사항은 [모든 스윕 설정 옵션 목록](/guides/sweeps/define-sweep-configuration)을 참조하세요.

```python
sweep_config = {
    "method": "random", # grid 또는 random 시도
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
중앙화된 프로세스로, 이를 쿼리하는 누구에게든 `parameters`의 설정을 제공하고
`wandb` 로그를 통해 `metrics`의 성능을 반환하기를 기대합니다.

```python
sweep_id = wandb.sweep(sweep_config, project="XGBoost-sweeps")
```

### 트레이닝 프로세스 정의
스윕을 실행하기 전에,
모델을 생성하고 트레이닝하는 함수를 정의해야 합니다 --
하이퍼파라미터 값을 받아들이고 메트릭을 내보내는 함수입니다.

스크립트에 `wandb`도 통합되어야 합니다.
세 가지 주요 구성요소가 있습니다:
*   `wandb.init()` – 새로운 W&B run을 초기화합니다. 각 run은 트레이닝 스크립트의 단일 실행입니다.
*   `wandb.config` – 모든 하이퍼파라미터를 설정 오브젝트에 저장합니다. 이를 통해 [우리의 앱](https://wandb.ai)을 사용하여 하이퍼파라미터 값으로 run을 정렬하고 비교할 수 있습니다.
*   `wandb.log()` – 메트릭과 커스텀 오브젝트를 로그합니다 – 이는 이미지, 비디오, 오디오 파일, HTML, 플롯, 포인트 클라우드 등이 될 수 있습니다.

또한 데이터를 다운로드해야 합니다:

```python
!wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
```

```python
# Pima Indians 데이터셋용 XGBoost 모델
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

  wandb.init(config=config_defaults)  # 스윕 동안 기본값이 무시됩니다
  config = wandb.config

  # 데이터 로드 및 예측자와 대상 변수로 분할
  dataset = loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
  X, Y = dataset[:, :8], dataset[:, 8]

  # 데이터를 트레인 및 테스트 세트로 분할
  X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                      test_size=config.test_size,
                                                      random_state=config.seed)

  # 모델을 트레인에 맞춥니다
  model = XGBClassifier(booster=config.booster, max_depth=config.max_depth,
                        learning_rate=config.learning_rate, subsample=config.subsample)
  model.fit(X_train, y_train)

  # 테스트에서 예측
  y_pred = model.predict(X_test)
  predictions = [round(value) for value in y_pred]

  # 예측 평가
  accuracy = accuracy_score(y_test, predictions)
  print(f"Accuracy: {accuracy:.0%}")
  wandb.log({"accuracy": accuracy})
```

## 3. 에이전트로 스윕 실행

이제 `wandb.agent`를 호출하여 스윕을 시작합니다.

`wandb.agent`는 W&B에 로그인된 어떤 기계에서든
- `sweep_id`,
- 데이터셋 및 `train` 함수

가 있는 곳에서 호출할 수 있으며,
해당 기계는 스윕에 참여할 것입니다!

> _참고_: `random` 스윕은 기본적으로 영원히 실행됩니다,
새로운 파라미터 조합을 시도하면서 --
또는 [앱 UI에서 스윕을 끄기 전까지](/guides/sweeps/sweeps-ui).
에이전트가 완료할 총 `count`를 제공하여 이를 방지할 수 있습니다.

```python
wandb.agent(sweep_id, train, count=25)
```

## 결과 시각화

이제 스윕이 완료되었으니, 결과를 볼 시간입니다.

Weights & Biases는 자동으로 여러 유용한 플롯을 생성합니다.

### 병렬 좌표 플롯

이 플롯은 하이퍼파라미터 값을 모델 메트릭에 맵핑합니다. 최상의 모델 성능으로 이어진 하이퍼파라미터 조합을 조정하는 데 유용합니다.

이 플롯은 우리의 학습기로 트리를 사용하면 약간
하지만 압도적으로,
간단한 선형 모델을 학습기로 사용하는 것보다 성능이 더 좋다는 것을 나타냅니다.

![sweeps_xgboost](/images/tutorials/xgboost_sweeps/sweeps_xgboost2.png)

### 하이퍼파라미터 중요도 플롯

하이퍼파라미터 중요도 플롯은 메트릭에 가장 큰 영향을 미친 하이퍼파라미터 값을 보여줍니다.

우리는 선형 예측기로서의 상관 관계와 (결과에 대한 랜덤 포레스트 훈련 후) 특징 중요성을 모두 보고하여
가장 큰 영향을 미친 파라미터와 그 효과가 긍정적이었는지 부정적이었는지 알 수 있습니다.

이 차트를 읽어보면 병렬 좌표 차트에서 우리가 인식한 경향의 정량적 확인을 볼 수 있습니다:
검증 정확도에 가장 큰 영향을 미친 것은 학습자의 선택의 결과였으며,
`gblinear` 학습자는 일반적으로 `gbtree` 학습자보다 성능이 낮았습니다.

![sweeps_xgboost](/images/tutorials/xgboost_sweeps/sweeps_xgboost3.png)

이러한 시각화는 하이퍼파라미터 최적화를 실행하는 데 드는 시간과 자원을 절약하고, 중요한 파라미터(및 값의 범위)에 집중하여 추가 탐색에 가치가 있는지를 결정하는 데 도움이 될 수 있습니다.