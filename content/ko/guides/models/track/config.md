---
title: 실험 구성하기
description: 실험 설정을 저장하려면 딕셔너리와 비슷한 오브젝트를 사용하세요.
menu:
  default:
    identifier: ko-guides-models-track-config
    parent: experiments
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W%26B.ipynb" >}}

run 의 `config` 속성을 사용하여 트레이닝 설정을 저장하세요:
- 하이퍼파라미터
- Datasets 이름이나 모델 타입과 같은 입력 설정
- 그 외 실험의 독립 변수들

`wandb.Run.config` 속성은 Experiments 를 분석하거나 나중에 재현할 때 매우 유용합니다. W&B 앱에서는 설정 값별로 그룹화하거나, 다양한 W&B run 의 설정을 비교하고, 각각의 트레이닝 설정이 결과에 어떤 영향을 주는지 평가할 수 있습니다. `config` 속성은 여러 사전(dictionary)형 오브젝트로 구성 가능한, 사전과 유사한 오브젝트입니다.

{{% alert %}}
출력 메트릭 또는 loss, accuracy 같은 종속 변수를 저장하려면 `wandb.Run.config` 대신 `wandb.Run.log()` 를 사용하세요.
{{% /alert %}}


## 실험 설정 구성하기
설정(config)은 일반적으로 트레이닝 스크립트의 초반에 정의합니다. 하지만 기계학습 워크플로우는 다양하기 때문에, 반드시 트레이닝 스크립트 시작시에 설정을 정의할 필요는 없습니다.

설정 변수 이름에는 마침표(`.`) 대신 대시(`-`)나 언더스코어(`_`)를 사용하세요.

`wandb.Run.config`의 루트 아래 키를 엑세스할 경우에는 속성 접근(`config.key.value`)이 아닌 딕셔너리 엑세스 문법(`["key"]["value"]`)을 사용하세요.

아래 섹션에서는 실험 설정을 정의하는 여러 일반적인 시나리오를 소개합니다.

### 초기화 시 설정값 지정하기
스크립트 시작 부분에서 딕셔너리를 `wandb.init()` 호출 시 전달하여 W&B Run으로 데이터 동기화 및 로깅을 위한 백그라운드 프로세스를 생성하세요.

다음 코드조각은 파이썬 딕셔너리로 설정 값을 정의하고, 그 딕셔너리를 W&B Run 초기화 시 인수로 전달하는 방법을 보여줍니다.

```python
import wandb

# config 딕셔너리 오브젝트 정의
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B를 초기화할 때 config 딕셔너리 전달
with wandb.init(project="config_example", config=config) as run:
    ...
```

중첩된 딕셔너리를 `config`로 전달하면 W&B는 이름을 점(`.`)으로 연결해 평탄화합니다.

딕셔너리에서 값에 엑세스하는 방법은 파이썬 딕셔너리와 동일합니다:

```python
# 키를 인덱스 값으로 사용해 값에 엑세스
hidden_layer_sizes = run.config["hidden_layer_sizes"]
kernel_sizes = run.config["kernel_sizes"]
activation = run.config["activation"]

# 파이썬 딕셔너리 get() 메소드 사용
hidden_layer_sizes = run.config.get("hidden_layer_sizes")
kernel_sizes = run.config.get("kernel_sizes")
activation = run.config.get("activation")
```

{{% alert %}}
Developer Guide와 예시 전체에 걸쳐 설정 값을 별도의 변수에 복사해 사용하는 경우가 있습니다. 이 단계는 *선택적*입니다. 코드를 더 읽기 쉽게 하기 위함입니다.
{{% /alert %}}

### argparse로 설정값 지정하기
argparse 오브젝트로 설정을 지정할 수도 있습니다. [argparse](https://docs.python.org/3/library/argparse.html)는 파이썬 3.2 이후 표준 라이브러리로, 커맨드라인 인수를 쉽게 처리할 수 있게 합니다.

이 방법은 커맨드라인에서 실행되는 스크립트의 결과 추적에 유용합니다.

다음 파이썬 스크립트는 실험 설정을 정의하고 지정할 파서 객체 정의 예제입니다. 트레이닝 루프를 시뮬레이션하기 위한 `train_one_epoch`, `evaluate_one_epoch` 함수도 포함되어 있습니다:

```python
# config_experiment.py
import argparse
import random

import numpy as np
import wandb


# 트레이닝/평가 데모 코드
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main(args):
    # W&B Run 시작
    with wandb.init(project="config_example", config=args) as run:
        # config 딕셔너리에서 값을 읽어서 가독성을 위해 변수에 저장
        lr = run.config["learning_rate"]
        bs = run.config["batch_size"]
        epochs = run.config["epochs"]

        # 트레이닝 및 W&B에 값 로깅 시뮬레이션
        for epoch in np.arange(1, epochs):
            train_acc, train_loss = train_one_epoch(epoch, lr, bs)
            val_acc, val_loss = evaluate_one_epoch(epoch)

            run.log(
                {
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-b", "--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument(
        "-e", "--epochs", type=int, default=50, help="트레이닝 에포크 수"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=int, default=0.001, help="러닝레이트"
    )

    args = parser.parse_args()
    main(args)
```

### 스크립트 내에서 설정값 추가/수정하기
스크립트 실행 중 config 오브젝트에 parameter 를 계속 추가할 수 있습니다. 다음 코드조각은 config 오브젝트에 새로운 key-value 쌍을 추가하는 방법을 보여줍니다:

```python
import wandb

# config 딕셔너리 오브젝트 정의
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B를 초기화할 때 config 딕셔너리 전달
with wandb.init(project="config_example", config=config) as run:
    # W&B 초기화 후 config 업데이트
    run.config["dropout"] = 0.2
    run.config.epochs = 4
    run.config["batch_size"] = 32
```

여러 값을 한 번에 업데이트할 수도 있습니다:

```python
run.config.update({"lr": 0.1, "channels": 16})
```

### Run 종료 후 설정값 수정하기
완료된 run 의 설정값을 업데이트할 때는 [W&B Public API]({{< relref path="/ref/python/public-api/index.md" lang="ko" >}})를 사용하세요.

API에는 entity, 프로젝트 이름, run의 ID가 필요합니다. Run 오브젝트나 [W&B 앱]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})에서 이 정보를 확인할 수 있습니다:

```python
with wandb.init() as run:
    ...

# Run 오브젝트(스크립트 또는 노트북에서 생성된 경우)에서 아래 값을 찾거나,
# 혹은 W&B 앱 UI에서 복사하세요.
username = run.entity
project = run.project
run_id = run.id

# 참고: api.run()은 wandb.init()과는 다른 타입의 오브젝트를 반환합니다.
api = wandb.Api()
api_run = api.run(f"{username}/{project}/{run_id}")
api_run.config["bar"] = 32
api_run.update()
```



## `absl.FLAGS`


[`absl` flags](https://abseil.io/docs/python/guides/flags)도 config에 전달할 수 있습니다.

```python
flags.DEFINE_string("model", None, "실행할 모델")  # name, default, help

run.config.update(flags.FLAGS)  # absl flag들을 config에 추가
```

## 파일 기반 설정 사용하기
run 스크립트와 동일한 디렉토리에 `config-defaults.yaml` 파일을 두면, run 이 파일에서 정의한 key-value 쌍들을 자동으로 읽어서 `wandb.Run.config`에 전달합니다.

다음은 샘플 `config-defaults.yaml` 파일 예제입니다:

```yaml
batch_size:
  desc: 각 미니배치의 크기
  value: 32
```

`config-defaults.yaml`로부터 자동으로 로드된 기본값은 `wandb.init`의 `config` 인수에서 값을 새로 지정하면 덮어쓸 수 있습니다. 예를 들면:

```python
import wandb

# config-defaults.yaml에 정의된 값 대신 커스텀 값 지정
with wandb.init(config={"epochs": 200, "batch_size": 64}) as run:
    ...
```

`config-defaults.yaml` 외의 구성 파일을 사용하려면, `--configs` 커맨드라인 인수를 사용해 직접 파일 경로를 지정하면 됩니다:

```bash
python train.py --configs other-config.yaml
```

### 파일 기반 설정의 예시 유스 케이스
run 을 위한 메타데이터가 들어있는 YAML 파일과, 파이썬 스크립트 안에 선언한 하이퍼파라미터 딕셔너리를 모두 config 오브젝트에 중첩해 저장할 수 있습니다:

```python
hyperparameter_defaults = dict(
    dropout=0.5,
    batch_size=100,
    learning_rate=0.001,
)

config_dictionary = dict(
    yaml=my_yaml_file,
    params=hyperparameter_defaults,
)

with wandb.init(config=config_dictionary) as run:
    ...
```

## TensorFlow v1 플래그

TensorFlow flag를 바로 `wandb.Run.config` 오브젝트에 전달할 수 있습니다.

```python
with wandb.init() as run:
    run.config.epochs = 4

    flags = tf.app.flags
    flags.DEFINE_string("data_dir", "/tmp/data")
    flags.DEFINE_integer("batch_size", 128, "배치 크기.")
    run.config.update(flags.FLAGS)  # tensorflow flag들을 config에 추가
```