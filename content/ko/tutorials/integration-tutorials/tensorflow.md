---
title: TensorFlow
menu:
  tutorials:
    identifier: ko-tutorials-integration-tutorials-tensorflow
    parent: integration-tutorials
weight: 4
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Simple_TensorFlow_Integration.ipynb" >}}

## 이 노트북에서 다루는 내용

* 실험 추적을 위해 TensorFlow 파이프라인과 Weights & Biases 의 간편한 통합
* `keras.metrics`를 사용한 메트릭 계산
* 사용자 정의 트레이닝 루프에서 해당 메트릭을 기록하기 위해 `wandb.log` 사용

{{< img src="/images/tutorials/tensorflow/dashboard.png" alt="dashboard" >}}

**참고**: _Step_ 으로 시작하는 섹션은 기존 코드에 W&B 를 통합하는 데 필요한 전부입니다. 나머지는 표준 MNIST 예제입니다.

```python
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
```

## 설치, 임포트, 로그인

### W&B 설치


```jupyter
%%capture
!pip install wandb
```

### W&B 임포트 및 로그인


```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

wandb.login()
```

> 참고: W&B 를 처음 사용하거나 로그인하지 않은 경우 `wandb.login()` 실행 후 나타나는 링크를 통해 가입/로그인 페이지로 이동합니다. 가입은 클릭 한 번으로 간단하게 완료할 수 있습니다.

### 데이터셋 준비

```python
# 트레이닝 데이터셋 준비
BATCH_SIZE = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# tf.data를 사용하여 입력 파이프라인 구축
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(BATCH_SIZE)
```

## 모델 및 트레이닝 루프 정의

```python
def make_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x1 = keras.layers.Dense(64, activation="relu")(inputs)
    x2 = keras.layers.Dense(64, activation="relu")(x1)
    outputs = keras.layers.Dense(10, name="predictions")(x2)

    return keras.Model(inputs=inputs, outputs=outputs)
```


```python
def train_step(x, y, model, optimizer, loss_fn, train_acc_metric):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric.update_state(y, logits)

    return loss_value
```


```python
def test_step(x, y, model, loss_fn, val_acc_metric):
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    val_acc_metric.update_state(y, val_logits)

    return loss_value
```

## 트레이닝 루프에 `wandb.log` 추가


```python
def train(
    train_dataset,
    val_dataset,
    model,
    optimizer,
    train_acc_metric,
    val_acc_metric,
    epochs=10,
    log_step=200,
    val_log_step=50,
):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []
        val_loss = []

        # 데이터셋의 배치를 반복합니다.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(
                x_batch_train,
                y_batch_train,
                model,
                optimizer,
                loss_fn,
                train_acc_metric,
            )
            train_loss.append(float(loss_value))

        # 각 에포크가 끝날 때 검증 루프를 실행합니다.
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(
                x_batch_val, y_batch_val, model, loss_fn, val_acc_metric
            )
            val_loss.append(float(val_loss_value))

        # 각 에포크가 끝날 때 메트릭을 표시합니다.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # 각 에포크가 끝날 때 메트릭을 재설정합니다.
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # ⭐: wandb.log를 사용하여 메트릭 기록
        wandb.log(
            {
                "epochs": epoch,
                "loss": np.mean(train_loss),
                "acc": float(train_acc),
                "val_loss": np.mean(val_loss),
                "val_acc": float(val_acc),
            }
        )
```

## 트레이닝 실행

### `wandb.init`을 호출하여 run 시작

이를 통해 실험을 시작했음을 알 수 있으며,
고유한 ID와 대시보드를 제공할 수 있습니다.

[공식 문서 확인하기]({{< relref path="/ref/python/init" lang="ko" >}})

```python
# 프로젝트 이름과 함께 wandb를 초기화하고 선택적으로 구성을 초기화합니다.
# 구성 값을 변경하고 wandb 대시보드에서 결과를 확인하십시오.
config = {
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 64,
    "log_step": 200,
    "val_log_step": 50,
    "architecture": "CNN",
    "dataset": "CIFAR-10",
}

run = wandb.init(project='my-tf-integration', config=config)
config = run.config

# 모델 초기화.
model = make_model()

# 모델을 트레이닝하기 위한 옵티마이저 인스턴스화.
optimizer = keras.optimizers.SGD(learning_rate=config.learning_rate)
# 손실 함수 인스턴스화.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 메트릭 준비.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

train(
    train_dataset,
    val_dataset, 
    model,
    optimizer,
    train_acc_metric,
    val_acc_metric,
    epochs=config.epochs, 
    log_step=config.log_step, 
    val_log_step=config.val_log_step,
)

run.finish()  # Jupyter/Colab에서 완료되었음을 알립니다!
```

### 결과 시각화

라이브 결과를 보려면 위의 [**run page**]({{< relref path="/guides/models/track/runs/#view-logged-runs" lang="ko" >}}) 링크를 클릭하십시오.

## Sweep 101

Weights & Biases Sweeps 를 사용하여 하이퍼파라미터 최적화를 자동화하고 가능한 모델 공간을 탐색합니다.

## [W&B Sweeps 를 사용하여 TensorFlow 에서 하이퍼파라미터 최적화 확인](http://wandb.me/tf-sweeps-colab)

### W&B Sweeps 사용의 이점

* **빠른 설정**: 몇 줄의 코드만으로 W&B 스윕을 실행할 수 있습니다.
* **투명성**: 사용 중인 모든 알고리즘을 인용하고 [코드는 오픈 소스입니다](https://github.com/wandb/sweeps).
* **강력함**: 스윕은 완벽하게 사용자 정의하고 구성할 수 있습니다. 수십 대의 장치에서 스윕을 시작할 수 있으며 랩톱에서 스윕을 시작하는 것만큼 쉽습니다.

{{< img src="/images/tutorials/tensorflow/sweeps.png" alt="Sweep result" >}}

## 예제 갤러리

W&B 로 추적하고 시각화한 프로젝트의 예제를 예제 갤러리에서 확인하세요. [완전 연결 →](https://wandb.me/fc)

## 모범 사례
1. **Projects**: 여러 runs 를 프로젝트에 기록하여 비교합니다. `wandb.init(project="project-name")`
2. **Groups**: 여러 프로세스 또는 교차 검증 폴드의 경우 각 프로세스를 runs 로 기록하고 함께 그룹화합니다. `wandb.init(group="experiment-1")`
3. **Tags**: 현재 베이스라인 또는 프로덕션 모델을 추적하기 위해 태그를 추가합니다.
4. **Notes**: 테이블에 노트를 입력하여 runs 간의 변경 사항을 추적합니다.
5. **Reports**: 동료와 공유하고 ML 프로젝트의 대시보드 및 스냅샷을 만들기 위해 진행 상황에 대한 빠른 노트를 작성합니다.

### 고급 설정
1. [환경 변수]({{< relref path="/guides/hosting/env-vars/" lang="ko" >}}): 관리형 클러스터에서 트레이닝을 실행할 수 있도록 환경 변수에 API 키를 설정합니다.
2. [오프라인 모드]({{< relref path="/support/kb-articles/run_wandb_offline.md" lang="ko" >}})
3. [On-prem]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ko" >}}): 자체 인프라의 프라이빗 클라우드 또는 에어 갭 서버에 W&B 를 설치합니다. 학계에서 엔터프라이즈 팀에 이르기까지 모든 사람을 위한 로컬 설치가 있습니다.
4. [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}}): 모델을 트레이닝할 때 파이프라인 단계를 자동으로 선택하는 간소화된 방식으로 모델 및 데이터셋을 추적하고 버전을 관리합니다.
