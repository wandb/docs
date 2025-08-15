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

* TensorFlow 파이프라인에 W&B를 손쉽게 연동해 실험을 추적하는 방법을 소개합니다.
* `keras.metrics`로 메트릭 계산하기
* 커스텀 트레이닝 루프에서 `wandb.log`를 사용해 해당 메트릭을 로그하는 방법

{{< img src="/images/tutorials/tensorflow/dashboard.png" alt="dashboard" >}}

**참고**: _Step_으로 시작하는 섹션만 따라 하시면 기존 코드에 W&B를 통합할 수 있습니다. 나머지는 표준 MNIST 예제입니다.

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

> 참고: W&B를 처음 사용하거나 아직 로그인하지 않았다면, `wandb.login()` 실행 후 나타나는 링크로 이동해 회원가입/로그인을 하실 수 있습니다. 회원가입도 한 번의 클릭으로 쉽게 완료됩니다.

### 데이터셋 준비

```python
# 트레이닝 데이터셋 준비
BATCH_SIZE = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# tf.data로 입력 파이프라인 구성
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

## 트레이닝 루프에 `wandb.log` 추가하기


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
    run = wandb.init(
        project="my-tf-integration",
        config={
            "epochs": epochs,
            "log_step": log_step,
            "val_log_step": val_log_step,
            "architecture": "MLP",
            "dataset": "MNIST",
        },
    )
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []
        val_loss = []

        # 데이터셋 배치를 반복
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

        # 각 에포크 마지막에 검증 루프 실행
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(
                x_batch_val, y_batch_val, model, loss_fn, val_acc_metric
            )
            val_loss.append(float(val_loss_value))

        # 각 에포크 마지막에 메트릭 출력
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # 각 에포크 끝나고 메트릭 리셋
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # run.log()로 메트릭 기록
        run.log(
            {
                "epochs": epoch,
                "loss": np.mean(train_loss),
                "acc": float(train_acc),
                "val_loss": np.mean(val_loss),
                "val_acc": float(val_acc),
            }
        )
    run.finish()
```

## 트레이닝 실행

### `wandb.init()` 호출해서 run 시작하기

이 함수로 실험을 시작했다고 알려주면, 고유 ID와 대시보드를 제공합니다.

[공식 문서를 확인하세요]({{< relref path="/ref/python/sdk/functions/init" lang="ko" >}})

```python
# 프로젝트 이름과 옵션 설정을 지정해 wandb를 초기화합니다.
# config 값을 여러 가지로 바꿔보고, wandb 대시보드에서 결과를 확인해 보세요.
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

# 모델 초기화
model = make_model()

# 옵티마이저 인스턴스 준비
optimizer = keras.optimizers.SGD(learning_rate=config.learning_rate)
# 손실 함수 인스턴스 준비
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 메트릭 준비
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

run.finish()  # 주피터/Colab 환경에서는, 작업이 끝났다고 꼭 알려주세요!
```

### 결과 시각화

위의 [run 페이지]({{< relref path="/guides/models/track/runs/#view-logged-runs" lang="ko" >}}) 링크를 클릭해 실시간 결과를 확인해보세요.

## Sweep 101

W&B Sweeps를 사용해 하이퍼파라미터 최적화를 자동화하고 다양한 모델 공간을 탐색할 수 있습니다.

[W&B Sweeps를 활용한 하이퍼파라미터 최적화 예시 Colab 노트북을 확인해 보세요](https://wandb.me/tf-sweeps-colab)

### W&B Sweeps를 쓰는 이유

* **간편한 설정**: 몇 줄의 코드만으로도 W&B Sweeps를 실행할 수 있습니다.
* **투명성**: 사용하는 모든 알고리즘을 공개하고 [코드도 오픈소스](https://github.com/wandb/sweeps)입니다.
* **강력함**: Sweep은 완전히 사용자 정의 및 설정이 가능합니다. 여러 대의 머신에서 sweep을 실행할 수도 있고, 노트북에서 바로 시작하는 것만큼 쉽습니다.

{{< img src="/images/tutorials/tensorflow/sweeps.png" alt="Sweep result" >}}

## 예제 갤러리

W&B로 트래킹하고 시각화한 다양한 프로젝트 예제를 [Fully Connected →](https://wandb.me/fc) 갤러리에서 만나보세요.

## 모범 사례
1. **Projects**: 여러 run을 같은 프로젝트로 기록해 비교하세요. `wandb.init(project="project-name")`
2. **Groups**: 여러 프로세스나 교차검증 폴드는 각각 run으로 기록해서 그룹으로 묶으세요. `wandb.init(group="experiment-1")`
3. **Tags**: 현재 베이스라인이나 프로덕션 모델을 태그로 구분하세요.
4. **Notes**: 테이블에 노트를 입력해 run 간의 변경점을 추적하세요.
5. **Reports**: 진행 상황을 빠르게 기록해 동료와 공유하거나, 대시보드 및 프로젝트 스냅샷을 만드세요.

### 고급 설정
1. [환경 변수]({{< relref path="/guides/hosting/env-vars/" lang="ko" >}}): 환경 변수로 API 키를 설정해, 관리형 클러스터에서 트레이닝을 실행할 수 있습니다.
2. [오프라인 모드]({{< relref path="/support/kb-articles/run_wandb_offline.md" lang="ko" >}})
3. [온프레미스]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ko" >}}): 프라이빗 클라우드나 완전 폐쇄망 서버 등에서 W&B를 직접 설치하세요. 연구자부터 기업 팀까지 모두 사용할 수 있는 로컬 설치 옵션이 있습니다.
4. [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}}): 모델과 데이터셋을 자동으로 트래킹하고 버전 관리합니다. 파이프라인 단계가 자동으로 기록되어, 모델 트레이닝 시 손쉽게 관리할 수 있습니다.