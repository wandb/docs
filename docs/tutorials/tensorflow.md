---
title: TensorFlow
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Simple_TensorFlow_Integration.ipynb'/>

Weights & Biases를 사용하여 기계학습 실험 추적, 데이터셋 버전 관리 및 프로젝트 협업을 수행하세요.

![](/images/tutorials/huggingface-why.png)

## 이 노트북에서 다루는 내용

* 당신의 TensorFlow 파이프라인과 Weights and Biases의 쉬운 통합을 통한 실험 추적.
* `keras.metrics`를 사용하여 메트릭 계산하기
* `wandb.log`를 사용하여 사용자 정의 트레이닝 루프에서 이러한 메트릭을 로그하기

## 대화형 W&B 대시보드는 다음과 같이 보일 것입니다:

![dashboard](/images/tutorials/tensorflow/dashboard.png)

**참고**: _Step_으로 시작하는 섹션은 W&B를 기존 코드에 통합하기 위해 필요한 모든 것입니다. 나머지는 표준 MNIST 예제일 뿐입니다.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

# 🚀 설치, 가져오기, 로그인

## Step 0️⃣: W&B 설치


```python
%%capture
!pip install wandb
```

## Step 1️⃣: W&B 가져오기 및 로그인


```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

wandb.login()
```

> 참고: W&B를 처음 사용하거나 로그인하지 않은 경우 `wandb.login()`을 실행한 후 나타나는 링크는 회원 가입/로그인 페이지로 안내됩니다. 가입은 한 번의 클릭으로 간단합니다.

# 👩‍🍳 데이터셋 준비


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

# 🧠 모델 및 트레이닝 루프 정의


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

## Step 2️⃣: 트레이닝 루프에 `wandb.log` 추가


```python
def train(train_dataset, val_dataset,  model, optimizer,
          train_acc_metric, val_acc_metric,
          epochs=10,  log_step=200, val_log_step=50):
  
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []   
        val_loss = []

        # 데이터셋 배치 반복
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train, 
                                    model, optimizer, 
                                    loss_fn, train_acc_metric)
            train_loss.append(float(loss_value))

        # 각 에포크가 끝날 때 검증 루프 실행
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(x_batch_val, y_batch_val, 
                                       model, loss_fn, 
                                       val_acc_metric)
            val_loss.append(float(val_loss_value))
            
        # 각 에포크가 끝날 때 메트릭 표시
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # 각 에포크가 끝날 때 메트릭 초기화
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # ⭐: wandb.log를 사용하여 메트릭 로그
        wandb.log({'epochs': epoch,
                   'loss': np.mean(train_loss),
                   'acc': float(train_acc), 
                   'val_loss': np.mean(val_loss),
                   'val_acc':float(val_acc)})
```

# 👟 트레이닝 실행

## Step 3️⃣: `wandb.init` 호출하여 run 시작

이것은 당신이 실험을 시작하고 있음을 알리며, 우리는 고유한 ID와 대시보드를 제공할 수 있습니다.

[공식 문서 확인하기](/ref/python/init)

```python
# 프로젝트 이름과 선택적으로 설정 값을 가지고 wandb를 초기화합니다.
# 설정 값을 가지고 실험해 보고 wandb 대시보드에서 결과를 확인하세요.
config = {
              "learning_rate": 0.001,
              "epochs": 10,
              "batch_size": 64,
              "log_step": 200,
              "val_log_step": 50,
              "architecture": "CNN",
              "dataset": "CIFAR-10"
           }

run = wandb.init(project='my-tf-integration', config=config)
config = wandb.config

# 모델 초기화
model = make_model()

# 모델을 트레이닝할 옵티마이저 인스턴스화
optimizer = keras.optimizers.SGD(learning_rate=config.learning_rate)
# 손실 함수 인스턴스화
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 메트릭 준비
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

train(train_dataset,
      val_dataset, 
      model,
      optimizer,
      train_acc_metric,
      val_acc_metric,
      epochs=config.epochs, 
      log_step=config.log_step, 
      val_log_step=config.val_log_step)

run.finish()  # Jupyter/Colab에서 완료되었음을 알려주세요!
```

# 👀 결과 시각화

위 [**런 페이지**](/guides/app/pages/run-page) 링크를 클릭하여 라이브 결과를 확인하세요.

# 🧹 스윕 101

Weights & Biases Sweeps를 사용하여 하이퍼파라미터 최적화를 자동화하고 가능한 모델의 공간을 탐색하세요.

## [W&B Sweeps를 사용한 TensorFlow 하이퍼파라미터 최적화 확인하기](http://wandb.me/tf-sweeps-colab)

### W&B Sweeps를 사용하는 이점

* **빠른 설정**: 몇 줄의 코드만으로 W&B 스윕을 실행할 수 있습니다.
* **투명성**: 우리가 사용하는 모든 알고리즘을 인용하며, [우리의 코드는 오픈 소스입니다](https://github.com/wandb/client/tree/master/wandb/sweeps).
* **강력함**: 우리의 스윕은 완전히 사용자 정의 가능하고 구성 가능합니다. 수십 대의 머신에서 스윕을 시작하는 것도 랩톱에서 스윕을 시작하는 것만큼 쉽습니다.

![Sweep result](/images/tutorials/tensorflow/sweeps.png)

# 🎨 예제 갤러리

우리의 예제 갤러리에서 W&B로 추적하고 시각화한 프로젝트들의 예제를 보세요, [Fully Connected →](https://wandb.me/fc)

# 📏 모범 사례
1. **Projects**: 여러 run을 로그하여 프로젝트에 비교합니다. `wandb.init(project="project-name")`
2. **Groups**: 여러 프로세스나 교차검증 폴드를 위해 각 프로세스를 run으로 로그하고 함께 그룹화합니다. `wandb.init(group='experiment-1')`
3. **Tags**: 현재 베이스라인이나 프로덕션 모델을 추적하기 위해 태그를 추가합니다.
4. **Notes**: 테이블에 메모를 입력하여 run 간의 변화를 추적합니다.
5. **Reports**: 진행 사항에 대한 빠른 메모를 작성하여 동료와 공유하고 ML 프로젝트의 대시보드와 스냅샷을 만드세요.

## 🤓 고급 설정
1. [환경 변수](/guides/hosting/env-vars): API 키를 환경 변수에 설정하여 관리되는 클러스터에서 트레이닝을 실행할 수 있습니다.
2. [오프라인 모드](/guides/technical-faq/setup/#can-i-run-wandb-offline)
3. [온프레미스](/guides/hosting/hosting-options/self-managed): 프라이빗 클라우드나 자체 인프라의 에어갭 서버에 W&B를 설치하세요. 학계부터 기업 팀까지 모두를 위한 로컬 설치가 가능합니다.
4. [Artifacts](/guides/artifacts): 모델과 데이터셋을 추적하고 버전 관리하는 효율적인 방법으로, 모델을 트레이닝할 때 파이프라인 단계를 자동으로 캡쳐합니다.