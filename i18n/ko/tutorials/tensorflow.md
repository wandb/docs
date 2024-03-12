
# TensorFlow

[**여기에서 Colab 노트북으로 시도해 보세요 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Simple_TensorFlow_Integration.ipynb)

기계학습 실험 추적, 데이터셋 버전 관리 및 프로젝트 협업을 위해 Weights & Biases를 사용하세요.

<div><img /></div>

<img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

<div><img /></div>

## 이 노트북에서 다루는 내용

* 실험 추적을 위한 TensorFlow 파이프라인과 Weights and Biases의 쉬운 통합.
* `keras.metrics`를 사용한 메트릭 계산
* 사용자 정의 트레이닝 루프에서 `wandb.log`를 사용하여 해당 메트릭 로깅.

## 인터랙티브한 W&B 대시보드는 다음과 같습니다:


![대시보드](/images/tutorials/tensorflow/dashboard.png)

**참고**: _Step_으로 시작하는 섹션은 기존 코드에 W&B를 통합하는 데 필요한 모든 것입니다. 나머지는 단지 표준 MNIST 예제일 뿐입니다.




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
from wandb.keras import WandbCallback

wandb.login()
```

> 부가 설명: W&B를 처음 사용하거나 로그인하지 않은 경우, `wandb.login()`을 실행한 후 나타나는 링크가 가입/로그인 페이지로 이동합니다. 가입은 한 번의 클릭으로 간단합니다.

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
        print("\n에포크 %d 시작" % (epoch,))

        train_loss = []   
        val_loss = []

        # 데이터셋의 배치를 반복 처리
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train, 
                                    model, optimizer, 
                                    loss_fn, train_acc_metric)
            train_loss.append(float(loss_value))

        # 각 에포크의 끝에 검증 루프 실행
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(x_batch_val, y_batch_val, 
                                       model, loss_fn, 
                                       val_acc_metric)
            val_loss.append(float(val_loss_value))
            
        # 각 에포크의 끝에 메트릭 표시
        train_acc = train_acc_metric.result()
        print("에포크별 트레이닝 정확도: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("검증 정확도: %.4f" % (float(val_acc),))

        # 각 에포크의 끝에 메트릭 재설정
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # ⭐: wandb.log를 사용하여 메트릭 로깅
        wandb.log({'epochs': epoch,
                   'loss': np.mean(train_loss),
                   'acc': float(train_acc), 
                   'val_loss': np.mean(val_loss),
                   'val_acc':float(val_acc)})
```

# 👟 트레이닝 실행

## Step 3️⃣: 실행을 시작하기 위해 `wandb.init` 호출

실험을 시작한다는 것을 알려주기 위해 이를 호출하면, 고유 ID와 대시보드가 제공됩니다.

[여기서 공식 문서 확인 $\rightarrow$](https://docs.wandb.com/library/init)



```python
# 프로젝트 이름과 선택적으로 설정을 사용하여 wandb 초기화.
# 설정 값들을 변경해보고 wandb 대시보드에서 결과를 확인하세요.
config = {
              "learning_rate": 0.001,
              "epochs": 10,
              "batch_size": 64,
              "log_step": 200,
              "val_log_step": 50,
              "아키텍처": "CNN",
              "데이터셋": "CIFAR-10"
           }

run = wandb.init(project='my-tf-integration', config=config)
config = wandb.config

# 모델 초기화.
model = make_model()

# 모델을 트레이닝하기 위한 옵티마이저 인스턴스화.
optimizer = keras.optimizers.SGD(learning_rate=config.learning_rate)
# 손실 함수 인스턴스화.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 메트릭 준비.
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

run.finish()  # Jupyter/Colab에서, 작업이 끝났음을 알립니다!
```

# 👀 결과 시각화

위의 [**실행 페이지**](https://docs.wandb.ai/ref/app/pages/run-page)
링크를 클릭하여 실시간 결과를 확인하세요.

# 🧹 스윕 101

하이퍼파라미터 최적화를 자동화하고 가능한 모델의 공간을 탐색하기 위해 Weights & Biases 스윕을 사용하세요.

## [W&B 스윕을 사용한 TensorFlow에서의 하이퍼파라미터 최적화 확인하기 $\rightarrow$](http://wandb.me/tf-sweeps-colab)

### W&B 스윕 사용의 이점

* **빠른 설정**: 몇 줄의 코드로 W&B 스윕을 실행할 수 있습니다.
* **투명성**: 우리가 사용하는 모든 알고리즘을 인용하며, [우리의 코드는 오픈 소스입니다](https://github.com/wandb/client/tree/master/wandb/sweeps).
* **강력함**: 우리의 스윕은 완전히 맞춤 설정 및 구성 가능합니다. 여러분의 노트북에서 스윕을 시작하는 것이 여러 대의 기계에 걸쳐 스윕을 시작하는 것만큼 쉽습니다.


<img src="https://i.imgur.com/6eWHZhg.png" alt="Sweep Result" />

# 🎨 예시 갤러리

W&B에서 추적 및 시각화된 프로젝트의 예시를 우리의 예시 갤러리, [Fully Connected →](https://wandb.me/fc)에서 확인하세요.

# 📏 모범 사례
1. **프로젝트**: 여러 실행을 프로젝트에 로그하여 비교합니다. `wandb.init(project="project-name")`
2. **그룹**: 여러 프로세스 또는 교차검증 폴드의 경우, 각 프로세스를 실행으로 로그하고 함께 그룹화합니다. `wandb.init(group='experiment-1')`
3. **태그**: 현재 베이스라인 또는 프로덕션 모델을 추적하기 위해 태그를 추가합니다.
4. **노트**: 실행 사이의 변경 사항을 추적하기 위해 테이블에 노트를 입력합니다.
5. **리포트**: 진행 상황에 대한 빠른 노트를 동료와 공유하고 ML 프로젝트의 대시보드 및 스냅샷을 만듭니다.

## 🤓 고급 설정
1. [환경 변수](https://docs.wandb.com/library/environment-variables): 관리 클러스터에서 트레이닝을 실행할 수 있도록 환경 변수에 API 키를 설정합니다.
2. [오프라인 모드](https://docs.wandb.com/library/technical-faq#can-i-run-wandb-offline): `dryrun` 모드를 사용하여 오프라인으로 트레이닝을 하고 나중에 결과를 동기화합니다.
3. [온프레미스](https://docs.wandb.com/self-hosted): W&B를 귀하의 인프라 내의 프라이빗 클라우드 또는 에어갭 서버에 설치하세요. 우리는 학계부터 엔터프라이즈 팀까지 모든 사람을 위한 로컬 설치를 제공합니다.
4. [아티팩트](http://wandb.me/artifacts-colab): 모델과 데이터셋을 스트림라인 방식으로 추적 및 버전 관리하여 모델을 트레이닝할 때 파이프라인 단계를 자동으로 파악합니다.