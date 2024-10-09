---
title: TensorFlow Sweeps
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Hyperparameter_Optimization_in_TensorFlow_using_W&B_Sweeps.ipynb'/>

Weights & Biases를 사용하여 기계학습 실험 추적, 데이터셋 버전 관리 및 프로젝트 협업을 수행하세요.

![](/images/tutorials/huggingface-why.png)

Weights & Biases의 Sweeps를 사용하여 하이퍼파라미터 최적화를 자동화하고 가능한 모델의 공간을 탐색하세요. 이와 같은 인터랙티브한 대시보드도 제공합니다:

![](/images/tutorials/tensorflow/sweeps.png)

## 🤔 Sweeps를 사용해야 하는 이유는 무엇인가요?

* **빠른 설정**: 몇 줄의 코드만으로 W&B sweeps를 실행할 수 있습니다.
* **투명성**: 우리가 사용하는 모든 알고리즘을 명시하고 있으며, [우리의 코드는 오픈 소스입니다](https://github.com/wandb/client/tree/master/wandb/sweeps).
* **강력함**: 우리의 sweeps는 완전히 커스터마이즈 가능하고 설정 가능합니다. 여러 대의 기계에서 스윕을 시작해도 개인 노트북에서 하는 것만큼 쉽습니다.

**[공식 문서 보기](/guides/sweeps)**

## 이 노트북에서 다루는 내용

* TensorFlow에서 커스텀 트레이닝 루프를 사용하여 W&B Sweep을 시작하는 간단한 단계들.
* 우리의 이미지 분류 작업을 위한 최적의 하이퍼파라미터를 찾습니다.

**참고**: _Step_으로 시작하는 섹션들은 기존 코드에서 하이퍼파라미터 탐색을 수행하는 데 필요한 모든 것입니다.
나머지 코드는 단순한 예제를 구성하기 위한 것입니다.

## 🚀 설치, 가져오기 및 로그인

### Step 0️⃣: W&B 설치

```python
%%capture
!pip install wandb
```

### Step 1️⃣: W&B 가져오기 및 로그인

```python
import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

wandb.login()
```

> 보충 설명: W&B를 처음 사용하시거나 로그인되지 않은 경우 `wandb.login()`을 실행한 후 나타나는 링크가 회원가입/로그인 페이지로 이동시켜 줍니다. 회원가입은 몇 번의 클릭만으로 가능합니다.

## 👩‍🍳 데이터셋 준비

```python
# 트레이닝 데이터셋 준비
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))
```

## 🧠 모델 및 트레이닝 루프 정의하기

## 🏗️ 간단한 분류 MLP 구축하기

```python
def Model():
    inputs = keras.Input(shape=(784,), name="digits")
    x1 = keras.layers.Dense(64, activation="relu")(inputs)
    x2 = keras.layers.Dense(64, activation="relu")(x1)
    outputs = keras.layers.Dense(10, name="predictions")(x2)

    return keras.Model(inputs=inputs, outputs=outputs)

    
def train_step(x, y, model, optimizer, loss_fn, train_acc_metric):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric.update_state(y, logits)

    return loss_value

    
def test_step(x, y, model, loss_fn, val_acc_metric):
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    val_acc_metric.update_state(y, val_logits)

    return loss_value
```

## 🔁 트레이닝 루프 작성하기

### Step 3️⃣: `wandb.log`로 메트릭 기록하기

```python
def train(train_dataset,
          val_dataset, 
          model,
          optimizer,
          loss_fn,
          train_acc_metric,
          val_acc_metric,
          epochs=10, 
          log_step=200, 
          val_log_step=50):
  
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []   
        val_loss = []

        # 데이터셋의 배치를 반복 처리
        for step, (x_batch_train, y_batch_train) in tqdm.tqdm(enumerate(train_dataset), total=len(train_dataset)):
            loss_value = train_step(x_batch_train, y_batch_train, 
                                    model, optimizer, 
                                    loss_fn, train_acc_metric)
            train_loss.append(float(loss_value))

        # 에포크 종료 시 검증 루프 실행
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(x_batch_val, y_batch_val, 
                                       model, loss_fn, 
                                       val_acc_metric)
            val_loss.append(float(val_loss_value))
            
        # 에포크 종료 시 메트릭 표시
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # 에포크 종료 시 메트릭 리셋
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # 3️⃣ wandb.log를 사용하여 메트릭 기록
        wandb.log({'epochs': epoch,
                   'loss': np.mean(train_loss),
                   'acc': float(train_acc), 
                   'val_loss': np.mean(val_loss),
                   'val_acc':float(val_acc)})
```

### Step 4️⃣: Sweep 설정하기

여기서는 다음을 수행하세요:
* 탐색할 하이퍼파라미터 정의
* 하이퍼파라미터 최적화 메소드 제공. `random`, `grid`, `bayes` 메소드를 사용할 수 있습니다.
* `bayes`를 사용하는 경우 `metric`과 `목표`를 제공하여, 예를 들어 `val_loss`를 `minimize`하도록 설정.
* 성능이 좋지 않은 실행의 조기 종료를 위해 `hyperband` 사용

#### [자세한 Sweep 설정 정보 보기](/guides/sweeps/define-sweep-configuration)

```python
sweep_config = {
  'method': 'random', 
  'metric': {
      'name': 'val_loss',
      'goal': 'minimize'
  },
  'early_terminate':{
      'type': 'hyperband',
      'min_iter': 5
  },
  'parameters': {
      'batch_size': {
          'values': [32, 64, 128, 256]
      },
      'learning_rate':{
          'values': [0.01, 0.005, 0.001, 0.0005, 0.0001]
      }
  }
}
```

### Step 5️⃣: 트레이닝 루프 감싸기

`train`이 호출되기 전에 하이퍼파라미터를 설정하기 위해 `wandb.config`를 사용하는, 아래의 `sweep_train`과 같은 함수가 필요합니다.

```python
def sweep_train(config_defaults=None):
    # 기본 값 설정
    config_defaults = {
        "batch_size": 64,
        "learning_rate": 0.01
    }
    # 예제 프로젝트 이름으로 wandb 초기화
    wandb.init(config=config_defaults)  # 스윕에서 덮어씁니다.

    # 기타 하이퍼파라미터를 설정에 명시
    wandb.config.epochs = 2
    wandb.config.log_step = 20
    wandb.config.val_log_step = 50
    wandb.config.architecture_name = "MLP"
    wandb.config.dataset_name = "MNIST"

    # tf.data를 사용해 입력 파이프라인 구축
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (train_dataset.shuffle(buffer_size=1024)
                                  .batch(wandb.config.batch_size)
                                  .prefetch(buffer_size=tf.data.AUTOTUNE))

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = (val_dataset.batch(wandb.config.batch_size)
                              .prefetch(buffer_size=tf.data.AUTOTUNE))

    # 모델 초기화
    model = Model()

    # 모델을 트레이닝 할 옵티마이저 인스턴스화
    optimizer = keras.optimizers.SGD(learning_rate=wandb.config.learning_rate)
    # 손실 함수 인스턴스화
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # 메트릭 준비.
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    train(train_dataset,
          val_dataset, 
          model,
          optimizer,
          loss_fn,
          train_acc_metric,
          val_acc_metric,
          epochs=wandb.config.epochs, 
          log_step=wandb.config.log_step, 
          val_log_step=wandb.config.val_log_step)
```

### Step 6️⃣: Sweep 초기화 및 에이전트 실행

```python
sweep_id = wandb.sweep(sweep_config, project="sweeps-tensorflow")
```

`count` 파라미터로 총 실행 횟수를 제한할 수 있으며, 스크립트를 빠르게 실행하기 위해 10으로 제한합니다. 실행 횟수를 늘려보고 어떤 변화가 있는지 확인하세요.

```python
wandb.agent(sweep_id, function=sweep_train, count=10)
```

## 👀 결과 시각화

위의 **Sweep URL** 링크를 클릭하여 라이브 결과를 확인하세요.

## 🎨 예제 갤러리

W&B로 추적하고 시각화된 프로젝트의 예제를 [갤러리에서 →](https://app.wandb.ai/gallery) 확인하세요.

## 📏 모범 사례
1. **Projects**: 여러 runs를 로그하여 프로젝트에서 비교하세요. `wandb.init(project="project-name")`
2. **Groups**: 여러 프로세스나 교차검증을 위해, 각각의 프로세스를 run으로 로그하고 그룹화하세요. `wandb.init(group='experiment-1')`
3. **Tags**: 현재 베이스라인이나 프로덕션 모델을 추적하기 위해 태그 추가.
4. **Notes**: runs 간의 변경 사항을 추적하기 위해 테이블에 노트 작성.
5. **Reports**: 동료와 공유하기 위해 진행 상황에 대한 빠른 노트를 작성하고 대시보드 및 ML 프로젝트의 스냅샷 생성.

## 🤓 고급 설정
1. [환경 변수](/guides/hosting/env-vars): 관리되는 클러스터에서 트레이닝을 실행할 수 있도록 API 키를 환경 변수에 설정.
2. [오프라인 모드](/guides/technical-faq/setup/#can-i-run-wandb-offline)
3. [온프레미스](/guides/hosting/hosting-options/self-managed): 프라이빗 클라우드 또는 자체 인프라의 에어갭 서버에 W&B 설치. 학계부터 기업 팀까지 모든 사람을 위한 로컬 설치 제공.