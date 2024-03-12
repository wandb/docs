
# TensorFlow Sweeps

[**여기에서 Colab 노트북으로 시도해보세요 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Hyperparameter_Optimization_in_TensorFlow_using_W&B_Sweeps.ipynb)

기계학습 실험 추적, 데이터셋 버전 관리 및 프로젝트 협업을 위해 Weights & Biases를 사용하세요.

<img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

이와 같은 상호작용 대시보드를 포함하여 하이퍼파라미터 최적화를 자동화하고 가능한 모델의 공간을 탐색하기 위해 Weights & Biases Sweeps를 사용하세요:

![](https://i.imgur.com/AN0qnpC.png)

## 🤔 왜 Sweeps를 사용해야 할까요?

* **빠른 설정**: 몇 줄의 코드만으로 W&B sweeps를 실행할 수 있습니다.
* **투명성**: 사용하는 모든 알고리즘을 인용하며, [우리의 코드는 오픈 소스입니다](https://github.com/wandb/client/tree/master/wandb/sweeps).
* **강력함**: 우리의 sweeps는 완전히 사용자 정의가 가능하고 구성 가능합니다. 수십 대의 기계에 걸쳐 sweep을 시작할 수 있으며, 노트북에서 sweep을 시작하는 것만큼 쉽습니다.

**[공식 문서 확인하기 $\rightarrow$](https://docs.wandb.com/sweeps)**

## 이 노트북에서 다루는 것



* TensorFlow에서 사용자 정의 트레이닝 루프로 W&B Sweep를 시작하는 간단한 단계.
* 이미지 분류 작업을 위한 최적의 하이퍼파라미터를 찾습니다.

**참고**: _Step_으로 시작하는 섹션은 기존 코드에서 하이퍼파라미터 탐색을 수행하는 데 필요한 모든 것입니다.
그 외의 코드는 간단한 예제를 설정하기 위한 것입니다.

# 🚀 설치, 가져오기 및 로그인

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
from wandb.keras import WandbCallback

wandb.login()
```

> 사이드 노트: W&B를 처음 사용하거나 로그인하지 않은 경우, `wandb.login()`을 실행한 후 나타나는 링크가 가입/로그인 페이지로 이동합니다. 가입은 몇 번의 클릭으로 쉽게 완료할 수 있습니다.

# 👩‍🍳 데이터셋 준비하기


```python
# 트레이닝 데이터셋 준비
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))
```

# 🧠 모델 및 트레이닝 루프 정의하기

## 🏗️ 간단한 분류기 MLP 구축하기


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

### Step 3️⃣: `wandb.log`로 메트릭 로깅하기


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

        # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in tqdm.tqdm(enumerate(train_dataset), total=len(train_dataset)):
            loss_value = train_step(x_batch_train, y_batch_train, 
                                    model, optimizer, 
                                    loss_fn, train_acc_metric)
            train_loss.append(float(loss_value))

        # Run a validation loop at the end of each epoch
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(x_batch_val, y_batch_val, 
                                       model, loss_fn, 
                                       val_acc_metric)
            val_loss.append(float(val_loss_value))
            
        # Display metrics at the end of each epoch
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # Reset metrics at the end of each epoch
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # 3️⃣ wandb.log를 사용하여 메트릭 로깅
        wandb.log({'epochs': epoch,
                   'loss': np.mean(train_loss),
                   'acc': float(train_acc), 
                   'val_loss': np.mean(val_loss),
                   'val_acc':float(val_acc)})
```

# Step 4️⃣: 스윕 구성하기

여기서는 다음을 수행합니다:
* 스윕할 하이퍼파라미터 정의
* 하이퍼파라미터 최적화 방법 제공. `random`, `grid`, `bayes` 방법이 있습니다.
* `bayes`를 사용하는 경우, 예를 들어 `val_loss`를 `최소화`하기 위한 목표와 `metric`을 제공합니다.
* 성능이 낮은 실행을 조기에 종료하기 위해 `hyperband` 사용

#### [스윕 구성에 대해 더 알아보기 $\rightarrow$](https://docs.wandb.com/sweeps/configuration)


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

# Step 5️⃣: 트레이닝 루프 래핑하기

`train`이 호출되기 전에 `wandb.config`를 사용하여 하이퍼파라미터를 설정하는 함수(아래의 `sweep_train`과 같은)가 필요합니다.


```python
def sweep_train(config_defaults=None):
    # 기본값 설정
    config_defaults = {
        "batch_size": 64,
        "learning_rate": 0.01
    }
    # 샘플 프로젝트 이름으로 wandb 초기화
    wandb.init(config=config_defaults)  # 이것은 스윕에서 덮어씁니다

    # 구성에 다른 하이퍼파라미터를 지정하십시오(있는 경우)
    wandb.config.epochs = 2
    wandb.config.log_step = 20
    wandb.config.val_log_step = 50
    wandb.config.architecture_name = "MLP"
    wandb.config.dataset_name = "MNIST"

    # tf.data를 사용하여 입력 파이프라인 구축
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (train_dataset.shuffle(buffer_size=1024)
                                  .batch(wandb.config.batch_size)
                                  .prefetch(buffer_size=tf.data.AUTOTUNE))

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = (val_dataset.batch(wandb.config.batch_size)
                              .prefetch(buffer_size=tf.data.AUTOTUNE))

    # 모델 초기화
    model = Model()

    # 모델을 훈련시키기 위한 옵티마이저 인스턴스화
    optimizer = keras.optimizers.SGD(learning_rate=wandb.config.learning_rate)
    # 손실 함수 인스턴스화
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # 메트릭 준비
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

# Step 6️⃣: 스윕 초기화 및 에이전트 실행하기 


```python
sweep_id = wandb.sweep(sweep_config, project="sweeps-tensorflow")
```

총 실행 수를 `count` 매개변수로 제한할 수 있습니다. 스크립트가 빠르게 실행되도록 10으로 제한할 것이지만, 실행 수를 늘려보고 무슨 일이 일어나는지 확인해 보세요.


```python
wandb.agent(sweep_id, function=sweep_train, count=10)
```

# 👀 결과 시각화하기

위의 **Sweep URL** 링크를 클릭하여 실시간 결과를 확인하세요.

# 🎨 예시 갤러리

[W&B 갤러리 →](https://app.wandb.ai/gallery)에서 W&B로 추적 및 시각화된 프로젝트 예시를 확인하세요.

# 📏 모범 사례
1. **프로젝트**: 여러 실행을 프로젝트에 로그하여 비교합니다. `wandb.init(project="project-name")`
2. **그룹**: 여러 프로세스나 교차검증 폴드의 경우, 각 프로세스를 실행으로 로그하고 함께 그룹화합니다. `wandb.init(group='experiment-1')`
3. **태그**: 현재 베이스라인이나 프로덕션 모델을 추적하기 위해 태그를 추가합니다.
4. **노트**: 실행 간 변경 사항을 추적하기 위해 테이블에 노트를 입력합니다.
5. **리포트**: 진행 상황에 대한 빠른 노트를 동료들과 공유하고 ML 프로젝트의 대시보드와 스냅샷을 만듭니다.

# 🤓 고급 설정
1. [환경 변수](https://docs.wandb.com/library/environment-variables): 관리된 클러스터에서 트레이닝을 실행할 수 있도록 환경 변수에 API 키를 설정합니다.
2. [오프라인 모드](https://docs.wandb.com/library/technical-faq#can-i-run-wandb-offline): `dryrun` 모드를 사용하여 오프라인으로 트레이닝을 하고 나중에 결과를 동기화합니다.
3. [온프레미스](https://docs.wandb.com/self-hosted): W&B를 자체 인프라의 프라이빗 클라우드나 에어갭 서버에 설치합니다. 우리는 학계부터 엔터프라이즈 팀까지 모든 사람을 위한 로컬 설치를 제공합니다.