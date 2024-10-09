---
title: TensorFlow
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

<CTAButtons colabLink="https://colab.research.google.com/drive/1JCpAbjkCFhYMT7LCQ399y35TS3jlMpvM"></CTAButtons>

이미 TensorBoard를 사용하고 있다면, wandb와 쉽게 통합할 수 있습니다.

```python
import tensorflow as tf
import wandb
wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)
```

## 커스텀 메트릭

TensorBoard에 로그되지 않는 추가적인 커스텀 메트릭을 로그하려면, `wandb.log`를 코드에서 호출하세요. `wandb.log({"custom": 0.8})`

Tensorboard와 동기화할 때 `wandb.log`에서 step 인수를 설정할 수 없습니다. 다른 스텝 카운트를 설정하려면, 다음과 같이 스텝 메트릭으로 메트릭을 로그하세요:

`wandb.log({"custom": 0.8, "global_step":global_step}, step=global_step)`

## TensorFlow Hook

로그되는 항목에 대해 더 많은 제어가 필요하다면, wandb는 TensorFlow 추정자에 대한 훅도 제공합니다. 이는 그래프 내의 모든 `tf.summary` 값을 로그합니다.

```python
import tensorflow as tf
import wandb

wandb.init(config=tf.FLAGS)

estimator.train(hooks=[wandb.tensorflow.WandbHook(steps_per_log=1000)])
```

## 수동 로그

TensorFlow에서 메트릭을 로그하는 가장 간단한 방법은 TensorFlow 로거를 사용하여 `tf.summary`를 로그하는 것입니다:

```python
import wandb

with tf.Session() as sess:
    # ...
    wandb.tensorflow.log(tf.summary.merge_all())
```

TensorFlow 2에서는 커스텀 루프를 사용하여 모델을 트레이닝하는 권장 방법은 `tf.GradientTape`를 사용하는 것입니다. 이에 대해 더 읽어보세요 [여기](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough). 자신만의 TensorFlow 트레이닝 루프에 `wandb`를 통합하여 메트릭을 로그하고 싶다면, 다음 코드를 따라하세요 -

```python
    with tf.GradientTape() as tape:
        # 확률을 얻습니다
        predictions = model(features)
        # 손실을 계산합니다
        loss = loss_func(labels, predictions)

    # 메트릭을 로그합니다
    wandb.log("loss": loss.numpy())
    # 그레이디언트를 얻습니다
    gradients = tape.gradient(loss, model.trainable_variables)
    # 가중치를 업데이트합니다
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

전체 예시는 [여기](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2)에서 확인할 수 있습니다.

## W&B와 TensorBoard의 차이점은 무엇인가요?

공동 창업자들이 W&B 작업을 시작할 때, 그들은 OpenAI의 불만을 가진 TensorBoard 사용자들을 위해 툴을 만들어야겠다고 영감을 얻었습니다. 우리가 개선을 위해 집중했던 몇 가지가 있습니다:

1. **모델 재현**: Weights & Biases는 실험, 탐구 및 모델을 나중에 재현하는 데 유리합니다. 우리는 메트릭뿐만 아니라 하이퍼파라미터 및 코드의 버전도 캡처하며, 버전 제어 상태와 모델 체크포인트를 저장하여 프로젝트가 재현 가능하도록 합니다.
2. **자동화된 조직화**: 협력자에게서 프로젝트를 인수하거나, 휴가에서 돌아오거나, 오래된 프로젝트를 다시 살펴본 경우에도 W&B는 시도한 모든 모델을 쉽게 확인할 수 있어 아무도 실험을 반복 실행하여 시간, GPU 사이클, 또는 탄소를 낭비하는 일이 없습니다.
3. **빠르고 유연한 인테그레이션**: 5분 만에 W&B를 프로젝트에 추가하십시오. 우리의 무료 오픈 소스 Python 패키지를 설치하고 코드에 몇 줄을 추가하면, 모델을 실행할 때마다 멋진 로그된 메트릭과 기록을 얻을 수 있습니다.
4. **지속적이고 중앙 집중화된 대시보드**: 로컬 머신, 공유 실험실 클러스터, 클라우드의 스팟 인스턴스 등 어디서든 모델을 트레이닝하더라도 결과는 동일한 중앙 집중화된 대시보드에 공유됩니다. 여러 머신에서 TensorBoard 파일을 복사하고 조직하는 데 시간을 낭비할 필요가 없습니다.
5. **강력한 테이블**: 다양한 모델의 결과를 검색, 필터, 정렬 및 그룹화합니다. 수천 개의 모델 버전을 살펴보고 다양한 작업에 대해 가장 잘 수행되는 모델을 찾는 것이 용이합니다. TensorBoard는 대형 프로젝트에 잘 맞지 않습니다.
6. **협업을 위한 툴**: 복잡한 기계학습 프로젝트를 조직하기 위해 W&B를 사용하세요. W&B의 링크를 쉽게 공유할 수 있으며, 개인 팀을 통해 모든 사람이 결과를 공유 프로젝트로 보낼 수 있습니다. 인터랙티브 시각화를 추가하고 작업을 마크다운으로 설명하면서 리포트를 통해 협업을 지원합니다. 이는 작업 로그를 유지하거나, 슈퍼바이저에게 발견한 내용을 공유하거나, 연구실이나 팀에 발견한 내용을 발표하는 훌륭한 방법입니다.

[무료 계정](https://wandb.ai)으로 시작하세요.

## 예제

인테그레이션이 어떻게 작동하는지 보기 위한 몇 가지 예제를 만들었습니다:

* [GitHub의 예제](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-estimator-mnist/mnist.py): TensorFlow Estimators를 사용한 MNIST 예제
* [GitHub의 예제](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-cnn-fashion/train.py): Raw TensorFlow를 사용한 Fashion MNIST 예제
* [Wandb 대시보드](https://app.wandb.ai/l2k2/examples-tf-estimator-mnist/runs/p0ifowcb): W&B에서 결과 보기
* TensorFlow 2에서의 커스터마이징 트레이닝 루프 - [기사](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2) | [대시보드](https://app.wandb.ai/sayakpaul/custom_training_loops_tf)