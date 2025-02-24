---
title: TensorFlow
menu:
  default:
    identifier: ko-guides-integrations-tensorflow
    parent: integrations
weight: 440
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1JCpAbjkCFhYMT7LCQ399y35TS3jlMpvM" >}}

## 시작하기

이미 TensorBoard를 사용하고 있다면 wandb와 쉽게 통합할 수 있습니다.

```python
import tensorflow as tf
import wandb
wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)
```

## 사용자 정의 메트릭 기록

TensorBoard에 기록되지 않은 추가 사용자 정의 메트릭을 기록해야 하는 경우 코드에서 `wandb.log`를 호출할 수 있습니다. `wandb.log({"custom": 0.8}) `

Tensorboard를 동기화할 때 `wandb.log`에서 step 인수를 설정하는 기능은 꺼져 있습니다. 다른 step 수를 설정하려면 step 메트릭을 사용하여 메트릭을 기록하면 됩니다.

``` python
wandb.log({"custom": 0.8, "global_step":global_step}, step=global_step)
```

## TensorFlow estimator 훅

로그되는 항목을 더 세밀하게 제어하려면 wandb는 TensorFlow estimator에 대한 훅도 제공합니다. 그래프의 모든 `tf.summary` 값이 기록됩니다.

```python
import tensorflow as tf
import wandb

wandb.init(config=tf.FLAGS)

estimator.train(hooks=[wandb.tensorflow.WandbHook(steps_per_log=1000)])
```

## 수동으로 기록

TensorFlow에서 메트릭을 기록하는 가장 간단한 방법은 TensorFlow 로거를 사용하여 `tf.summary`를 기록하는 것입니다.

```python
import wandb

with tf.Session() as sess:
    # ...
    wandb.tensorflow.log(tf.summary.merge_all())
```

TensorFlow 2에서는 사용자 정의 루프를 사용하여 모델을 트레이닝하는 데 권장되는 방법은 `tf.GradientTape`를 사용하는 것입니다. 자세한 내용은 [여기](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough)에서 확인할 수 있습니다. 사용자 정의 TensorFlow 트레이닝 루프에서 메트릭을 기록하기 위해 `wandb`를 통합하려면 다음 스니펫을 따르세요.

```python
    with tf.GradientTape() as tape:
        # Get the probabilities
        predictions = model(features)
        # Calculate the loss
        loss = loss_func(labels, predictions)

    # Log your metrics
    wandb.log("loss": loss.numpy())
    # Get the gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    # Update the weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

전체 예제는 [여기](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2)에서 확인할 수 있습니다.

## W&B는 TensorBoard와 어떻게 다른가요?

W&B 공동 창립자들이 W&B 작업을 시작했을 때 OpenAI에서 좌절한 TensorBoard 사용자들을 위한 툴을 만들도록 영감을 받았습니다. 다음은 개선하는 데 중점을 둔 몇 가지 사항입니다.

1. **모델 재현**: Weights & Biases는 실험, 탐색 및 나중에 모델을 재현하는 데 유용합니다. 메트릭뿐만 아니라 하이퍼파라미터와 코드 버전을 캡처하고, 프로젝트를 재현할 수 있도록 버전 제어 상태와 모델 체크포인트를 저장할 수 있습니다.
2. **자동 구성**: 공동 작업자로부터 프로젝트를 받거나, 휴가에서 돌아오거나, 오래된 프로젝트를 다시 시작할 때 W&B를 사용하면 시도한 모든 모델을 쉽게 볼 수 있으므로 GPU 주기 또는 탄소를 낭비하여 Experiments를 다시 실행하는 사람이 없습니다.
3. **빠르고 유연한 인테그레이션**: 5분 안에 W&B를 프로젝트에 추가하세요. 무료 오픈 소스 Python 패키지를 설치하고 코드에 몇 줄을 추가하면 모델을 실행할 때마다 멋진 기록된 메트릭과 기록이 생성됩니다.
4. **영구적인 중앙 집중식 대시보드**: 로컬 머신, 공유 랩 클러스터 또는 클라우드의 스팟 인스턴스 등 모델을 트레이닝하는 위치에 관계없이 결과는 동일한 중앙 집중식 대시보드에 공유됩니다. 시간을 들여 다른 머신의 TensorBoard 파일을 복사하고 구성할 필요가 없습니다.
5. **강력한 테이블**: 다양한 모델의 결과를 검색, 필터링, 정렬 및 그룹화합니다. 수천 개의 모델 버전을 쉽게 살펴보고 다양한 작업에 가장 적합한 모델을 찾을 수 있습니다. TensorBoard는 대규모 프로젝트에서 잘 작동하도록 구축되지 않았습니다.
6. **협업 툴**: W&B를 사용하여 복잡한 기계 학습 프로젝트를 구성합니다. W&B에 대한 링크를 쉽게 공유할 수 있으며, 비공개 Teams를 사용하여 모든 사람이 결과를 공유 프로젝트로 보낼 수 있습니다. 또한 리포트를 통한 협업도 지원합니다. 대화형 시각화를 추가하고 마크다운으로 작업을 설명합니다. 이는 작업 로그를 유지하고, 지도교수와 발견한 내용을 공유하거나, 랩 또는 팀에 발견한 내용을 발표하는 좋은 방법입니다.

[무료 계정](https://wandb.ai)으로 시작하세요.

## 예제

인테그레이션이 어떻게 작동하는지 보여주는 몇 가지 예제를 만들었습니다.

* [Github의 예제](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-estimator-mnist/mnist.py): TensorFlow Estimator를 사용하는 MNIST 예제
* [Github의 예제](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-cnn-fashion/train.py): Raw TensorFlow를 사용하는 Fashion MNIST 예제
* [Wandb 대시보드](https://app.wandb.ai/l2k2/examples-tf-estimator-mnist/runs/p0ifowcb): W&B에서 결과 보기
* TensorFlow 2에서 트레이닝 루프 사용자 정의 - [기사](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2) | [대시보드](https://app.wandb.ai/sayakpaul/custom_training_loops_tf)
