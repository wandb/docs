---
displayed_sidebar: default
---

# TensorFlow

TensorBoard를 이미 사용하고 있다면 wandb와 쉽게 통합할 수 있습니다.

```python
import tensorflow as tf
import wandb
wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)
```

## 커스텀 메트릭

TensorBoard에 로그되지 않는 추가적인 커스텀 메트릭을 로그해야 할 필요가 있다면, 코드에서 `wandb.log`를 호출할 수 있습니다 `wandb.log({"custom": 0.8}) `

Tensorboard와 동기화할 때 `wandb.log`의 스텝 인수를 설정하는 것은 비활성화됩니다. 다른 스텝 카운트를 설정하고 싶다면, 다음과 같이 스텝 메트릭으로 메트릭을 로그할 수 있습니다:

`wandb.log({"custom": 0.8, "global_step":global_step}, step=global_step)`

## TensorFlow 훅

로그되는 내용을 더 많이 제어하고 싶다면, wandb는 TensorFlow 추정기에 대한 훅도 제공합니다. 그래프에서 `tf.summary` 값을 모두 로그합니다.

```python
import tensorflow as tf
import wandb

wandb.init(config=tf.FLAGS)

estimator.train(hooks=[wandb.tensorflow.WandbHook(steps_per_log=1000)])
```

## 수동 로깅

TensorFlow에서 메트릭을 로깅하는 가장 간단한 방법은 TensorFlow 로거를 사용하여 `tf.summary`를 로깅하는 것입니다:

```python
import wandb

with tf.Session() as sess:
    # ...
    wandb.tensorflow.log(tf.summary.merge_all())
```

TensorFlow 2에서는 커스텀 루프를 사용하여 모델을 트레이닝하는 권장 방법이 `tf.GradientTape`를 사용하는 것입니다. 이에 대해 더 자세히 알아보려면 [여기](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough)를 참조하세요. 커스텀 TensorFlow 트레이닝 루프에서 `wandb`를 사용하여 메트릭을 로그하고 싶다면 다음 코드 조각을 따라 할 수 있습니다 -

```python
    with tf.GradientTape() as tape:
        # 확률을 얻음
        predictions = model(features)
        # 손실을 계산함
        loss = loss_func(labels, predictions)

    # 메트릭을 로그함
    wandb.log("loss": loss.numpy())
    # 그레이디언트를 얻음
    gradients = tape.gradient(loss, model.trainable_variables)
    # 가중치를 업데이트함
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

전체 예제는 [여기](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2)에서 확인할 수 있습니다.

## W&B가 TensorBoard와 어떻게 다른가요?

W&B의 공동 창립자들이 OpenAI에서 TensorBoard 사용자들의 불만을 해결하기 위한 툴을 만들겠다는 영감을 받아 W&B를 개발하기 시작했습니다. 다음은 우리가 개선에 초점을 맞춘 몇 가지 사항입니다:

1. **모델 재현**: Weights & Biases는 실험, 탐색 및 나중에 모델을 재현하는 데 유용합니다. 메트릭뿐만 아니라 하이퍼파라미터와 코드의 버전도 캡처하며, 버전 제어 상태와 모델 체크포인트를 저장해 프로젝트를 재현할 수 있게 해줍니다.
2. **자동 조직**: 협업자로부터 프로젝트를 이어받거나, 휴가에서 돌아오거나, 오래된 프로젝트를 다시 시작할 때, W&B를 사용하면 시도한 모든 모델을 쉽게 볼 수 있어 시간, GPU 사이클 또는 탄소를 낭비하여 실험을 다시 실행하는 일이 없습니다.
3. **빠르고 유연한 통합**: W&B를 프로젝트에 5분 만에 추가하세요. 무료 오픈 소스 Python 패키지를 설치하고 코드에 몇 줄을 추가하면 모델을 실행할 때마다 좋은 로그 메트릭과 기록을 얻을 수 있습니다.
4. **영구적이고 중앙 집중화된 대시보드**: 모델을 로컬 머신, 공유된 연구실 클러스터 또는 클라우드의 스팟 인스턴스에서 트레이닝하든, 결과는 같은 중앙 집중화된 대시보드로 공유됩니다. 다른 머신에서 TensorBoard 파일을 복사하고 조직하는 데 시간을 낭비할 필요가 없습니다.
5. **강력한 테이블**: 서로 다른 모델의 결과를 검색, 필터링, 정렬 및 그룹화하는 것이 쉽습니다. 수천 개의 모델 버전을 살펴보고 다양한 작업에 가장 성능이 좋은 모델을 찾을 수 있습니다. TensorBoard는 대규모 프로젝트에서 잘 작동하도록 설계되지 않았습니다.
6. **협업 도구**: W&B를 사용하여 복잡한 기계학습 프로젝트를 조직하세요. W&B에 대한 링크를 공유하기 쉽고, 개인 팀을 사용하여 모든 사람이 공유 프로젝트로 결과를 보내도록 할 수 있습니다. 또한 리포트를 통한 협업을 지원합니다 — 대화형 시각화를 추가하고 마크다운으로 작업을 설명하세요. 이는 작업 로그를 유지하고, 지도자에게 발견한 내용을 공유하거나, 연구실이나 팀에 발견한 내용을 발표하는 데 좋은 방법입니다.

[무료 개인 계정으로 시작하세요 →](https://wandb.ai)

## 예제

통합이 어떻게 작동하는지 보여주기 위해 몇 가지 예제를 만들었습니다:

* [Github에서의 예제](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-estimator-mnist/mnist.py): TensorFlow 추정기를 사용한 MNIST 예제
* [Github에서의 예제](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-cnn-fashion/train.py): Raw TensorFlow를 사용한 Fashion MNIST 예제
* [Wandb 대시보드](https://app.wandb.ai/l2k2/examples-tf-estimator-mnist/runs/p0ifowcb): W&B에서 결과 보기
* TensorFlow 2에서 트레이닝 루프를 커스터마이징하기 - [기사](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2) | [Colab 노트북](https://colab.research.google.com/drive/1JCpAbjkCFhYMT7LCQ399y35TS3jlMpvM) | [대시보드](https://app.wandb.ai/sayakpaul/custom_training_loops_tf)