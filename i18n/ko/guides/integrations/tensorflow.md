---
displayed_sidebar: default
---

# TensorFlow

TensorBoard를 이미 사용 중이라면 wandb와 통합하기가 쉽습니다.

```python
import tensorflow as tf
import wandb
wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)
```

## 사용자 정의 메트릭

TensorBoard에 로그되지 않는 추가 사용자 정의 메트릭을 로그해야 하는 경우 코드에서 `wandb.log`를 호출할 수 있습니다 `wandb.log({"custom": 0.8}) `

Tensorboard와 동기화할 때 `wandb.log`에서 step 인수를 설정하는 것은 비활성화됩니다. 다른 스텝 수를 설정하고 싶다면 스텝 메트릭을 사용하여 메트릭을 로그할 수 있습니다:

`wandb.log({"custom": 0.8, "global_step":global_step}, step=global_step)`

## TensorFlow 후크

로그할 내용을 더 많이 제어하고 싶다면, wandb는 TensorFlow 추정기를 위한 후크도 제공합니다. 그래프의 모든 `tf.summary` 값이 로그됩니다.

```python
import tensorflow as tf
import wandb

wandb.init(config=tf.FLAGS)

estimator.train(hooks=[wandb.tensorflow.WandbHook(steps_per_log=1000)])
```

## 수동 로깅

TensorFlow에서 메트릭을 로깅하는 가장 간단한 방법은 TensorFlow 로거와 함께 `tf.summary`를 로깅하는 것입니다:

```python
import wandb

with tf.Session() as sess:
    # ...
    wandb.tensorflow.log(tf.summary.merge_all())
```

TensorFlow 2에서 사용자 정의 루프를 사용하여 모델을 학습하는 권장 방법은 `tf.GradientTape`를 사용하는 것입니다. 이에 대해 더 자세히 알아보려면 [여기](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough)를 참조하십시오. 사용자 정의 TensorFlow 학습 루프에서 메트릭을 로그하기 위해 `wandb`를 통합하려면 이 코드 조각을 따르십시오 -

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

전체 예제는 [여기](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2)에서 확인할 수 있습니다.

## W&B가 TensorBoard와 어떻게 다른가요?

W&B의 공동 창립자들이 OpenAI에서 TensorBoard 사용자들을 위한 도구를 만들기 위해 영감을 받아 W&B 작업을 시작했습니다. 우리가 개선에 중점을 둔 몇 가지 사항은 다음과 같습니다:

1. **모델 재현**: Weights & Biases는 실험, 탐색 및 나중에 모델을 재현하기에 좋습니다. 우리는 메트릭뿐만 아니라 하이퍼파라미터와 코드의 버전도 캡처하며, 버전 제어 상태와 모델 체크포인트를 저장하여 프로젝트를 재현할 수 있도록 도와줍니다.
2. **자동 조직**: 협업자로부터 프로젝트를 이어받거나, 휴가에서 돌아오거나, 오래된 프로젝트를 다시 시작할 때, W&B는 시도된 모든 모델을 쉽게 볼 수 있도록 하여 누군가가 실험을 다시 실행하는 데 시간, GPU 사이클 또는 탄소를 낭비하지 않도록 도와줍니다.
3. **빠르고 유연한 통합**: 프로젝트에 W&B를 5분 만에 추가하세요. 우리의 무료 오픈 소스 파이썬 패키지를 설치하고 코드에 몇 줄을 추가하면 모델을 실행할 때마다 좋은 로그된 메트릭과 기록을 얻을 수 있습니다.
4. **지속적이고 중앙집중식 대시보드**: 모델을 로컬 머신, 공유 랩 클러스터 또는 클라우드의 스팟 인스턴스에서 학습하는지 여부와 관계없이 결과는 동일한 중앙집중식 대시보드로 공유됩니다. 다른 기계에서 TensorBoard 파일을 복사하고 정리하는 데 시간을 소비할 필요가 없습니다.
5. **강력한 테이블**: 다른 모델에서 나온 결과를 검색, 필터링, 정렬 및 그룹화할 수 있습니다. 대규모 프로젝트에서 잘 작동하도록 설계되지 않은 TensorBoard와 달리, 다양한 작업에 가장 적합한 모델을 쉽게 찾을 수 있습니다.
6. **협업 도구**: W&B를 사용하여 복잡한 머신 러닝 프로젝트를 정리합니다. W&B에 링크를 공유하기 쉽고, 개인 팀을 사용하여 모든 사람이 공유 프로젝트에 결과를 보내도록 할 수 있습니다. 우리는 또한 마크다운으로 작업을 설명하고 대화형 시각화를 추가하는 리포트를 통해 협업을 지원합니다. 이는 작업 로그를 유지하고, 감독관에게 발견한 내용을 공유하거나, 연구실이나 팀에 발견한 내용을 발표하는 데 좋은 방법입니다.

[무료 개인 계정으로 시작하기 →](https://wandb.ai)

## 예제

통합 방식을 확인할 수 있는 몇 가지 예제를 만들었습니다:

* [Github에서의 예제](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-estimator-mnist/mnist.py): TensorFlow 추정기를 사용한 MNIST 예제
* [Github에서의 예제](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-cnn-fashion/train.py): Raw TensorFlow를 사용한 Fashion MNIST 예제
* [Wandb 대시보드](https://app.wandb.ai/l2k2/examples-tf-estimator-mnist/runs/p0ifowcb): W&B에서 결과 보기
* TensorFlow 2에서 학습 루프 사용자 정의 - [기사](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2) | [Colab 노트북](https://colab.research.google.com/drive/1JCpAbjkCFhYMT7LCQ399y35TS3jlMpvM) | [대시보드](https://app.wandb.ai/sayakpaul/custom_training_loops_tf)