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

이미 TensorBoard를 사용하고 계시다면, wandb 와 쉽게 연동하실 수 있습니다.

```python
import tensorflow as tf
import wandb
```

## 커스텀 메트릭 기록하기

TensorBoard 에 기록되지 않는 추가적인 커스텀 메트릭을 기록하고 싶다면, 코드에서 `run.log()` 를 호출하시면 됩니다.  
예시: `run.log({"custom": 0.8}) `

Tensorboard 와 동기화할 때는 `run.log()` 의 step 인수 사용이 비활성화됩니다. 만약 다른 step 값을 지정하고 싶으시다면, 아래와 같이 step 메트릭과 함께 기록할 수 있습니다:

``` python
with wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True) as run:
    run.log({"custom": 0.8, "global_step":global_step}, step=global_step)
```

## TensorFlow Estimator 훅

로그에 대해 더 세밀하게 제어하고 싶을 때는 wandb 에서 TensorFlow Estimator 용 훅도 제공합니다. 이 훅은 그래프 내의 모든 `tf.summary` 값을 기록합니다.

```python
import tensorflow as tf
import wandb

run = wandb.init(config=tf.FLAGS)

estimator.train(hooks=[wandb.tensorflow.WandbHook(steps_per_log=1000)])
run.finish()
```

## 직접 기록하기

TensorFlow 에서 메트릭을 기록하는 가장 간단한 방법은 TensorFlow 로거와 함께 `tf.summary` 를 기록하는 것입니다:

```python
import wandb
run = wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)
with tf.Session() as sess:
    # ...
    wandb.tensorflow.log(tf.summary.merge_all())
```

TensorFlow 2에서는 커스텀 루프를 이용해 모델을 트레이닝할 때 `tf.GradientTape` 를 사용하는 것이 권장됩니다.  
자세한 내용은 [TensorFlow 커스텀 트레이닝 워크스루](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough)에서 볼 수 있습니다.  
커스텀 TensorFlow 트레이닝 루프에 wandb 를 활용해 메트릭을 기록하고 싶으시다면 다음 예제를 참고하세요:

```python
    with tf.GradientTape() as tape:
        # 예측값 계산
        predictions = model(features)
        # 손실 값 계산
        loss = loss_func(labels, predictions)

    # 메트릭 기록하기
    run.log("loss": loss.numpy())
    # 그레이디언트 계산
    gradients = tape.gradient(loss, model.trainable_variables)
    # 가중치 업데이트
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

[TensorFlow 2에서 트레이닝 루프를 커스터마이즈하는 전체 예시](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2)도 참고하실 수 있습니다.

## W&B는 TensorBoard와 무엇이 다른가요?

공동 창업자들이 W&B를 만들기 시작했을 때, OpenAI에서 TensorBoard 사용에 답답함을 느꼈던 사용자들을 위해 툴을 만들고자 했습니다. W&B가 개선에 집중한 주요 포인트는 다음과 같습니다:

1. **모델 재현성**: W&B는 실험, 탐험, 그리고 모델을 추후 재현하는 데에 탁월합니다. 메트릭뿐만 아니라 하이퍼파라미터, 코드의 버전, 그리고 버전 컨트롤 상태 및 모델 체크포인트도 저장해 프로젝트를 재현할 수 있게 도와줍니다.
2. **자동 조직화**: 다른 동료의 프로젝트를 이어받거나, 오랜만에 복귀하거나, 예전에 하던 프로젝트를 다시 볼 때도 W&B는 시도된 모든 모델을 손쉽게 확인할 수 있어 불필요한 반복 실험이나 GPU 낭비, 시간·탄소배출을 줄일 수 있습니다.
3. **빠르고 유연한 인테그레이션**: 5분만에 W&B를 프로젝트에 추가하세요. 무료 오픈소스 파이썬 패키지를 설치하고 코드에 몇 줄만 추가하면, 모델을 실행할 때마다 메트릭이 예쁘게 기록되고 기록이 남습니다.
4. **영구적이고 중앙화된 대시보드**: 로컬 컴퓨터, 공유 연구실 클러스터, 혹은 클라우드 스팟 인스턴스 어디에서 모델을 트레이닝하든 결과가 같은 중앙 대시보드에 공유됩니다. 머신마다 TensorBoard 파일을 직접 복사·조직할 필요가 없습니다.
5. **강력한 테이블 기능**: 다양한 모델의 결과를 검색, 필터, 정렬, 그룹화할 수 있습니다. 수천 개의 모델 버전 중에서도 최적의 모델을 손쉽게 찾을 수 있습니다. TensorBoard는 대형 프로젝트에 적합하지 않습니다.
6. **협업을 위한 툴**: W&B를 활용해 복잡한 기계학습 프로젝트도 잘 조직할 수 있습니다. W&B 링크를 쉽게 공유하고, 비공개 팀 기능을 활용해 모두가 결과를 공동 프로젝트에 보낼 수도 있습니다. 리포트 기능으로 인터랙티브 시각화와 마크다운 작성도 지원하므로, 작업 로그를 남기거나, 지도교수께 공유하거나, 연구실/팀에 결과를 발표하기에 좋습니다.

[무료 계정](https://wandb.ai)으로 지금 시작해보세요.

## 예시

연동이 어떻게 동작하는지 확인하실 수 있도록 몇 가지 예제를 준비했습니다:

* [Github 예제](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-estimator-mnist/mnist.py): TensorFlow Estimator를 사용한 MNIST 예제
* [Github 예제](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-cnn-fashion/train.py): 순수 TensorFlow를 사용한 Fashion MNIST 예제
* [Wandb Dashboard](https://app.wandb.ai/l2k2/examples-tf-estimator-mnist/runs/p0ifowcb): W&B에서 결과 보기
* TensorFlow 2에서 트레이닝 루프 커스터마이징하기 - [아티클](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2) | [대시보드](https://app.wandb.ai/sayakpaul/custom_training_loops_tf)