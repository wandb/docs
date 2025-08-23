---
title: Experiments
description: W&B로 머신러닝 실험을 추적하세요.
cascade:
- url: guides/track/:filename
menu:
  default:
    identifier: ko-guides-models-track-_index
    parent: w-b-models
url: guides/track
weight: 1
---

{{< cta-button productLink="https://wandb.ai/stacey/deep-drive/workspace?workspace=user-lavanyashukla" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb" >}}

몇 줄의 코드만으로 머신러닝 실험을 추적하세요. 그 후 [대시보드에서 직관적으로 결과를 확인]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})하거나, [Public API]({{< relref path="/ref/python/public-api/index.md" lang="ko" >}})를 통해 데이터를 Python으로 내보내 프로그래밍 방식으로 접근할 수 있습니다.

PyTorch, [Keras]({{< relref path="/guides/integrations/keras.md" lang="ko" >}}), [Scikit]({{< relref path="/guides/integrations/scikit.md" lang="ko" >}}) 등 인기 있는 프레임워크가 있다면, W&B 인테그레이션을 활용해 보세요. 더 다양한 인테그레이션과 코드 적용 방법은 [Integration 가이드]({{< relref path="/guides/integrations/" lang="ko" >}})에서 확인할 수 있습니다.

{{< img src="/images/experiments/experiments_landing_page.png" alt="Experiments dashboard" >}}

위 이미지는 여러 [run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})의 메트릭을 한눈에 비교할 수 있는 대시보드 예시입니다.

## 작동 방식

아래와 같은 과정으로 쉽게 머신러닝 실험을 추적할 수 있습니다:
1. [W&B Run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})을 생성합니다.
2. 하이퍼파라미터(예: learning rate, 모델 타입 등) 사전을 설정에 저장합니다([`wandb.Run.config`]({{< relref path="./config.md" lang="ko" >}})).
3. 트레이닝 루프에서 정확도, 손실 등 메트릭을 [`wandb.Run.log()`]({{< relref path="/guides/models/track/log/" lang="ko" >}})로 꾸준히 기록합니다.
4. 모델 가중치나 예측 결과 등의 산출물을 run에 저장합니다.

아래는 일반적인 W&B 실험 추적 워크플로우 예시입니다:

```python
# run을 시작합니다.
#
# 이 블록을 벗어나면, 기록된 데이터가 업로드될 때까지 대기합니다.
# 만약 예외가 발생하면 해당 run은 실패로 표기됩니다.
with wandb.init(entity="", project="my-project-name") as run:
  # 모델 입력값과 하이퍼파라미터를 저장합니다.
  run.config.learning_rate = 0.01

  # 실험 코드를 실행합니다.
  for epoch in range(num_epochs):
    # 트레이닝을 수행...

    # 시간 경과에 따라 메트릭을 기록해 모델 성능을 시각화합니다.
    run.log({"loss": loss})

  # 모델 산출물을 artifact로 업로드합니다.
  run.log_artifact(model)
```

## 시작하기

여러분의 상황에 맞는 W&B Experiments 시작법은 아래 리소스를 참고하세요:

* [W&B 퀵스타트]({{< relref path="/guides/quickstart.md" lang="ko" >}})에서는 W&B Python SDK 명령어로 Datasets artifact를 만들고 추적하며 활용하는 순서를 단계별로 배울 수 있습니다.
* 이 챕터에서는 다음 내용을 다룹니다:
  * 실험 만들기
  * 실험 설정하기
  * 실험에서 데이터 기록하기
  * 실험 결과 확인하기
* [W&B API Reference Guide]({{< relref path="/ref/" lang="ko" >}})와 [W&B Python Library]({{< relref path="/ref/python/index.md" lang="ko" >}}) 문서도 함께 활용해 보세요.

## 모범 사례와 팁

Experiments와 로깅의 모범 사례와 팁은 [Best Practices: Experiments and Logging](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#w&b-experiments-and-logging)에서 확인할 수 있습니다.