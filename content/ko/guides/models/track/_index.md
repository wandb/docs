---
title: Experiments
description: W&B로 기계 학습 experiments 를 추적하세요.
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

몇 줄의 코드로 기계 학습 실험을 추적하세요. 그런 다음 [대화형 대시보드]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})에서 결과를 검토하거나, [Public API]({{< relref path="/ref/python/public-api/" lang="ko" >}})를 사용하여 프로그래밍 방식으로 엑세스할 수 있도록 데이터를 Python으로 내보낼 수 있습니다.

[PyTorch]({{< relref path="/guides/integrations/pytorch.md" lang="ko" >}}), [Keras]({{< relref path="/guides/integrations/keras.md" lang="ko" >}}) 또는 [Scikit]({{< relref path="/guides/integrations/scikit.md" lang="ko" >}})과 같은 널리 사용되는 프레임워크를 사용하는 경우 W&B Integrations를 활용하세요. 통합에 대한 전체 목록과 W&B를 코드에 추가하는 방법에 대한 정보는 [Integration 가이드]({{< relref path="/guides/integrations/" lang="ko" >}})를 참조하세요.

{{< img src="/images/experiments/experiments_landing_page.png" alt="" >}}

위의 이미지는 여러 [runs]({{< relref path="/guides/models/track/runs/" lang="ko" >}})에서 메트릭을 보고 비교할 수 있는 대시보드 예를 보여줍니다.

## 작동 방식

몇 줄의 코드로 기계 학습 실험을 추적하세요.
1. [W&B run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})을 만듭니다.
2. 학습률 또는 모델 유형과 같은 하이퍼파라미터 사전을 구성 ([`wandb.config`]({{< relref path="/guides/models/track/config.md" lang="ko" >}}))에 저장합니다.
3. 정확도 및 손실과 같이 트레이닝 루프에서 시간 경과에 따른 메트릭 ([`wandb.log()`]({{< relref path="/guides/models/track/log/" lang="ko" >}}))을 기록합니다.
4. 모델 가중치 또는 예측 테이블과 같은 run의 출력을 저장합니다.

다음 의사 코드는 일반적인 W&B Experiment 추적 워크플로우를 보여줍니다.

```python showLineNumbers
# 1. W&B Run 시작
wandb.init(entity="", project="my-project-name")

# 2. 모드 입력 및 하이퍼파라미터 저장
wandb.config.learning_rate = 0.01

# 모델 및 데이터 임포트
model, dataloader = get_model(), get_data()

# 모델 트레이닝 코드가 여기에 들어갑니다

# 3. 시간 경과에 따른 메트릭을 기록하여 성능 시각화
wandb.log({"loss": loss})

# 4. W&B에 아티팩트 기록
wandb.log_artifact(model)
```

## 시작 방법

유스 케이스에 따라 다음 리소스를 탐색하여 W&B Experiments를 시작하세요.

* 데이터셋 artifact를 생성, 추적 및 사용하는 데 사용할 수 있는 W&B Python SDK 코맨드에 대한 단계별 개요는 [W&B 퀵스타트]({{< relref path="/guides/quickstart.md" lang="ko" >}})를 참조하세요.
* 다음 방법을 배우려면 이 챕터를 탐색하세요.
  * 실험 만들기
  * 실험 구성
  * 실험에서 데이터 기록
  * 실험 결과 보기
* [W&B API Reference Guide]({{< relref path="/ref/" lang="ko" >}}) 내에서 [W&B Python Library]({{< relref path="/ref/python/" lang="ko" >}})를 탐색하세요.
