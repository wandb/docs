---
title: 시맨틱 run 플롯 범례
description: 차트에 대한 시맨틱 범례 만들기
menu:
  default:
    identifier: ko-guides-models-track-runs-color-code-runs
    parent: what-are-runs
---

W&B run 들을 메트릭이나 설정 파라미터로 색상 코딩하여, 시각적으로 의미 있는 선 그래프와 범례를 만들 수 있습니다. 실험 전체에서 run 들을 성능 메트릭(최고, 최저 또는 최신 값)에 따라 색상으로 구분하면 패턴과 트렌드를 쉽게 파악할 수 있습니다. W&B는 선택한 파라미터의 값에 따라 run 들을 자동으로 색상별 버킷으로 그룹화합니다.

워크스페이스의 설정 페이지에서 run 들의 색상을 메트릭 또는 설정 값 기준으로 지정할 수 있습니다.

1. W&B 프로젝트로 이동합니다.
2. 프로젝트 사이드바에서 **Workspace** 탭을 선택합니다.
3. 오른쪽 상단의 **Settings** 아이콘(⚙️)을 클릭합니다.
4. 드로어에서 **Runs**를 선택한 후 **Key-based colors**를 선택합니다.
    - **Key** 드롭다운에서 run 에 색상을 지정할 때 사용할 메트릭을 선택합니다.
    - **Y value** 드롭다운에서 run 에 색상을 지정할 때 사용할 y 값을 선택합니다.
    - 버킷 개수는 2에서 8 사이의 값으로 설정할 수 있습니다.

아래 섹션에서는 메트릭과 y 값을 설정하는 방법 및 run 에 색상을 할당할 때 사용할 버킷을 커스터마이즈하는 방법을 설명합니다.

## 메트릭 설정하기

**Key** 드롭다운에 표시되는 메트릭 옵션은 [W&B에 로그한 key-value 쌍]({{< relref path="guides/models/track/runs/color-code-runs/#custom-metrics" lang="ko" >}})과 W&B에서 정의한 [기본 메트릭]({{< relref path="guides/models/track/runs/color-code-runs/#default-metrics" lang="ko" >}})에서 가져옵니다.

### 기본 메트릭

* `Relative Time (Process)`: run 이 시작된 이후 경과한 초단위 상대적 시간(프로세스 타임 기준)입니다.
* `Relative Time (Wall)`: run 이 시작된 이후 경과한 초단위 상대적 시간(월 클록 기준, 즉 실제 시간 기준)입니다.
* `Wall Time`: run 의 월 클록 시간을 에포크 기준 초로 나타냅니다.
* `Step`: run 의 스텝 번호로, 주로 트레이닝이나 평가 과정의 진행 현황을 나타냅니다.

### 커스텀 메트릭

트레이닝 또는 평가 스크립트에서 로그한 커스텀 메트릭으로 run 들을 색상별로 표시하고, 의미 있는 그래프 범례를 만들 수 있습니다. 커스텀 메트릭은 키(key)가 메트릭 이름이고 값(value)이 해당 값인 key-value 쌍으로 로그됩니다.

예를 들어, 아래 코드조각은 트레이닝 루프에서 정확도(`"acc"` 키)와 손실(`"loss"` 키)를 로그하는 방법을 보여줍니다.

```python
import wandb
import random

epochs = 10

with wandb.init(project="basic-intro") as run:
  # 블록은 메트릭을 로그하는 트레이닝 루프를 시뮬레이션합니다
  offset = random.random() / 5
  for epoch in range(2, epochs):
      acc = 1 - 2 ** -epoch - random.random() / epoch - offset
      loss = 2 ** -epoch + random.random() / epoch + offset

      # 스크립트에서 W&B로 메트릭을 로그합니다
      run.log({"acc": acc, "loss": loss})
```

**Key** 드롭다운에는 `"acc"`, `"loss"` 모두 표시됩니다.

## 설정 키 지정하기

**Key** 드롭다운에 표시되는 설정 옵션은 W&B run 을 초기화할 때 `config` 파라미터로 전달한 key-value 쌍에서 나옵니다. 설정 키는 주로 하이퍼파라미터나 트레이닝/평가 스크립트에 사용된 기타 설정 값 로그에 활용됩니다.

```python
import wandb

config = {
  "learning_rate": 0.01,
  "batch_size": 32,
  "optimizer": "adam"
}

with wandb.init(project="basic-intro", config=config) as run:
  # 여기에 트레이닝 코드를 작성합니다
  pass
```

**Key** 드롭다운에서 `"learning_rate"`, `"batch_size"`, `"optimizer"`를 선택할 수 있습니다.

## y 값 지정하기

다음 옵션 중에서 선택할 수 있습니다:

- **Latest**: 각 선의 마지막으로 로그된 Y 값 기준으로 색상 결정
- **Max**: 해당 메트릭에 대해 가장 높은 Y 값 기준으로 색상 결정
- **Min**: 해당 메트릭에 대해 가장 낮은 Y 값 기준으로 색상 결정

## 버킷 커스터마이즈하기

버킷은 선택한 메트릭 또는 설정 키의 값 범위에 따라 W&B가 run 을 분류하는 기준입니다. 버킷은 지정한 메트릭 또는 설정 키의 전체 값 범위를 균등하게 나눈 구간이며, 각 버킷마다 고유의 색상이 할당됩니다. 해당 버킷 값 범위에 속하는 run들은 그 색상으로 표시됩니다.

다음 예시를 참고하세요.

{{< img src="/images/track/color-coding-runs.png" alt="Color coded runs" >}}

- **Key**는 `"Accuracy"`(축약해서 `"acc"`)로 설정되어 있습니다.
- **Y value**는 `"Max"`로 지정되어 있습니다.

이 설정에서는 각 run 의 정확도 값에 따라 색상이 지정됩니다. 색상은 연한 노란색에서 진한 색까지 분포합니다. 연한 색상은 정확도 값이 낮을수록, 진한 색상은 정확도 값이 높을수록 나타납니다.

이 메트릭에 대해 6개의 버킷이 정의되어 있으며, 각 버킷은 다음과 같이 정확도 값 범위를 나타냅니다.

- 버킷 1: (Min - 0.7629)
- 버킷 2: (0.7629 - 0.7824)
- 버킷 3: (0.7824 - 0.8019)
- 버킷 4: (0.8019 - 0.8214)
- 버킷 5: (0.8214 - 0.8409)
- 버킷 6: (0.8409 - Max)

아래 선 그래프에서 정확도(0.8232)가 가장 높은 run 은 진한 보라색(버킷 5)으로, 정확도(0.7684)가 가장 낮은 run 은 연한 주황색(버킷 2)으로 표시됩니다. 나머지 run 들도 정확도 값에 따라 색상이 지정되어, 색상 그라데이션을 통해 상대적인 성능을 시각적으로 확인할 수 있습니다.

{{< img src="/images/track/color-code-runs-plot.png" alt="Color coded runs plot" >}}