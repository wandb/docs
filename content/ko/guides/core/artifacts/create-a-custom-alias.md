---
title: Artifacts 에일리어스 생성
description: W&B Artifacts에 대해 사용자 지정 별칭을 생성하세요.
menu:
  default:
    identifier: ko-guides-core-artifacts-create-a-custom-alias
    parent: artifacts
weight: 5
---

에일리어스를 사용하면 특정 버전을 쉽게 지정할 수 있습니다. 기본적으로, `Run.log_artifact` 는 기록된 버전에 `latest` 에일리어스를 자동으로 추가합니다.

처음 아티팩트를 저장하면, 아티팩트 버전 `v0` 이 생성되고 해당 아티팩트에 연결됩니다. 동일한 아티팩트에 다시 저장할 경우, W&B 는 콘텐츠의 체크섬을 계산하여 변경 사항이 있으면 새로운 버전인 `v1` 을 만듭니다.

예를 들어, 트레이닝 스크립트에서 항상 가장 최신 데이터셋 버전을 사용하고 싶다면, 아티팩트를 사용할 때 `latest` 를 지정하면 됩니다. 아래 예시는 `latest` 에일리어스를 가진 `bike-dataset` 데이터셋 아티팩트를 다운로드하는 방법을 보여줍니다.

```python
import wandb

run = wandb.init(project="<example-project>")

artifact = run.use_artifact("bike-dataset:latest")

artifact.download()
```

또한, 원하는 에일리어스를 직접 아티팩트 버전에 추가할 수도 있습니다. 예를 들어, 특정 모델 체크포인트가 AP-50 지표에서 가장 좋은 성능을 냈다는 것을 표시하고 싶다면, 모델 아티팩트를 저장할 때 `'best-ap50'` 에일리어스를 함께 추가할 수 있습니다.

```python
artifact = wandb.Artifact("run-3nq3ctyy-bike-model", type="model")
artifact.add_file("model.h5")
run.log_artifact(artifact, aliases=["latest", "best-ap50"])
```