---
title: Create an artifact alias
description: W&B Artifacts에 대한 사용자 지정 에일리어스를 만드세요.
menu:
  default:
    identifier: ko-guides-core-artifacts-create-a-custom-alias
    parent: artifacts
weight: 5
---

에일리어스를 특정 버전의 포인터로 사용하세요. 기본적으로 `Run.log_artifact`는 기록된 버전에 `latest` 에일리어스를 추가합니다.

아티팩트를 처음 기록할 때 아티팩트 버전 `v0`가 생성되어 아티팩트에 연결됩니다. W&B는 동일한 아티팩트에 다시 기록할 때 콘텐츠의 체크섬을 계산합니다. 아티팩트가 변경되면 W&B는 새 버전 `v1`을 저장합니다.

예를 들어, 트레이닝 스크립트가 데이터셋의 최신 버전을 가져오도록 하려면 해당 아티팩트를 사용할 때 `latest`를 지정하세요. 다음 코드 예제는 `latest`라는 에일리어스가 있는 `bike-dataset`이라는 최신 데이터셋 아티팩트를 다운로드하는 방법을 보여줍니다.

```python
import wandb

run = wandb.init(project="<example-project>")

artifact = run.use_artifact("bike-dataset:latest")

artifact.download()
```

사용자 지정 에일리어스를 아티팩트 버전에 적용할 수도 있습니다. 예를 들어, 해당 모델 체크포인트가 AP-50 메트릭에서 가장 좋다고 표시하려면 모델 아티팩트를 기록할 때 문자열 `'best-ap50'`을 에일리어스로 추가하면 됩니다.

```python
artifact = wandb.Artifact("run-3nq3ctyy-bike-model", type="model")
artifact.add_file("model.h5")
run.log_artifact(artifact, aliases=["latest", "best-ap50"])
```