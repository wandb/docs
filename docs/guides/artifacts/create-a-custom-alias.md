---
title: Create an artifact alias
description: W&B Artifacts에 대한 맞춤형 에일리어스를 생성합니다.
displayed_sidebar: default
---

에일리어스를 특정 버전의 포인터로 사용하세요. 기본적으로, `Run.log_artifact`는 로그된 버전에 `latest` 에일리어스를 추가합니다.

아티팩트를 처음으로 로그할 때 아티팩트 버전 `v0`가 생성되어 아티팩트에 첨부됩니다. 동일한 아티팩트에 다시 로그할 때 W&B는 내용을 체크섬합니다. 아티팩트가 변경되면, W&B는 새 버전 `v1`을 저장합니다.

예를 들어, 트레이닝 스크립트가 데이터셋의 가장 최근 버전을 가져오려면, 해당 아티팩트를 사용할 때 `latest`를 지정하세요. 다음 코드 예제는 에일리어스 `latest`가 있는 `bike-dataset`이라는 최근 데이터셋 아티팩트를 다운로드하는 방법을 보여줍니다:

```python
import wandb

run = wandb.init(project="<example-project>")

artifact = run.use_artifact("bike-dataset:latest")

artifact.download()
```

또한 아티팩트 버전에 사용자 정의 에일리어스를 적용할 수 있습니다. 예를 들어, 모델 체크포인트가 메트릭 AP-50에서 최고임을 표시하고 싶다면, 모델 아티팩트를 로그할 때 문자열 `'best-ap50'`을 에일리어스로 추가할 수 있습니다.

```python
artifact = wandb.Artifact("run-3nq3ctyy-bike-model", type="model")
artifact.add_file("model.h5")
run.log_artifact(artifact, aliases=["latest", "best-ap50"])
```