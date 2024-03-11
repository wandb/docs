---
description: Create custom aliases for W&B Artifacts.
displayed_sidebar: default
---

# 사용자 정의 에일리어스 생성

<head>
    <title>아티팩트에 대한 사용자 정의 에일리어스 생성하기.</title>
</head>

특정 버전을 가리키는 포인터로서 에일리어스를 사용하세요. 기본적으로 `Run.log_artifact`는 로그된 버전에 `latest` 에일리어스를 추가합니다.

아티팩트를 처음 로그할 때 `v0` 버전이 생성되고 아티팩트에 첨부됩니다. 다시 같은 아티팩트에 로그를 남길 때 W&B는 내용의 체크섬을 계산합니다. 아티팩트가 변경된 경우, W&B는 새로운 버전 `v1`을 저장합니다.

예를 들어, 트레이닝 스크립트가 데이터셋의 가장 최근 버전을 가져오도록 하려면, 해당 아티팩트를 사용할 때 `latest`를 지정하십시오. 다음 코드 예시는 `latest`라는 에일리어스를 가진 `bike-dataset`이라는 최근 데이터셋 아티팩트를 다운로드하는 방법을 보여줍니다:

```python
import wandb

run = wandb.init(project="<example-project>")

artifact = run.use_artifact("bike-dataset:latest")

artifact.download()
```

또한 아티팩트 버전에 사용자 정의 에일리어스를 적용할 수 있습니다. 예를 들어, 모델 체크포인트가 AP-50 메트릭에서 최고라고 표시하고 싶다면, 모델 아티팩트를 로그할 때 `'best-ap50'` 문자열을 에일리어스로 추가할 수 있습니다.

```python
artifact = wandb.Artifact("run-3nq3ctyy-bike-model", type="model")
artifact.add_file("model.h5")
run.log_artifact(artifact, aliases=["latest", "best-ap50"])
```