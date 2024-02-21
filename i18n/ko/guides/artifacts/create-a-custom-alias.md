---
description: Create custom aliases for W&B Artifacts.
displayed_sidebar: default
---

# 사용자 정의 별칭 만들기

<head>
    <title>아티팩트에 대한 사용자 정의 별칭 만들기.</title>
</head>

별칭을 특정 버전을 가리키는 포인터로 사용하세요. 기본적으로, `Run.log_artifact`은 로그된 버전에 `latest` 별칭을 추가합니다.

아티팩트를 처음 로그할 때 `v0` 버전이 생성되어 아티팩트에 첨부됩니다. 다시 같은 아티팩트에 로그할 때 W&B는 내용의 체크섬을 계산합니다. 아티팩트가 변경된 경우, W&B는 새 버전 `v1`을 저장합니다.

예를 들어, 학습 스크립트가 데이터세트의 가장 최신 버전을 불러오길 원한다면, 해당 아티팩트를 사용할 때 `latest`를 지정하세요. 다음 코드 예제는 별칭 `latest`가 있는 최근 데이터세트 아티팩트 `bike-dataset`을 다운로드하는 방법을 보여줍니다:

```python
import wandb

run = wandb.init(project="<example-project>")

artifact = run.use_artifact("bike-dataset:latest")

artifact.download()
```

또한 아티팩트 버전에 사용자 정의 별칭을 적용할 수도 있습니다. 예를 들어, 모델 체크포인트가 AP-50 지표에서 최고라는 것을 표시하고 싶다면, 모델 아티팩트를 로그할 때 문자열 `'best-ap50'`를 별칭으로 추가할 수 있습니다.

```python
artifact = wandb.Artifact("run-3nq3ctyy-bike-model", type="model")
artifact.add_file("model.h5")
run.log_artifact(artifact, aliases=["latest", "best-ap50"])
```