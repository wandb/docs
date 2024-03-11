---
description: Traverse automatically created direct acyclic W&B Artifact graphs.
displayed_sidebar: default
---

# 아티팩트 그래프 탐색 및 트래버스

<head>
    <title>W&B에서 방향성 비순환 아티팩트 그래프 탐색.</title>
</head>

W&B는 자동으로 특정 run이 기록한 아티팩트와 사용한 아티팩트를 추적합니다. W&B App UI 또는 프로그래밍적으로 아티팩트의 계보를 탐색하세요.

## W&B App UI로 아티팩트 트래버스하기

그래프 뷰는 파이프라인의 일반적인 개요를 보여줍니다.

아티팩트 그래프를 보려면:

1. W&B App UI에서 프로젝트로 이동합니다.
2. 왼쪽 패널에서 아티팩트 아이콘을 선택합니다.
3. **계보**를 선택합니다.

run과 아티팩트를 생성할 때 제공하는 `type`은 그래프를 생성하는 데 사용됩니다. run 또는 아티팩트의 입력과 출력은 화살표로 그래프에 표시됩니다. 아티팩트는 파란색 직사각형으로, Run은 초록색 직사각형으로 표시됩니다.

제공하는 아티팩트 타입은 **ARTIFACT** 라벨 옆의 짙은 파란색 헤더에 위치합니다. 아티팩트 이름과 함께 아티팩트 버전은 **ARTIFACT** 라벨 아래의 밝은 파란색 영역에 표시됩니다.

run을 초기화할 때 제공하는 job 유형은 **RUN** 라벨 옆에 위치합니다. W&B run 이름은 **RUN** 라벨 아래의 연두색 영역에 위치합니다.

:::info
왼쪽 사이드바와 **계보** 탭에서 아티팩트의 타입과 이름을 모두 볼 수 있습니다.
:::

예를 들어, 다음 이미지에서 "raw_dataset"이라는 타입으로 정의된 아티팩트(분홍색 사각형)가 있습니다. 아티팩트의 이름은 "MNIST_raw"(분홍색 선)입니다. 그런 다음 해당 아티팩트는 트레이닝에 사용되었습니다. 트레이닝 run의 이름은 "vivid-snow-42"입니다. 그 run은 "모델" 아티팩트(주황색 사각형) "mnist-19pofeku"를 생성했습니다.

![실험에 사용된 아티팩트, run의 DAG 뷰.](/images/artifacts/example_dag_with_sidebar.png)

자세한 뷰를 보려면, 대시보드의 상단 왼쪽에 있는 **Explode** 토글을 선택하세요. 확장된 그래프는 프로젝트에 기록된 모든 run과 모든 아티팩트의 세부 정보를 보여줍니다. 이 [예제 그래프 페이지](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/v0/lineage)에서 직접 시도해 보세요.

## 프로그래밍적으로 아티팩트 트래버스하기

W&B Public API([wandb.Api](../../ref/python/public-api/api.md))를 사용하여 아티팩트 오브젝트를 생성합니다. 프로젝트, 아티팩트, 아티팩트의 에일리어스 이름을 제공하세요:

```python
import wandb

api = wandb.Api()

artifact = api.artifact("project/artifact:alias")
```

아티팩트 오브젝트의 [`logged_by`](../../ref/python/artifact.md#logged_by) 및 [`used_by`](../../ref/python/artifact.md#used_by) 메소드를 사용하여 아티팩트에서 그래프를 따라 걸어갑니다:

```python
# 아티팩트에서 그래프를 따라 위아래로 걷기:
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()
```

#### run에서 트래버스하기

W&B Public API([wandb.Api.Run](../../ref/python/public-api/run.md))를 사용하여 아티팩트 오브젝트를 생성합니다. 엔티티, 프로젝트, run ID의 이름을 제공하세요:

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
```

주어진 run에서 그래프를 따라 걷기 위해 [`logged_artifacts`](../../ref/python/public-api/run.md#logged_artifacts) 및 [`used_artifacts`](../../ref/python/public-api/run.md#used_artifacts) 메소드를 사용하세요:

```python
# run에서 그래프를 따라 위아래로 걷기:
logged_artifacts = run.logged_artifacts()
used_artifacts = run.used_artifacts()
```