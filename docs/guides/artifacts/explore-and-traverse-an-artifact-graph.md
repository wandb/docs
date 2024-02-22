---
description: Traverse automatically created direct acyclic W&B Artifact graphs.
displayed_sidebar: default
---

# 아티팩트 그래프 탐색 및 트래버스

<head>
    <title>직접 비순환 W&B 아티팩트 그래프 탐색.</title>
</head>

W&B는 자동으로 주어진 실행이 기록한 아티팩트와 주어진 실행이 사용한 아티팩트를 추적합니다. W&B 앱 UI 또는 프로그래매틱하게 아티팩트의 계보를 탐색하세요.

## W&B 앱 UI로 아티팩트 트래버스하기

그래프 뷰는 파이프라인의 일반적인 개요를 보여줍니다.

아티팩트 그래프를 보려면:

1. W&B 앱 UI에서 프로젝트로 이동합니다.
2. 왼쪽 패널에서 아티팩트 아이콘을 선택합니다.
3. **계보**를 선택합니다.

실행과 아티팩트를 생성할 때 제공하는 `type`은 그래프를 생성하는 데 사용됩니다. 실행 또는 아티팩트의 입력과 출력은 그래프에서 화살표로 나타납니다. 아티팩트는 파란색 사각형으로, 실행은 녹색 사각형으로 표현됩니다.

제공하는 아티팩트 유형은 **아티팩트** 레이블 옆의 진한 파란색 헤더에 위치합니다. 아티팩트의 이름과 버전은 **아티팩트** 레이블 아래의 연한 파란색 영역에 표시됩니다.

실행을 초기화할 때 제공하는 작업 유형은 **실행** 레이블 옆에 위치합니다. W&B 실행 이름은 **실행** 레이블 아래의 연한 녹색 영역에 위치합니다.

:::info
왼쪽 사이드바와 **계보** 탭에서 아티팩트의 유형과 이름을 모두 볼 수 있습니다.
:::

예를 들어, 다음 이미지에서는 "raw_dataset"이라는 유형으로 정의된 아티팩트(분홍색 사각형)가 있습니다. 아티팩트의 이름은 "MNIST_raw"(분홍색 선)입니다. 그 다음 이 아티팩트는 학습에 사용되었습니다. 학습 실행의 이름은 "vivid-snow-42"입니다. 그 실행은 "mnist-19pofeku"(주황색 사각형)라는 이름의 "모델" 아티팩트를 생성했습니다.

![실험에 사용된 아티팩트, 실행의 DAG 뷰](/images/artifacts/example_dag_with_sidebar.png)

더 상세한 뷰를 보려면, 대시보드 상단 왼쪽에 있는 **Explode** 토글을 선택하세요. 확장된 그래프는 로그된 프로젝트의 모든 실행과 모든 아티팩트의 세부 사항을 보여줍니다. 이 [예제 그래프 페이지](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/v0/lineage)에서 직접 시도해 보세요.

## 프로그래매틱하게 아티팩트 트래버스하기

W&B 공개 API([wandb.Api](../../ref/python/public-api/api.md))를 사용하여 아티팩트 개체를 생성하세요. 프로젝트, 아티팩트 및 아티팩트의 별칭 이름을 제공합니다:

```python
import wandb

api = wandb.Api()

artifact = api.artifact("project/artifact:alias")
```

아티팩트 개체의 [`logged_by`](../../ref/python/artifact.md#logged_by) 및 [`used_by`](../../ref/python/artifact.md#used_by) 메서드를 사용하여 아티팩트에서 그래프를 트래버스하세요:

```python
# 아티팩트에서 그래프를 상하로 트래버스:
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()
```

#### 실행에서 트래버스하기

W&B 공개 API([wandb.Api.Run](../../ref/python/public-api/run.md))를 사용하여 아티팩트 개체를 생성하세요. 엔터티, 프로젝트, 실행 ID의 이름을 제공합니다:

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
```

주어진 실행에서 그래프를 트래버스하기 위해 [`logged_artifacts`](../../ref/python/public-api/run.md#logged_artifacts) 및 [`used_artifacts`](../../ref/python/public-api/run.md#used_artifacts) 메서드를 사용하세요:

```python
# 실행에서 그래프를 상하로 트래버스:
logged_artifacts = run.logged_artifacts()
used_artifacts = run.used_artifacts()
```