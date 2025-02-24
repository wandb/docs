---
title: 'Tutorial: Log tables, visualize and query data'
description: W&B Tables 사용법을 5분 퀵스타트 가이드에서 알아보세요.
menu:
  default:
    identifier: ko-guides-core-tables-tables-walkthrough
    parent: tables
weight: 1
---

다음 퀵스타트는 데이터 테이블을 기록하고, 데이터를 시각화하고, 데이터를 쿼리하는 방법을 보여줍니다.

아래 버튼을 선택하여 MNIST 데이터에 대한 PyTorch 퀵스타트 예제 프로젝트를 사용해 보세요.

## 1. 테이블 기록
W&B로 테이블을 기록합니다. 새 테이블을 구성하거나 Pandas Dataframe을 전달할 수 있습니다.

{{< tabpane text=true >}}
{{% tab header="테이블 구성" value="construct" %}}
새로운 테이블을 구성하고 기록하려면 다음을 사용합니다:
- [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ko" >}}): 결과를 추적하기 위해 [run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})을 생성합니다.
- [`wandb.Table()`]({{< relref path="/ref/python/data-types/table.md" lang="ko" >}}): 새로운 테이블 오브젝트를 생성합니다.
  - `columns`: 열 이름을 설정합니다.
  - `data`: 각 행의 내용을 설정합니다.
- [`run.log()`]({{< relref path="/ref/python/log.md" lang="ko" >}}): 테이블을 기록하여 W&B에 저장합니다.

다음은 예제입니다:
```python
import wandb

run = wandb.init(project="table-test")
# Create and log a new table.
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```
{{% /tab %}}

{{% tab header="Pandas Dataframe" value="pandas"%}}
Pandas Dataframe을 `wandb.Table()`에 전달하여 새 테이블을 생성합니다.

```python
import wandb
import pandas as pd

df = pd.read_csv("my_data.csv")

run = wandb.init(project="df-table")
my_table = wandb.Table(dataframe=df)
wandb.log({"Table Name": my_table})
```

지원되는 데이터 유형에 대한 자세한 내용은 W&B API 참조 가이드의 [`wandb.Table`]({{< relref path="/ref/python/data-types/table.md" lang="ko" >}})을 참조하십시오.
{{% /tab %}}
{{< /tabpane >}}

## 2. 프로젝트 워크스페이스에서 테이블 시각화

워크스페이스에서 결과 테이블을 봅니다.

1. W&B 앱에서 프로젝트로 이동합니다.
2. 프로젝트 워크스페이스에서 해당 run의 이름을 선택합니다. 각 고유 테이블 키에 대해 새 패널이 추가됩니다.

{{< img src="/images/data_vis/wandb_demo_logged_sample_table.png" alt="" >}}

이 예제에서 `my_table`은 `"Table Name"` 키 아래에 기록됩니다.

## 3. 모델 버전 간 비교

여러 W&B Runs에서 샘플 테이블을 기록하고 프로젝트 워크스페이스에서 결과를 비교합니다. 이 [예제 워크스페이스](https://wandb.ai/carey/table-test?workspace=user-carey)에서는 동일한 테이블에서 여러 다른 버전의 행을 결합하는 방법을 보여줍니다.

{{< img src="/images/data_vis/wandb_demo_toggle_on_and_off_cross_run_comparisons_in_tables.gif" alt="" >}}

테이블 필터, 정렬 및 그룹화 기능을 사용하여 모델 결과를 탐색하고 평가합니다.

{{< img src="/images/data_vis/wandb_demo_filter_on_a_table.png" alt="" >}}
