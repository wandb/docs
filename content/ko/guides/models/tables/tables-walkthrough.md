---
title: '튜토리얼: Table 로그, 데이터 시각화 및 쿼리'
description: Learn how to use W&B Tables with this 5-minute quickstart.
menu:
  default:
    identifier: ko-guides-models-tables-tables-walkthrough
    parent: tables
weight: 1
---

다음 퀵스타트에서는 데이터 테이블을 어떻게 로그하고, 데이터를 시각화하며, 데이터를 쿼리하는 방법을 보여줍니다.

아래 버튼을 클릭하여 MNIST 데이터에 대한 PyTorch 퀵스타트 예제 프로젝트를 바로 체험해 볼 수 있습니다.

## 1. 테이블 로그하기
W&B로 테이블을 로그할 수 있습니다. 새 테이블을 직접 만들거나, Pandas DataFrame을 전달할 수 있습니다.

{{< tabpane text=true >}}
{{% tab header="테이블 직접 만들기" value="construct" %}}
새로운 Table을 만들고 로그하려면 다음을 사용합니다:
- [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}}): 결과를 추적할 [run]({{< relref path="/guides/models/track/runs/" lang="ko" >}}) 생성
- [`wandb.Table()`]({{< relref path="/ref/python/sdk/data-types/table.md" lang="ko" >}}): 새로운 테이블 오브젝트 생성
  - `columns`: 컬럼 이름 지정
  - `data`: 각 행의 데이터 지정
- [`wandb.Run.log()`]({{< relref path="/ref/python/sdk/classes/run.md/#method-runlog" lang="ko" >}}): 테이블을 로그하여 W&B에 저장

예시입니다:
```python
import wandb

with wandb.init(project="table-test") as run:
    # 새로운 테이블을 생성해서 로그합니다.
    my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
    run.log({"Table Name": my_table})
```
{{% /tab %}}

{{% tab header="Pandas Dataframe" value="pandas"%}}
Pandas DataFrame을 `wandb.Table()`에 전달하면 새로운 테이블이 만들어집니다.

```python
import wandb
import pandas as pd

df = pd.read_csv("my_data.csv")

with wandb.init(project="df-table") as run:
    # DataFrame에서 새로운 테이블을 생성하고
    # W&B에 로그합니다.
  my_table = wandb.Table(dataframe=df)
  run.log({"Table Name": my_table})
```

지원되는 데이터 타입에 대한 자세한 내용은 W&B API Reference Guide의 [`wandb.Table`]({{< relref path="/ref/python/sdk/data-types/table.md" lang="ko" >}})을 참고하세요.
{{% /tab %}}
{{< /tabpane >}}


## 2. 프로젝트 워크스페이스에서 테이블 시각화하기

로그된 테이블을 워크스페이스에서 바로 확인할 수 있습니다.

1. W&B 앱에서 본인의 프로젝트로 이동하세요.
2. 프로젝트 워크스페이스에서 run의 이름을 클릭하세요. 각 테이블 키마다 새로운 패널이 추가됩니다.

{{< img src="/images/data_vis/wandb_demo_logged_sample_table.png" alt="Sample table logged" >}}

이 예시에서는 `my_table`이 `"Table Name"` 키로 로그되어 있습니다.

## 3. 모델 버전별 비교하기

여러 W&B Run에서 샘플 테이블을 로그하고, 프로젝트 워크스페이스에서 결과를 비교할 수 있습니다. 이 [예시 워크스페이스](https://wandb.ai/carey/table-test?workspace=user-carey)에서는 서로 다른 여러 버전의 행을 하나의 테이블에서 결합하는 방법을 보여줍니다.

{{< img src="/images/data_vis/wandb_demo_toggle_on_and_off_cross_run_comparisons_in_tables.gif" alt="Cross-run table comparison" >}}

테이블의 필터, 정렬, 그룹 기능을 활용해 모델 결과를 탐색하고 평가해 보세요.

{{< img src="/images/data_vis/wandb_demo_filter_on_a_table.png" alt="Table filtering" >}}