---
title: Tutorial: Log tables, visualize and query data
description: W&B Tables를 사용하는 방법을 5분 퀵스타트로 알아보세요.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

다음 퀵스타트에서는 데이터 테이블을 로그하고, 데이터를 시각화하며, 데이터를 쿼리하는 방법을 시연합니다.

아래 버튼을 선택하여 MNIST 데이터에 대한 PyTorch 퀵스타트 예제 프로젝트를 시도해 보세요.

## 1. 테이블 로그하기
W&B로 테이블을 로그합니다. 새 테이블을 생성하거나 Pandas DataFrame을 전달할 수 있습니다.

<Tabs
  defaultValue="construct"
  values={[
    {label: '테이블 생성하기', value: 'construct'},
    {label: 'Pandas DataFrame', value: 'pandas'},
  ]}>
  <TabItem value="construct">

새로운 테이블을 생성하고 로그하려면 다음을 사용합니다:
- [`wandb.init()`](../../ref/python/init.md): 결과를 추적할 [run](../runs/intro.md)을 생성합니다.
- [`wandb.Table()`](../../ref/python/data-types/table.md): 새로운 테이블 오브젝트를 생성합니다.
  - `columns`: 열 이름을 설정합니다.
  - `data`: 각 행의 내용을 설정합니다.
- [`run.log()`](../../ref/python/log.md): W&B에 테이블을 저장합니다.

다음은 예제입니다:
```python
import wandb

run = wandb.init(project="table-test")
# 새로운 테이블을 만들고 로그합니다.
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```
  </TabItem>
  <TabItem value="pandas">

`wandb.Table()`에 Pandas DataFrame을 전달하여 새로운 테이블을 만듭니다.

```python
import wandb
import pandas as pd

df = pd.read_csv("my_data.csv")

run = wandb.init(project="df-table")
my_table = wandb.Table(dataframe=df)
wandb.log({"Table Name": my_table})
```

지원되는 데이터 유형에 대한 자세한 내용은 W&B API Reference Guide의 [`wandb.Table`](../../ref/python/data-types/table.md)를 참조하세요.

  </TabItem>
</Tabs>

## 2. 프로젝트 워크스페이스에서 테이블 시각화

워크스페이스에서 결과 테이블을 확인합니다.

1. W&B 앱에서 프로젝트로 이동합니다.
2. 프로젝트 워크스페이스에서 본인의 run 이름을 선택합니다. 각 고유한 테이블 키에 대해 새로운 패널이 추가됩니다.

![](/images/data_vis/wandb_demo_logged_sample_table.png)

이 예제에서 `my_table`은 키 `"Table Name"` 아래에 로그됩니다.

## 3. 모델 버전을 비교하기

다수의 W&B Runs에서 샘플 테이블을 로그하고, 프로젝트 워크스페이스에서 결과를 비교합니다. 이 [예제 워크스페이스](https://wandb.ai/carey/table-test?workspace=user-carey)에서, 동일한 테이블에 여러 다른 버전의 행을 결합하는 방법을 보여줍니다.

![](/images/data_vis/wandb_demo_toggle_on_and_off_cross_run_comparisons_in_tables.gif)

테이블 필터, 정렬, 그룹화를 사용하여 모델 결과를 탐색하고 평가하세요.

![](/images/data_vis/wandb_demo_filter_on_a_table.png)