---
description: Explore how to use W&B Tables with this 5 minute Quickstart.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 가이드

다음 퀵스타트는 데이터 테이블 로깅, 데이터 시각화 및 데이터 쿼리 방법을 보여줍니다.

아래 버튼을 선택하여 MNIST 데이터에 대한 PyTorch 퀵스타트 예제 프로젝트를 시도해 보십시오.

## 1. 테이블 로깅
W&B로 테이블을 로그합니다. 새 테이블을 구성하거나 Pandas DataFrame을 전달할 수 있습니다.

<Tabs
  defaultValue="construct"
  values={[
    {label: '테이블 구성', value: 'construct'},
    {label: 'Pandas DataFrame', value: 'pandas'},
  ]}>
  <TabItem value="construct">

새로운 테이블을 구성하고 로그하려면 다음을 사용합니다:
- [`wandb.init()`](../../ref/python/init.md): 결과를 추적하기 위한 [실행](../runs/intro.md)을 생성합니다.
- [`wandb.Table()`](../../ref/python/data-types/table.md): 새로운 테이블 개체를 생성합니다.
  - `columns`: 열 이름을 설정합니다.
  - `data`: 각 행의 내용을 설정합니다.
- [`run.log()`](../../ref/python/log.md): 테이블을 로그하여 W&B에 저장합니다.

예시는 다음과 같습니다:
```python
import wandb

run = wandb.init(project="table-test")
# 새 테이블 생성 및 로그.
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```
  </TabItem>
  <TabItem value="pandas">

Pandas DataFrame을 `wandb.Table()`에 전달하여 새 테이블을 생성합니다.

```python
import wandb
import pandas as pd

df = pd.read_csv("my_data.csv")

run = wandb.init(project="df-table")
my_table = wandb.Table(dataframe=df)
wandb.log({"Table Name": my_table})
```

지원되는 데이터 유형에 대한 자세한 정보는 W&B API 참조 가이드의 [`wandb.Table`](../../ref/python/data-types/table.md)을 참조하십시오.

  </TabItem>
</Tabs>

## 2. 프로젝트 워크스페이스에서 테이블 시각화

워크스페이스에서 결과 테이블을 봅니다.

1. W&B 앱에서 프로젝트로 이동합니다.
2. 프로젝트 워크스페이스에서 실행 이름을 선택합니다. 각 고유한 테이블 키에 대해 새 패널이 추가됩니다.

![](/images/data_vis/wandb_demo_logged_sample_table.png)

이 예에서, `my_table`은 `"Table Name"` 키 아래에 로그됩니다.

## 3. 모델 버전 비교

여러 W&B 실행에서 샘플 테이블을 로그하고 프로젝트 워크스페이스에서 결과를 비교합니다. 이 [예제 워크스페이스](https://wandb.ai/carey/table-test?workspace=user-carey)에서는 동일한 테이블에서 여러 다른 버전의 행을 결합하는 방법을 보여줍니다.

![](/images/data_vis/wandb_demo_toggle_on_and_off_cross_run_comparisons_in_tables.gif)

테이블 필터, 정렬 및 그룹화 기능을 사용하여 모델 결과를 탐색하고 평가합니다.

![](/images/data_vis/wandb_demo_filter_on_a_table.png)