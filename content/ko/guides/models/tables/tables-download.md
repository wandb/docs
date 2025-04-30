---
title: Export table data
description: 테이블에서 데이터를 내보내는 방법.
menu:
  default:
    identifier: ko-guides-models-tables-tables-download
    parent: tables
---

W&B Artifacts와 마찬가지로, Tables는 쉬운 데이터 내보내기를 위해 pandas 데이터프레임으로 변환할 수 있습니다.

## `table`을 `artifact`로 변환하기
먼저, 테이블을 아티팩트로 변환해야 합니다. `artifact.get(table, "table_name")`을 사용하여 가장 쉽게 수행할 수 있습니다.

```python
# 새로운 테이블을 생성하고 로그합니다.
with wandb.init() as r:
    artifact = wandb.Artifact("my_dataset", type="dataset")
    table = wandb.Table(
        columns=["a", "b", "c"], data=[(i, i * 2, 2**i) for i in range(10)]
    )
    artifact.add(table, "my_table")
    wandb.log_artifact(artifact)

# 생성된 아티팩트를 사용하여 생성된 테이블을 검색합니다.
with wandb.init() as r:
    artifact = r.use_artifact("my_dataset:latest")
    table = artifact.get("my_table")
```

## `artifact`를 Dataframe으로 변환하기
다음으로, 테이블을 데이터프레임으로 변환합니다.

```python
# 이전 코드 예제에서 계속됩니다.
df = table.get_dataframe()
```

## 데이터 내보내기
이제 데이터프레임이 지원하는 모든 방법을 사용하여 내보낼 수 있습니다.

```python
# 테이블 데이터를 .csv로 변환
df.to_csv("example.csv", encoding="utf-8")
```

# 다음 단계
- `artifacts`에 대한 [참조 문서]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ko" >}})를 확인하세요.
- [Tables Walktrough]({{< relref path="/guides/models/tables/tables-walkthrough.md" lang="ko" >}}) 가이드를 살펴보세요.
- [Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) 참조 문서를 확인하세요.
