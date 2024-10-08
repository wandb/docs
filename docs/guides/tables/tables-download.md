---
title: Export table data
description: 테이블에서 데이터를 내보내는 방법.
displayed_sidebar: default
---

W&B Artifacts와 마찬가지로, 테이블도 데이터 내보내기를 쉽게 하기 위해 pandas 데이터프레임으로 변환할 수 있습니다.

## `table`을 `artifact`로 변환하기
먼저, 테이블을 아티팩트로 변환해야 합니다. 가장 쉬운 방법은 `artifact.get(table, "table_name")`을 사용하는 것입니다:

```python
# 새로운 테이블 생성 및 로그.
with wandb.init() as r:
    artifact = wandb.Artifact("my_dataset", type="dataset")
    table = wandb.Table(
        columns=["a", "b", "c"], data=[(i, i * 2, 2**i) for i in range(10)]
    )
    artifact.add(table, "my_table")
    wandb.log_artifact(artifact)

# 생성한 테이블을 사용한 아티팩트로 가져오기.
with wandb.init() as r:
    artifact = r.use_artifact("my_dataset:latest")
    table = artifact.get("my_table")
```

## `artifact`를 데이터프레임으로 변환하기
그런 다음, 테이블을 데이터프레임으로 변환합니다:

```python
# 마지막 코드 예제에 이어서:
df = table.get_dataframe()
```

## 데이터 내보내기
이제 데이터프레임이 지원하는 모든 메소드를 사용하여 데이터 내보내기를 할 수 있습니다:

```python
# 테이블 데이터를 .csv로 변환하기
df.to_csv("example.csv", encoding="utf-8")
```

# 다음 단계
- `artifacts`에 대한 [참고 문서](../artifacts/construct-an-artifact.md)를 확인해보세요.
- [Tables Walkthrough](../tables/tables-walkthrough.md) 가이드를 살펴보세요.
- [Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) 참고 문서를 확인해보세요.