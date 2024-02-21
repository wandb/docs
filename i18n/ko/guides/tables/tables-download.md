---
description: How to export data from tables.
displayed_sidebar: default
---

# 테이블 데이터 내보내기
모든 W&B 아티팩트와 마찬가지로, 테이블도 쉬운 데이터 내보내기를 위해 판다스 데이터프레임으로 변환될 수 있습니다.

## `table`을 `artifact`으로 변환하기
먼저, 테이블을 아티팩트로 변환해야 합니다. 이를 가장 쉽게 하는 방법은 `artifact.get(table, "table_name")`을 사용하는 것입니다:

```python
# 새로운 테이블 생성 및 로그
with wandb.init() as r:
    artifact = wandb.Artifact("my_dataset", type="dataset")
    table = wandb.Table(
        columns=["a", "b", "c"], data=[(i, i * 2, 2**i) for i in range(10)]
    )
    artifact.add(table, "my_table")
    wandb.log_artifact(artifact)

# 생성한 테이블을 생성한 아티팩트를 사용하여 검색합니다.
with wandb.init() as r:
    artifact = r.use_artifact("my_dataset:latest")
    table = artifact.get("my_table")
```

## `artifact`을 데이터프레임으로 변환하기
그 다음, 테이블을 데이터프레임으로 변환합니다:

```python
# 마지막 코드 예제에서 이어집니다:
df = table.get_dataframe()
```

## 데이터 내보내기
이제 데이터프레임이 지원하는 어떤 메서드든 사용하여 내보낼 수 있습니다:

```python
# 테이블 데이터를 .csv로 변환
df.to_csv("example.csv", encoding="utf-8")
```

# 다음 단계
- `아티팩트`에 관한 [참조 문서](../artifacts/construct-an-artifact.md)를 확인하세요.
- [테이블 워크스루](../tables/tables-walkthrough.md) 가이드를 살펴보세요.
- [데이터프레임](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) 참조 문서를 확인하세요.