---
title: 테이블 데이터 내보내기
description: 테이블에서 데이터 내보내는 방법.
menu:
  default:
    identifier: ko-guides-models-tables-tables-download
    parent: tables
---

모든 W&B Artifacts 와 마찬가지로, Tables 도 pandas 데이터프레임으로 변환하여 쉽게 데이터를 내보낼 수 있습니다.

## `table`을 `artifact`로 변환하기
먼저, 테이블을 아티팩트로 변환해야 합니다. 가장 쉬운 방법은 `artifact.get(table, "table_name")`을 사용하는 것입니다:

```python
# 새로운 테이블을 만들고 로그합니다.
with wandb.init() as r:
    artifact = wandb.Artifact("my_dataset", type="dataset")
    table = wandb.Table(
        columns=["a", "b", "c"], data=[(i, i * 2, 2**i) for i in range(10)]
    )
    artifact.add(table, "my_table")
    wandb.log_artifact(artifact)

# 생성한 artifact 를 사용해 테이블을 불러옵니다.
with wandb.init() as r:
    artifact = r.use_artifact("my_dataset:latest")
    table = artifact.get("my_table")
```

## `artifact`를 Dataframe으로 변환하기
그 다음, 테이블을 데이터프레임으로 변환합니다:

```python
# 앞선 코드 예시를 계속해서 사용합니다:
df = table.get_dataframe()
```

## 데이터 내보내기
이제 데이터프레임이 제공하는 다양한 방법을 사용하여 데이터를 내보낼 수 있습니다:

```python
# 테이블 데이터를 .csv 파일로 변환하기
df.to_csv("example.csv", encoding="utf-8")
```

# 다음 단계
- [`artifacts`에 관한 참고 문서]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ko" >}})를 확인하세요.
- [Tables Walktrough]({{< relref path="/guides/models/tables/tables-walkthrough.md" lang="ko" >}}) 가이드를 살펴보세요.
- [Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) 공식 문서도 참고해보세요.