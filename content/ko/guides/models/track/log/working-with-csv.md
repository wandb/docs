---
title: Track CSV files with experiments
description: W\&B로 데이터를 가져오고 로깅하기
menu:
  default:
    identifier: ko-guides-models-track-log-working-with-csv
    parent: log-objects-and-media
---

W&B Python 라이브러리를 사용하여 CSV 파일을 기록하고 [W&B 대시보드]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})에서 시각화합니다. W&B 대시보드는 머신러닝 모델의 결과를 정리하고 시각화하는 데 핵심적인 장소입니다. 이는 W&B에 기록되지 않은 [이전 머신러닝 실험 정보를 담은 CSV 파일]({{< relref path="#import-and-log-your-csv-of-experiments" lang="ko" >}})이 있거나, [데이터셋을 담은 CSV 파일]({{< relref path="#import-and-log-your-dataset-csv-file" lang="ko" >}})이 있는 경우 특히 유용합니다.

## 데이터셋 CSV 파일 가져오기 및 기록

CSV 파일 내용을 더 쉽게 재사용할 수 있도록 W&B Artifacts를 활용하는 것이 좋습니다.

1. 시작하려면 먼저 CSV 파일을 가져옵니다. 다음 코드 조각에서 `iris.csv` 파일 이름을 CSV 파일 이름으로 바꿉니다.

```python
import wandb
import pandas as pd

# CSV 파일을 새로운 DataFrame으로 읽기
new_iris_dataframe = pd.read_csv("iris.csv")
```

2. CSV 파일을 [W&B 대시보드]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})에서 활용할 수 있도록 W&B Table로 변환합니다.

```python
# DataFrame을 W&B Table로 변환
iris_table = wandb.Table(dataframe=new_iris_dataframe)
```

3. 다음으로, W&B Artifact를 생성하고 테이블을 Artifact에 추가합니다.

```python
# 행
# 제한을 200000으로 늘리고 재사용하기 쉽도록 테이블을 Artifact에 추가합니다.
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# 데이터를 보존하기 위해 Artifact 내에 원시 csv 파일을 기록합니다.
iris_table_artifact.add_file("iris.csv")
```
W&B Artifacts에 대한 자세한 내용은 [Artifacts 챕터]({{< relref path="/guides/core/artifacts/" lang="ko" >}})를 참조하십시오.

4. 마지막으로, `wandb.init`을 사용하여 W&B Run을 새로 시작하여 W&B에 추적하고 기록합니다.

```python
# 데이터를 기록하기 위해 W&B run을 시작합니다.
run = wandb.init(project="tables-walkthrough")

# run으로 시각화하기 위해 테이블을 기록합니다...
run.log({"iris": iris_table})

# 사용 가능한 행 제한을 늘리기 위해 Artifact로 기록합니다!
run.log_artifact(iris_table_artifact)
```

`wandb.init()` API는 Run에 데이터를 기록하기 위해 새로운 백그라운드 프로세스를 생성하고, (기본적으로) wandb.ai에 데이터를 동기화합니다. W&B Workspace 대시보드에서 실시간 시각화를 확인하세요. 다음 이미지는 코드 조각 데모의 출력을 보여줍니다.

{{< img src="/images/track/import_csv_tutorial.png" alt="CSV 파일을 W&B 대시보드로 가져옴" >}}

위 코드 조각이 포함된 전체 스크립트는 아래에서 확인할 수 있습니다.

```python
import wandb
import pandas as pd

# CSV 파일을 새로운 DataFrame으로 읽기
new_iris_dataframe = pd.read_csv("iris.csv")

# DataFrame을 W&B Table로 변환
iris_table = wandb.Table(dataframe=new_iris_dataframe)

# 행
# 제한을 200000으로 늘리고 재사용하기 쉽도록 테이블을 Artifact에 추가합니다.
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# 데이터를 보존하기 위해 Artifact 내에 원시 csv 파일을 기록합니다.
iris_table_artifact.add_file("iris.csv")

# 데이터를 기록하기 위해 W&B run을 시작합니다.
run = wandb.init(project="tables-walkthrough")

# run으로 시각화하기 위해 테이블을 기록합니다...
run.log({"iris": iris_table})

# 사용 가능한 행 제한을 늘리기 위해 Artifact로 기록합니다!
run.log_artifact(iris_table_artifact)

# run을 종료합니다 (노트북에서 유용합니다)
run.finish()
```

## 실험 CSV 파일 가져오기 및 기록

경우에 따라 실험 세부 정보가 CSV 파일에 있을 수 있습니다. 이러한 CSV 파일에서 흔히 볼 수 있는 세부 정보는 다음과 같습니다.

* 실험 Run 이름
* 초기 [노트]({{< relref path="/guides/models/track/runs/#add-a-note-to-a-run" lang="ko" >}})
* 실험을 구별하기 위한 [태그]({{< relref path="/guides/models/track/runs/tags.md" lang="ko" >}})
* 실험에 필요한 설정 (W&B [Sweeps 하이퍼파라미터 튜닝]({{< relref path="/guides/models/sweeps/" lang="ko" >}})을 활용할 수 있는 추가적인 이점 포함).

| 실험         | 모델 이름       | 노트                                           | 태그          | 레이어 수 | 최종 트레이닝 정확도 | 최종 검증 정확도 | 트레이닝 손실                               |
| ------------ | ---------------- | ----------------------------------------------- | ------------- | ---------- | --------------- | ------------- | ------------------------------------ |
| 실험 1       | mnist-300-layers | 트레이닝 데이터에 대해 과적합이 너무 심함       | \[latest]     | 300        | 0.99            | 0.90          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| 실험 2       | mnist-250-layers | 현재 최고의 모델                              | \[prod, best] | 250        | 0.95            | 0.96          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| 실험 3       | mnist-200-layers | 베이스라인 모델보다 성능이 나쁨. 디버깅 필요 | \[debug]      | 200        | 0.76            | 0.70          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| ...          | ...              | ...                                             | ...           | ...        | ...             | ...           |                                      |
| 실험 N       | mnist-X-layers   | 노트                                           | ...           | ...        | ...             | ...           | \[..., ...]                          |

W&B는 실험 CSV 파일을 가져와서 W&B 실험 Run으로 변환할 수 있습니다. 다음 코드 조각 및 코드 스크립트는 실험 CSV 파일을 가져오고 기록하는 방법을 보여줍니다.

1. 시작하려면 먼저 CSV 파일을 읽고 Pandas DataFrame으로 변환합니다. `"experiments.csv"`를 CSV 파일 이름으로 바꿉니다.

```python
import wandb
import pandas as pd

FILENAME = "experiments.csv"
loaded_experiment_df = pd.read_csv(FILENAME)

PROJECT_NAME = "Converted Experiments"

EXPERIMENT_NAME_COL = "Experiment"
NOTES_COL = "Notes"
TAGS_COL = "Tags"
CONFIG_COLS = ["Num Layers"]
SUMMARY_COLS = ["Final Train Acc", "Final Val Acc"]
METRIC_COLS = ["Training Losses"]

# Pandas DataFrame을 형식을 지정하여 작업하기 쉽게 만듭니다.
for i, row in loaded_experiment_df.iterrows():
    run_name = row[EXPERIMENT_NAME_COL]
    notes = row[NOTES_COL]
    tags = row[TAGS_COL]

    config = {}
    for config_col in CONFIG_COLS:
        config[config_col] = row[config_col]

    metrics = {}
    for metric_col in METRIC_COLS:
        metrics[metric_col] = row[metric_col]

    summaries = {}
    for summary_col in SUMMARY_COLS:
        summaries[summary_col] = row[summary_col]
```

2. 다음으로, [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ko" >}})으로 W&B Run을 새로 시작하여 W&B에 추적하고 기록합니다.

```python
run = wandb.init(
    project=PROJECT_NAME, name=run_name, tags=tags, notes=notes, config=config
)
```

실험이 실행되는 동안 메트릭의 모든 인스턴스를 기록하여 W&B에서 보고, 쿼리하고, 분석할 수 있도록 할 수 있습니다. 이를 달성하려면 [`run.log()`]({{< relref path="/ref/python/log.md" lang="ko" >}}) 코맨드를 사용하십시오.

```python
run.log({key: val})
```

선택적으로 최종 요약 메트릭을 기록하여 Run 결과를 정의할 수 있습니다. 이를 달성하려면 W&B [`define_metric`]({{< relref path="/ref/python/run.md#define_metric" lang="ko" >}}) API를 사용하십시오. 이 예제에서는 `run.summary.update()`를 사용하여 요약 메트릭을 Run에 추가합니다.

```python
run.summary.update(summaries)
```

요약 메트릭에 대한 자세한 내용은 [요약 메트릭 기록]({{< relref path="./log-summary.md" lang="ko" >}})을 참조하십시오.

아래는 위의 샘플 테이블을 [W&B 대시보드]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})로 변환하는 전체 예제 스크립트입니다.

```python
FILENAME = "experiments.csv"
loaded_experiment_df = pd.read_csv(FILENAME)

PROJECT_NAME = "Converted Experiments"

EXPERIMENT_NAME_COL = "Experiment"
NOTES_COL = "Notes"
TAGS_COL = "Tags"
CONFIG_COLS = ["Num Layers"]
SUMMARY_COLS = ["Final Train Acc", "Final Val Acc"]
METRIC_COLS = ["Training Losses"]

for i, row in loaded_experiment_df.iterrows():
    run_name = row[EXPERIMENT_NAME_COL]
    notes = row[NOTES_COL]
    tags = row[TAGS_COL]

    config = {}
    for config_col in CONFIG_COLS:
        config[config_col] = row[config_col]

    metrics = {}
    for metric_col in METRIC_COLS:
        metrics[metric_col] = row[metric_col]

    summaries = {}
    for summary_col in SUMMARY_COLS:
        summaries[summary_col] = row[summary_col]

    run = wandb.init(
        project=PROJECT_NAME, name=run_name, tags=tags, notes=notes, config=config
    )

    for key, val in metrics.items():
        if isinstance(val, list):
            for _val in val:
                run.log({key: _val})
        else:
            run.log({key: val})

    run.summary.update(summaries)
    run.finish()
```