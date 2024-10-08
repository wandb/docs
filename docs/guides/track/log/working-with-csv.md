---
title: Track CSV files with experiments
description: W&B에 데이터 가져오기 및 로그 생성하기
displayed_sidebar: default
---

W&B Python 라이브러리를 사용하여 CSV 파일을 로그하고 [W&B 대시보드](../app.md)에서 시각화하세요. W&B 대시보드는 기계학습 모델의 결과를 조직하고 시각화하는 중심 장소입니다. W&B에 로그되지 않은 [이전 기계학습 실험에 대한 정보를 포함한 CSV 파일](#import-and-log-your-csv-of-experiments)이 있거나 [데이터셋을 포함한 CSV 파일](#import-and-log-your-dataset-csv-file)이 있을 경우 특히 유용합니다.

## 데이터셋 CSV 파일 가져오기 및 로그하기

CSV 파일의 내용을 재사용하기 쉽게 만들기 위해 W&B Artifacts를 활용하는 것을 권장합니다.

1. 시작하려면 먼저 CSV 파일을 가져오세요. 다음 코드조각에서는 `iris.csv` 파일명을 본인의 CSV 파일명으로 교체하세요:

```python
import wandb
import pandas as pd

# CSV를 새로운 DataFrame으로 읽어들입니다.
new_iris_dataframe = pd.read_csv("iris.csv")
```

2. CSV 파일을 W&B 테이블로 변환하여 [W&B 대시보드](../app.md)를 활용하세요.

```python
# DataFrame을 W&B 테이블로 변환합니다.
iris_table = wandb.Table(dataframe=new_iris_dataframe)
```

3. 다음으로, W&B Artifact를 생성하고 테이블을 Artifact에 추가하세요:

```python
# 행의 수를 200000으로 늘리고 재사용을 쉽게 하기 위해 Artifact에 테이블을 추가합니다.
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# 데이터를 보존하기 위해 raw csv 파일을 artifact 내에 로그합니다
iris_table_artifact.add_file("iris.csv")
```
W&B Artifacts에 대한 자세한 정보는 [Artifacts chapter](../../artifacts/intro.md)를 참고하세요.

4. 마지막으로, 새로운 W&B Run을 시작하여 `wandb.init`과 함께 W&B에 트랙하고 로그하세요:

```python
# 데이터 로그를 위해 W&B run을 시작합니다.
run = wandb.init(project="tables-walkthrough")

# run과 함께 시각화할 테이블을 로그합니다...
run.log({"iris": iris_table})

# 그리고 사용 가능한 행 제한을 늘리기 위해 Artifact로 로그합니다!
run.log_artifact(iris_table_artifact)
```

`wandb.init()` API는 데이터를 Run에 로그하기 위한 새로운 백그라운드 프로세스를 생성하고 기본적으로 wandb.ai에 데이터를 동기화합니다. W&B 워크스페이스 대시보드에서 실시간 시각화를 확인할 수 있습니다. 다음 이미지는 코드조각 데모의 출력을 보여줍니다.

![CSV file imported into W&B Dashboard](/images/track/import_csv_tutorial.png)

위 코드조각이 포함된 전체 스크립트는 아래에 있습니다:

```python
import wandb
import pandas as pd

# CSV를 새로운 DataFrame으로 읽어들입니다.
new_iris_dataframe = pd.read_csv("iris.csv")

# DataFrame을 W&B 테이블로 변환합니다.
iris_table = wandb.Table(dataframe=new_iris_dataframe)

# 행의 수를 200000으로 늘리고 재사용을 쉽게 하기 위해 Artifact에 테이블을 추가합니다.
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# 데이터를 보존하기 위해 raw csv 파일을 artifact 내에 로그합니다
iris_table_artifact.add_file("iris.csv")

# 데이터 로그를 위해 W&B run을 시작합니다.
run = wandb.init(project="tables-walkthrough")

# run과 함께 시각화할 테이블을 로그합니다...
run.log({"iris": iris_table})

# 그리고 사용 가능한 행 제한을 늘리기 위해 Artifact로 로그합니다!
run.log_artifact(iris_table_artifact)

# run을 완료합니다 (노트북에서 유용)
run.finish()
```

## 실험 CSV 파일 가져오기 및 로그하기

어떤 경우에는 CSV 파일에 실험 세부 정보가 포함되어 있을 수 있습니다. 이러한 CSV 파일에는 일반적으로 다음과 같은 세부 정보가 포함됩니다:

* 실험 run의 이름
* 초기 [노트](../../app/features/notes.md)
* 실험을 구분하는 [태그](../../app/features/tags.md)
* 실험에 필요한 설정 (우리의 [Sweeps 하이퍼파라미터 튜닝](../../sweeps/intro.md)을 활용할 수 있는 추가 이점 포함).

| Experiment   | Model Name       | Notes                                            | Tags          | Num Layers | Final Train Acc | Final Val Acc | Training Losses                       |
| ------------ | ---------------- | ------------------------------------------------ | ------------- | ---------- | --------------- | ------------- | ------------------------------------- |
| Experiment 1 | mnist-300-layers | Overfit way too much on training data            | \[latest]     | 300        | 0.99            | 0.90          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| Experiment 2 | mnist-250-layers | Current best model                               | \[prod, best] | 250        | 0.95            | 0.96          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| Experiment 3 | mnist-200-layers | Did worse than the baseline model. Need to debug | \[debug]      | 200        | 0.76            | 0.70          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| ...          | ...              | ...                                              | ...           | ...        | ...             | ...           |                                       |
| Experiment N | mnist-X-layers   | NOTES                                            | ...           | ...        | ...             | ...           | \[..., ...]                           |

W&B는 실험의 CSV 파일을 받아 W&B 실험 Run으로 변환할 수 있습니다. 다음 코드조각과 스크립트는 실험의 CSV 파일을 가져오고 로그하는 방법을 보여줍니다:

1. 시작하려면, CSV 파일을 읽고 Pandas DataFrame으로 변환하세요. `"experiments.csv"`를 CSV 파일명으로 교체하세요:

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

# 작업하기 쉽게 Pandas DataFrame을 포맷합니다.
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

2. 다음으로, 새로운 W&B Run을 시작하여 [`wandb.init()`](../../../ref/python/init.md)과 함께 W&B에 트랙하고 로그하세요:

```python
run = wandb.init(
    project=PROJECT_NAME, name=run_name, tags=tags, notes=notes, config=config
)
```

실험이 진행되는 동안, 모든 메트릭 인스턴스를 로그하여 W&B와 함께 조회하고 분석할 수 있도록 할 수 있습니다. [`run.log()`](../../../ref/python/log.md) 커맨드를 사용하여 이를 실현하세요:

```python
run.log({key: val})
```

Run의 결과를 정의하기 위해 최종 요약 메트릭을 로그할 수도 있습니다. 이를 위해 W&B [`define_metric`](../../../ref/python/run.md#define_metric) API를 사용하세요. 이 예제에서는 최종 요약 메트릭을 `run.summary.update()`와 함께 Run에 추가할 것입니다:

```python
run.summary.update(summaries)
```

요약 메트릭에 대한 자세한 정보는 [Log Summary Metrics](./log-summary.md)를 참고하세요.

아래는 위의 샘플 테이블을 [W&B 대시보드](../app.md)로 변환하는 전체 예제 스크립트입니다:

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