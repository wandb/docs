---
title: Experiments에서 CSV 파일 추적하기
description: W&B에 데이터 가져오기 및 로그 남기기
menu:
  default:
    identifier: ko-guides-models-track-log-working-with-csv
    parent: log-objects-and-media
---

W&B Python 라이브러리를 활용해 CSV 파일을 로깅하고 [W&B Dashboard]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})에서 시각화할 수 있습니다. W&B Dashboard는 기계학습 모델의 결과를 정리하고 시각화하는 중심 장소입니다. W&B에 아직 로깅되지 않은 [기존 기계학습 실험 정보가 담긴 CSV 파일]({{< relref path="#import-and-log-your-csv-of-experiments" lang="ko" >}})이나, [데이터셋이 담긴 CSV 파일]({{< relref path="#import-and-log-your-dataset-csv-file" lang="ko" >}})이 있을 때 특히 유용하게 활용할 수 있습니다.

## 데이터셋 CSV 파일 가져오기 및 로깅




CSV 파일의 재사용성을 높이기 위해 W&B Artifacts를 활용하는 것을 권장합니다.

1. 먼저 CSV 파일을 가져옵니다. 아래 코드조각에서 `iris.csv` 파일명을 여러분의 CSV 파일명으로 바꿔주세요:

```python
import wandb
import pandas as pd

# CSV 파일을 새 DataFrame으로 읽어오기
new_iris_dataframe = pd.read_csv("iris.csv")
```

2. CSV 파일을 W&B Table로 변환하여 [W&B Dashboards]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})에서 시각화할 수 있습니다.

```python
# DataFrame을 W&B Table로 변환
iris_table = wandb.Table(dataframe=new_iris_dataframe)
```

3. W&B Artifact를 생성하고, 해당 Table을 Artifact에 추가합니다:

```python
# Table을 Artifact에 추가하여
# 행 제한을 200,000으로 늘리고, 재사용을 용이하게 함
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# 원본 csv 파일도 artifact 안에 추가하여 데이터를 보존
iris_table_artifact.add_file("iris.csv")
```
W&B Artifacts에 대해 더 자세한 내용은 [Artifacts 챕터]({{< relref path="/guides/core/artifacts/" lang="ko" >}})를 참고하세요.  

4. 마지막으로, `wandb.init`으로 새로운 W&B Run을 시작하여 데이터를 추적 및 로깅합니다:

```python
# 데이터를 기록할 W&B run 시작
run = wandb.init(project="tables-walkthrough")

# Table을 run에 로깅하여 시각화...
run.log({"iris": iris_table})

# 그리고 Artifact로도 로깅하여 행 제한을 늘릴 수 있음!
run.log_artifact(iris_table_artifact)
```

`wandb.init()` API는 Run에 데이터를 기록하는 백그라운드 프로세스를 생성하며, 데이터는 기본적으로 wandb.ai에 동기화됩니다. W&B Workspace Dashboard에서 실시간 시각화를 확인할 수 있습니다. 아래 이미지는 위 코드조각 실행 결과 화면 예시입니다.

{{< img src="/images/track/import_csv_tutorial.png" alt="CSV 파일이 W&B Dashboard에 임포트된 모습" >}}


위의 코드조각을 모두 합친 전체 스크립트는 다음과 같습니다:

```python
import wandb
import pandas as pd

# CSV 파일을 새 DataFrame으로 읽어오기
new_iris_dataframe = pd.read_csv("iris.csv")

# DataFrame을 W&B Table로 변환
iris_table = wandb.Table(dataframe=new_iris_dataframe)

# Table을 Artifact에 추가하여
# 행 제한을 200,000으로 늘리고, 재사용을 용이하게 함
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# 원본 csv 파일도 artifact 안에 추가하여 데이터를 보존
iris_table_artifact.add_file("iris.csv")

# 데이터를 기록할 W&B run 시작
run = wandb.init(project="tables-walkthrough")

# Table을 run에 로깅하여 시각화...
run.log({"iris": iris_table})

# 그리고 Artifact로도 로깅하여 행 제한을 늘릴 수 있음!
run.log_artifact(iris_table_artifact)

# run 종료 (노트북에서 유용)
run.finish()
```

## 실험 CSV 파일 가져오기 및 로깅




경우에 따라 실험 세부정보가 CSV 파일에 있을 수 있습니다. 다음과 같은 정보가 주로 포함됩니다:

* 실험 run의 이름
* [노트]({{< relref path="/guides/models/track/runs/#add-a-note-to-a-run" lang="ko" >}}) 정보
* 실험을 구분할 [태그]({{< relref path="/guides/models/track/runs/tags.md" lang="ko" >}})
* 실험에 필요한 설정값(추가적으로 [Sweeps 하이퍼파라미터 튜닝]({{< relref path="/guides/models/sweeps/" lang="ko" >}}) 활용 가능)

| Experiment   | Model Name       | Notes                                            | Tags          | Num Layers | Final Train Acc | Final Val Acc | Training Losses                       |
| ------------ | ---------------- | ------------------------------------------------ | ------------- | ---------- | --------------- | ------------- | ------------------------------------- |
| Experiment 1 | mnist-300-layers | 트레이닝 데이터에 너무 과적합함                  | \[latest]     | 300        | 0.99            | 0.90          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| Experiment 2 | mnist-250-layers | 현재 최고의 모델                                | \[prod, best] | 250        | 0.95            | 0.96          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| Experiment 3 | mnist-200-layers | 베이스라인 모델보다 성능이 나쁨. 디버깅 필요     | \[debug]      | 200        | 0.76            | 0.70          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| ...          | ...              | ...                                              | ...           | ...        | ...             | ...           |                                       |
| Experiment N | mnist-X-layers   | NOTES                                            | ...           | ...        | ...             | ...           | \[..., ...]                           |

W&B는 실험이 기록된 CSV 파일을 받아서 각각을 W&B Experiment Run으로 변환할 수 있습니다. 아래 코드조각 및 전체 스크립트 예시는 실험 CSV 파일을 가져와서 로깅하는 과정을 보여줍니다:

1. 먼저 CSV 파일을 읽고, Pandas DataFrame으로 변환합니다. `"experiments.csv"`를 여러분의 CSV 파일명으로 바꿔주세요:

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

# 실험 데이터를 다루기 쉽게 DataFrame 포맷 변환
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

2. 다음으로, [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ko" >}})을 사용해 새로운 W&B Run을 시작합니다:

    ```python
    run = wandb.init(
        project=PROJECT_NAME, name=run_name, tags=tags, notes=notes, config=config
    )
    ```

실험이 진행되면서 메트릭 값을 모두 로깅하여 W&B에서 조회, 쿼리, 분석하면 좋습니다. [`run.log()`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog" lang="ko" >}}) 명령어를 사용하세요:

```python
run.log({key: val})
```

run의 결과를 나타내는 요약 메트릭을 로깅하려면 [`define_metric`]({{< relref path="/ref/python/sdk/classes/run#define_metric" lang="ko" >}}) API를 사용할 수 있습니다. 아래 예시는 `run.summary.update()`로 summary 메트릭을 추가합니다:

```python
run.summary.update(summaries)
```

summary metrics에 대한 자세한 내용은 [Log Summary Metrics]({{< relref path="./log-summary.md" lang="ko" >}})를 참고하세요.

아래는 위의 샘플 테이블을 [W&B Dashboard]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})에 변환하는 전체 예시 스크립트입니다:

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