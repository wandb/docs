---
description: Importing and logging data into W&B
displayed_sidebar: default
---

# CSV 파일에서 데이터와 실험 로그하기

<head>
  <title>W&B 실험을 통한 CSV 파일 추적</title>
</head>

W&B Python 라이브러리를 사용하여 CSV 파일을 로그하고 [W&B 대시보드](../app.md)에서 시각화하세요. W&B 대시보드는 기계학습 모델의 결과를 구성하고 시각화하는 중심 장소입니다. 이는 W&B에 로그되지 않은 [이전 기계학습 실험의 정보가 포함된 CSV 파일이 있는 경우](#import-and-log-your-csv-of-experiments) 또는 [데이터셋이 포함된 CSV 파일이 있는 경우](#import-and-log-your-dataset-csv-file)에 특히 유용합니다.

## 데이터셋 CSV 파일 가져오기 및 로그하기

CSV 파일의 내용을 재사용하기 쉽게 만들기 위해 W&B 아티팩트를 사용하는 것이 좋습니다.

1. 시작하려면, 먼저 CSV 파일을 가져옵니다. 다음 코드조각에서 `iris.csv` 파일 이름을 귀하의 CSV 파일 이름으로 바꿉니다:

```python
import wandb
import pandas as pd

# 새로운 DataFrame으로 CSV 읽기
new_iris_dataframe = pd.read_csv("iris.csv")
```

2. [W&B 대시보드](../app.md)를 활용하기 위해 CSV 파일을 W&B 테이블로 변환하세요.

```python
# DataFrame을 W&B Table로 변환
iris_table = wandb.Table(dataframe=new_iris_dataframe)
```

3. 다음으로, W&B 아티팩트를 생성하고 테이블을 아티팩트에 추가하세요:

```python
# 행 제한을 200000으로 늘리고 재사용하기 쉽게 만들기 위해
# 테이블을 아티팩트에 추가
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# 데이터를 보존하기 위해 원시 csv 파일을 아티팩트에 로그
iris_table_artifact.add_file("iris.csv")
```
W&B 아티팩트에 대한 자세한 정보는 [아티팩트 챕터](../../artifacts/intro.md)를 참조하세요.

4. 마지막으로, `wandb.init`을 사용하여 새로운 W&B Run을 시작하여 W&B에 추적 및 로그하세요:

```python
# 데이터 로그를 위한 W&B run 시작
run = wandb.init(project="tables-walkthrough")

# run으로 테이블 로그하여 시각화...
run.log({"iris": iris_table})

# 그리고 사용 가능한 행 제한을 늘리기 위해 아티팩트로 로그!
run.log_artifact(iris_table_artifact)
```

`wandb.init()` API는 데이터를 Run에 로그하기 위해 새로운 백그라운드 프로세스를 생성하고 기본적으로 wandb.ai에 데이터를 동기화합니다. W&B 워크스페이스 대시보드에서 실시간 시각화를 확인하세요. 다음 이미지는 코드조각 데모의 출력을 보여줍니다.

![W&B 대시보드로 가져온 CSV 파일](/images/track/import_csv_tutorial.png)

앞서 언급한 코드조각으로 구성된 전체 스크립트는 아래와 같습니다:

```python
import wandb
import pandas as pd

# 새로운 DataFrame으로 CSV 읽기
new_iris_dataframe = pd.read_csv("iris.csv")

# DataFrame을 W&B Table로 변환
iris_table = wandb.Table(dataframe=new_iris_dataframe)

# 행 제한을 200000으로 늘리고 재사용하기 쉽게 만들기 위해
# 테이블을 아티팩트에 추가
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# 데이터를 보존하기 위해 원시 csv 파일을 아티팩트에 로그
iris_table_artifact.add_file("iris.csv")

# 데이터 로그를 위한 W&B run 시작
run = wandb.init(project="tables-walkthrough")

# run으로 테이블 로그하여 시각화...
run.log({"iris": iris_table})

# 그리고 사용 가능한 행 제한을 늘리기 위해 아티팩트로 로그!
run.log_artifact(iris_table_artifact)

# run 종료 (노트북에서 유용함)
run.finish()
```

## 실험 CSV 가져오기 및 로그하기

일부 경우에는 실험 세부사항이 CSV 파일에 있을 수 있습니다. 이러한 CSV 파일에 포함된 일반적인 세부사항은 다음과 같습니다:

* 실험 run의 이름
* 초기 [노트](../../app/features/notes.md)
* 실험을 구별하는 [태그](../../app/features/tags.md)
* 실험에 필요한 설정 (우리의 [Sweeps 하이퍼파라미터 튜닝](../../sweeps/intro.md)을 활용할 수 있는 추가 이점 포함).

| 실험         | 모델 이름             | 노트                                              | 태그            | Num Layers | 최종 트레이닝 정확도 | 최종 검증 정확도 | 트레이닝 손실                         |
| ------------ | ------------------ | ------------------------------------------------ | -------------- | ---------- | --------------- | ------------- | ------------------------------------- |
| 실험 1       | mnist-300-layers   | 트레이닝 데이터에 너무 과적합됨                  | \[latest]       | 300        | 0.99            | 0.90          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| 실험 2       | mnist-250-layers   | 현재 최고 모델                                   | \[prod, best]   | 250        | 0.95            | 0.96          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| 실험 3       | mnist-200-layers   | 베이스라인 모델보다 나빴음. 디버그 필요           | \[debug]        | 200        | 0.76            | 0.70          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| ...          | ...                | ...                                              | ...             | ...        | ...             | ...           |                                       |
| 실험 N       | mnist-X-layers     | 노트                                              | ...             | ...        | ...             | ...           | \[..., ...]                           |

W&B는 실험의 CSV 파일을 가져와서 W&B 실험 Run으로 변환할 수 있습니다. 다음 코드 조각과 코드 스크립트는 실험의 CSV 파일을 가져오고 로그하는 방법을 보여줍니다:

1. 시작하려면, 먼저 CSV 파일을 읽고 Pandas DataFrame으로 변환하세요. `"experiments.csv"`를 귀하의 CSV 파일 이름으로 바꿉니다:

```python
import wandb
import pandas as pd

FILENAME = "experiments.csv"
loaded_experiment_df = pd.read_csv(FILENAME)

PROJECT_NAME = "Converted Experiments"

EXPERIMENT_NAME_COL = "실험"
NOTES_COL = "노트"
TAGS_COL = "태그"
CONFIG_COLS = ["Num Layers"]
SUMMARY_COLS = ["최종 트레이닝 정확도", "최종 검증 정확도"]
METRIC_COLS = ["트레이닝 손실"]

# Pandas DataFrame을 작업하기 쉽게 포맷
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


2. 다음으로, [`wandb.init()`](../../../ref/python/init.md)을 사용하여 새로운 W&B Run을 시작하여 W&B에 추적 및 로그하세요:

```python
run = wandb.init(
    project=PROJECT_NAME, name=run_name, tags=tags, notes=notes, config=config
)
```

실험을 실행하는 동안, W&B를 사용하여 메트릭의 모든 인스턴스를 로그하여 조회, 분석할 수 있도록 하고 싶을 수 있습니다. 이를 위해 [`run.log()`](../../../ref/python/log.md) 명령을 사용하세요:

```python
run.log({key: val})
```

실험 실행의 결과를 정의하는 최종 요약 메트릭을 선택적으로 로그할 수 있습니다. 이 예제 경우에는, `run.summary.update()`로 우리의 run에 요약 메트릭을 추가할 것입니다:

```python
run.summary.update(summaries)
```

요약 메트릭 로그에 대한 자세한 정보는 [요약 메트릭 로그](./log-summary.md)를 참조하세요.

아래는 위 샘플 표를 [W&B 대시보드](../app.md)로 변환하는 전체 예제 스크립트입니다:

```python
FILENAME = "experiments.csv"
loaded_experiment_df = pd.read_csv(FILENAME)

PROJECT_NAME = "Converted Experiments"

EXPERIMENT_NAME_COL = "실험"
NOTES_COL = "노트"
TAGS_COL = "태그"
CONFIG_COLS = ["Num Layers"]
SUMMARY_COLS = ["최종 트레이닝 정확도", "최종 검증 정확도"]
METRIC_COLS = ["트레이닝 손실"]

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