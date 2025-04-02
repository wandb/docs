---
title: Log tables
description: W&B로 테이블을 로그합니다.
menu:
  default:
    identifier: ko-guides-models-track-log-log-tables
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}
`wandb.Table`을 사용하여 데이터를 기록하고 Weights & Biases로 시각화하고 쿼리합니다. 이 가이드에서는 다음 방법을 배울 수 있습니다.

1. [테이블 만들기]({{< relref path="./log-tables.md#create-tables" lang="ko" >}})
2. [데이터 추가]({{< relref path="./log-tables.md#add-data" lang="ko" >}})
3. [데이터 검색]({{< relref path="./log-tables.md#retrieve-data" lang="ko" >}})
4. [테이블 저장]({{< relref path="./log-tables.md#save-tables" lang="ko" >}})

## 테이블 만들기

테이블을 정의하려면 데이터의 각 행에 대해 보려는 열을 지정합니다. 각 행은 트레이닝 데이터셋의 단일 항목, 트레이닝 중의 특정 단계 또는 에포크, 테스트 항목에 대한 모델의 예측값, 모델에서 생성된 오브젝트 등이 될 수 있습니다. 각 열에는 숫자, 텍스트, 부울, 이미지, 비디오, 오디오 등 고정된 유형이 있습니다. 유형을 미리 지정할 필요는 없습니다. 각 열에 이름을 지정하고 해당 유형의 데이터만 해당 열 인덱스로 전달해야 합니다. 더 자세한 예는 [이 리포트](https://wandb.ai/stacey/mnist-viz/reports/Guide-to-W-B-Tables--Vmlldzo2NTAzOTk#1.-how-to-log-a-wandb.table)를 참조하십시오.

다음 두 가지 방법 중 하나로 `wandb.Table` 생성자를 사용합니다.

1. **행 목록:** 이름이 지정된 열과 데이터 행을 기록합니다. 예를 들어 다음 코드 조각은 두 개의 행과 세 개의 열이 있는 테이블을 생성합니다.

```python
wandb.Table(columns=["a", "b", "c"], data=[["1a", "1b", "1c"], ["2a", "2b", "2c"]])
```


2. **Pandas DataFrame:** `wandb.Table(dataframe=my_df)`를 사용하여 DataFrame을 기록합니다. 열 이름은 DataFrame에서 추출됩니다.

#### 기존 배열 또는 데이터 프레임에서

```python
# 모델이 다음 필드를 사용할 수 있는 네 개의 이미지에 대한 예측을 반환했다고 가정합니다.
# - 이미지 ID
# - wandb.Image()로 래핑된 이미지 픽셀
# - 모델의 예측 레이블
# - 그라운드 트루스 레이블
my_data = [
    [0, wandb.Image("img_0.jpg"), 0, 0],
    [1, wandb.Image("img_1.jpg"), 8, 0],
    [2, wandb.Image("img_2.jpg"), 7, 1],
    [3, wandb.Image("img_3.jpg"), 1, 1],
]

# 해당 열이 있는 wandb.Table() 생성
columns = ["id", "image", "prediction", "truth"]
test_table = wandb.Table(data=my_data, columns=columns)
```

## 데이터 추가

테이블은 변경 가능합니다. 스크립트가 실행될 때 테이블에 최대 200,000개의 행까지 더 많은 데이터를 추가할 수 있습니다. 테이블에 데이터를 추가하는 방법에는 두 가지가 있습니다.

1. **행 추가**: `table.add_data("3a", "3b", "3c")`. 새 행은 목록으로 표시되지 않습니다. 행이 목록 형식인 경우 별표 표기법 `*`을 사용하여 목록을 위치 인수로 확장합니다. `table.add_data(*my_row_list)`. 행에는 테이블의 열 수와 동일한 수의 항목이 포함되어야 합니다.
2. **열 추가**: `table.add_column(name="col_name", data=col_data)`. `col_data`의 길이는 테이블의 현재 행 수와 같아야 합니다. 여기서 `col_data`는 목록 데이터 또는 NumPy NDArray일 수 있습니다.

### 점진적으로 데이터 추가

이 코드 샘플은 W&B 테이블을 점진적으로 생성하고 채우는 방법을 보여줍니다. 가능한 모든 레이블에 대한 신뢰도 점수를 포함하여 미리 정의된 열로 테이블을 정의하고 추론 중에 행별로 데이터를 추가합니다. [run을 재개할 때 테이블에 점진적으로 데이터를 추가]( {{< relref path="#adding-data-to-resumed-runs" lang="ko" >}})할 수도 있습니다.

```python
# 각 레이블에 대한 신뢰도 점수를 포함하여 테이블의 열을 정의합니다.
columns = ["id", "image", "guess", "truth"]
for digit in range(10):  # 각 숫자(0-9)에 대한 신뢰도 점수 열을 추가합니다.
    columns.append(f"score_{digit}")

# 정의된 열로 테이블을 초기화합니다.
test_table = wandb.Table(columns=columns)

# 테스트 데이터셋을 반복하고 데이터를 행별로 테이블에 추가합니다.
# 각 행에는 이미지 ID, 이미지, 예측 레이블, 트루 레이블 및 신뢰도 점수가 포함됩니다.
for img_id, img in enumerate(mnist_test_data):
    true_label = mnist_test_data_labels[img_id]  # 그라운드 트루스 레이블
    guess_label = my_model.predict(img)  # 예측 레이블
    test_table.add_data(
        img_id, wandb.Image(img), guess_label, true_label
    )  # 테이블에 행 데이터를 추가합니다.
```

#### 재개된 run에 데이터 추가

아티팩트에서 기존 테이블을 로드하고, 데이터의 마지막 행을 검색하고, 업데이트된 메트릭을 추가하여 재개된 run에서 W&B 테이블을 점진적으로 업데이트할 수 있습니다. 그런 다음 호환성을 위해 테이블을 다시 초기화하고 업데이트된 버전을 W&B에 다시 기록합니다.

```python
# 아티팩트에서 기존 테이블을 로드합니다.
best_checkpt_table = wandb.use_artifact(table_tag).get(table_name)

# 재개를 위해 테이블에서 데이터의 마지막 행을 가져옵니다.
best_iter, best_metric_max, best_metric_min = best_checkpt_table.data[-1]

# 필요에 따라 최상의 메트릭을 업데이트합니다.

# 업데이트된 데이터를 테이블에 추가합니다.
best_checkpt_table.add_data(best_iter, best_metric_max, best_metric_min)

# 호환성을 보장하기 위해 업데이트된 데이터로 테이블을 다시 초기화합니다.
best_checkpt_table = wandb.Table(
    columns=["col1", "col2", "col3"], data=best_checkpt_table.data
)

# 업데이트된 테이블을 Weights & Biases에 기록합니다.
wandb.log({table_name: best_checkpt_table})
```

## 데이터 검색

데이터가 테이블에 있으면 열 또는 행별로 엑세스합니다.

1. **행 반복기**: 사용자는 `for ndx, row in table.iterrows(): ...`와 같은 테이블의 행 반복기를 사용하여 데이터의 행을 효율적으로 반복할 수 있습니다.
2. **열 가져오기**: 사용자는 `table.get_column("col_name")`을 사용하여 데이터 열을 검색할 수 있습니다. 편의를 위해 사용자는 `convert_to="numpy"`를 전달하여 열을 기본 요소의 NumPy NDArray로 변환할 수 있습니다. 이는 열에 기본 데이터에 직접 엑세스할 수 있도록 `wandb.Image`와 같은 미디어 유형이 포함된 경우에 유용합니다.

## 테이블 저장

예를 들어 모델 예측 테이블과 같이 스크립트에서 데이터 테이블을 생성한 후 결과를 라이브로 시각화하기 위해 W&B에 저장합니다.

### 테이블을 run에 기록

`wandb.log()`를 사용하여 다음과 같이 테이블을 run에 저장합니다.

```python
run = wandb.init()
my_table = wandb.Table(columns=["a", "b"], data=[["1a", "1b"], ["2a", "2b"]])
run.log({"table_key": my_table})
```

테이블이 동일한 키에 기록될 때마다 테이블의 새 버전이 생성되어 백엔드에 저장됩니다. 즉, 모델 예측이 시간이 지남에 따라 어떻게 향상되는지 확인하기 위해 여러 트레이닝 단계에서 동일한 테이블을 기록하거나 동일한 키에 기록되는 한 다른 run에서 테이블을 비교할 수 있습니다. 최대 200,000개의 행을 기록할 수 있습니다.

{{% alert %}}
200,000개 이상의 행을 기록하려면 다음을 사용하여 제한을 재정의할 수 있습니다.

`wandb.Table.MAX_ARTIFACT_ROWS = X`

그러나 이렇게 하면 UI에서 쿼리 속도 저하와 같은 성능 문제가 발생할 수 있습니다.
{{% /alert %}}

### 프로그래밍 방식으로 테이블 엑세스

백엔드에서 테이블은 Artifacts로 유지됩니다. 특정 버전에 엑세스하려면 아티팩트 API를 사용하여 엑세스할 수 있습니다.

```python
with wandb.init() as run:
    my_table = run.use_artifact("run-<run-id>-<table-name>:<tag>").get("<table-name>")
```

Artifacts에 대한 자세한 내용은 개발자 가이드의 [Artifacts 챕터]({{< relref path="/guides/core/artifacts/" lang="ko" >}})를 참조하십시오.

### 테이블 시각화

이러한 방식으로 기록된 테이블은 Run 페이지와 Project 페이지 모두의 Workspace에 표시됩니다. 자세한 내용은 [테이블 시각화 및 분석]({{< relref path="/guides/models/tables//visualize-tables.md" lang="ko" >}})을 참조하십시오.

## 아티팩트 테이블

`artifact.add()`를 사용하여 워크스페이스 대신 run의 Artifacts 섹션에 테이블을 기록합니다. 이는 한 번 기록한 다음 향후 run에 참조할 데이터셋이 있는 경우에 유용할 수 있습니다.

```python
run = wandb.init(project="my_project")
# 각 의미 있는 단계에 대한 wandb Artifact 생성
test_predictions = wandb.Artifact("mnist_test_preds", type="predictions")

# [위와 같이 예측 데이터 빌드]
test_table = wandb.Table(data=data, columns=columns)
test_predictions.add(test_table, "my_test_key")
run.log_artifact(test_predictions)
```

이미지 데이터와 함께 `artifact.add()`의 [자세한 예](http://wandb.me/dsviz-nature-colab)는 이 Colab을 참조하고, Artifacts 및 Tables를 사용하여 [테이블 형식 데이터의 버전 제어 및 중복 제거](http://wandb.me/TBV-Dedup)하는 방법의 예는 이 리포트를 참조하십시오.

### 아티팩트 테이블 결합

`wandb.JoinedTable(table_1, table_2, join_key)`를 사용하여 로컬에서 구성한 테이블 또는 다른 아티팩트에서 검색한 테이블을 결합할 수 있습니다.

| 인수       | 설명                                                                                                                    |
| --------- | --------------------------------------------------------------------------------------------------------------------- |
| table_1  | (str, `wandb.Table`, ArtifactEntry) 아티팩트의 `wandb.Table` 경로, 테이블 오브젝트 또는 ArtifactEntry |
| table_2  | (str, `wandb.Table`, ArtifactEntry) 아티팩트의 `wandb.Table` 경로, 테이블 오브젝트 또는 ArtifactEntry |
| join_key | (str, [str, str]) 결합을 수행할 키                                                                                              |

아티팩트 컨텍스트에서 이전에 기록한 두 개의 테이블을 결합하려면 아티팩트에서 가져와 결과를 새 테이블로 결합합니다.

예를 들어 `'original_songs'`라는 원본 노래의 테이블 하나와 동일한 노래의 합성 버전의 또 다른 테이블인 `'synth_songs'`를 읽는 방법을 보여줍니다. 다음 코드 예제는 `"song_id"`에서 두 테이블을 결합하고 결과 테이블을 새 W&B 테이블로 업로드합니다.

```python
import wandb

run = wandb.init(project="my_project")

# 원본 노래 테이블 가져오기
orig_songs = run.use_artifact("original_songs:latest")
orig_table = orig_songs.get("original_samples")

# 합성 노래 테이블 가져오기
synth_songs = run.use_artifact("synth_songs:latest")
synth_table = synth_songs.get("synth_samples")

# "song_id"에서 테이블 결합
join_table = wandb.JoinedTable(orig_table, synth_table, "song_id")
join_at = wandb.Artifact("synth_summary", "analysis")

# 아티팩트에 테이블을 추가하고 W&B에 기록
join_at.add(join_table, "synth_explore")
run.log_artifact(join_at)
```

서로 다른 Artifact 오브젝트에 저장된 두 개의 이전에 저장된 테이블을 결합하는 방법의 예는 [이 튜토리얼](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)을 참조하십시오.
