---
title: 테이블 로그
description: W&B로 테이블을 로그하세요.
menu:
  default:
    identifier: ko-guides-models-track-log-log-tables
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}
`wandb.Table`을 사용해 데이터를 기록하고 시각화 및 쿼리를 W&B에서 해보세요. 이 가이드에서는 다음과 같은 내용을 다룹니다:

1. [테이블 생성하기]({{< relref path="./log-tables.md#create-tables" lang="ko" >}})
2. [데이터 추가하기]({{< relref path="./log-tables.md#add-data" lang="ko" >}})
3. [데이터 불러오기]({{< relref path="./log-tables.md#retrieve-data" lang="ko" >}})
4. [테이블 저장하기]({{< relref path="./log-tables.md#save-tables" lang="ko" >}})

## 테이블 생성하기

Table을 정의하려면 각 데이터 행에 보고 싶은 컬럼을 지정하세요. 각 행은 트레이닝 데이터셋 안의 한 아이템, 트레이닝 중의 특정 step이나 epoch, 테스트 데이터에 대해 모델이 내린 예측값, 모델이 생성한 오브젝트 등일 수 있습니다. 각 컬럼은 고정 타입(숫자, 텍스트, 불리언, 이미지, 비디오, 오디오 등)을 갖습니다. 타입을 미리 지정할 필요는 없습니다. 각 컬럼에 이름을 주고, 해당 컬럼 인덱스에는 그 타입에 맞는 데이터만 전달하면 됩니다. 더 자세한 예시는 [W&B Tables 가이드](https://wandb.ai/stacey/mnist-viz/reports/Guide-to-W-B-Tables--Vmlldzo2NTAzOTk#1.-how-to-log-a-wandb.table)를 참고하세요.

`wandb.Table` 생성자는 두 가지 방법 중 하나로 사용할 수 있습니다:

1. **행 리스트로 생성:** 이름이 지정된 컬럼과 행 데이터를 기록합니다. 예를 들어, 아래 코드조각은 두 행, 세 컬럼의 테이블을 생성합니다:

```python
wandb.Table(columns=["a", "b", "c"], data=[["1a", "1b", "1c"], ["2a", "2b", "2c"]])
```

2. **Pandas DataFrame 사용:** `wandb.Table(dataframe=my_df)`을 통해 DataFrame을 테이블로 기록합니다. 컬럼 이름은 DataFrame에서 추출됩니다.

#### 기존 array 혹은 dataframe에서 생성

```python
# 모델이 네 장의 이미지에 대한 예측값을 반환했다고 가정합시다.
# 다음과 같은 값들이 있습니다:
# - 이미지 id
# - 이미지 원본(픽셀값), wandb.Image()로 래핑됨
# - 모델의 예측 레이블
# - 그라운드 트루스(정답) 레이블
my_data = [
    [0, wandb.Image("img_0.jpg"), 0, 0],
    [1, wandb.Image("img_1.jpg"), 8, 0],
    [2, wandb.Image("img_2.jpg"), 7, 1],
    [3, wandb.Image("img_3.jpg"), 1, 1],
]

# 해당 컬럼에 맞춰 wandb.Table() 생성
columns = ["id", "image", "prediction", "truth"]
test_table = wandb.Table(data=my_data, columns=columns)
```

## 데이터 추가하기

Tables는 변경이 가능합니다. 스크립트가 실행되는 동안 최대 20만 행까지 데이터를 추가할 수 있습니다. 테이블에 데이터를 추가하는 방법은 두 가지가 있습니다:

1. **행 추가:** `table.add_data("3a", "3b", "3c")`처럼 사용합니다. 새로운 행을 리스트 형태로 넘기지 않습니다. 만약 리스트 형식의 행이 있다면, star notation `*`을 써서 각각의 값을 인자로 확장할 수 있습니다: `table.add_data(*my_row_list)`. 행의 항목 수는 테이블의 컬럼 수와 같아야 합니다.
2. **컬럼 추가:** `table.add_column(name="col_name", data=col_data)`를 사용합니다. `col_data`의 길이는 테이블의 현재 행 개수와 같아야 합니다. 여기서 `col_data`는 list 혹은 NumPy NDArray가 될 수 있습니다.

### 데이터 점진적으로 추가하기

아래 예시는 Table을 정의하고 데이터(특정 라벨에 대한 confidence score 포함)를 예측 시점마다 한 행씩 추가하는 방법을 보여줍니다. 또한 [Resumed Run에서 테이블에 데이터를 점진적으로 추가]({{< relref path="#adding-data-to-resumed-runs" lang="ko" >}})할 수도 있습니다.

```python
# 각 라벨별 confidence score를 포함한 컬럼 정의
columns = ["id", "image", "guess", "truth"]
for digit in range(10):  # 각 digit(0~9)에 대해 score 컬럼 추가
    columns.append(f"score_{digit}")

# 정의한 컬럼으로 테이블 초기화
test_table = wandb.Table(columns=columns)

# 테스트 데이터셋을 순회하며 데이터를 한 행씩 추가
for img_id, img in enumerate(mnist_test_data):
    true_label = mnist_test_data_labels[img_id]  # 그라운드 트루스(정답) 라벨
    guess_label = my_model.predict(img)  # 예측 라벨
    test_table.add_data(
        img_id, wandb.Image(img), guess_label, true_label
    )  # 새 행을 테이블에 추가
```

#### 재시작된 Run에서 데이터 추가

아티팩트에서 기존 테이블을 불러온 후 마지막 행을 가져와 메트릭을 업데이트한 다음, 호환성 유지를 위해 테이블을 재초기화하여 W&B에 다시 기록할 수 있습니다.

```python
import wandb

# Run을 시작합니다.
with wandb.init(project="my_project") as run:

    # 아티팩트에서 기존 테이블 불러오기
    best_checkpt_table = run.use_artifact(table_tag).get(table_name)

    # 테이블에서 마지막 행 데이터를 가져옵니다
    best_iter, best_metric_max, best_metric_min = best_checkpt_table.data[-1]

    # 필요에 따라 best metric을 업데이트

    # 업데이트된 데이터를 테이블에 추가
    best_checkpt_table.add_data(best_iter, best_metric_max, best_metric_min)

    # 호환성 유지를 위해 데이터를 사용해 테이블 재초기화
    best_checkpt_table = wandb.Table(
        columns=["col1", "col2", "col3"], data=best_checkpt_table.data
    )

    # Run을 다시 초기화
    run = wandb.init()

    # 업데이트된 테이블을 W&B에 기록
    run.log({table_name: best_checkpt_table})
```

## 데이터 불러오기

Table에 기록된 데이터는 컬럼 또는 행 단위로 엑세스할 수 있습니다:

1. **행 반복자(Row Iterator):** `for ndx, row in table.iterrows(): ...` 과 같이 Table의 row iterator를 사용해 효율적으로 반복할 수 있습니다.
2. **컬럼 가져오기:** `table.get_column("col_name")`로 데이터 컬럼을 가져옵니다. `convert_to="numpy"` 옵션으로 컬럼을 NumPy NDArray로 변환할 수도 있습니다. 컬럼에 `wandb.Image` 등 미디어 타입이 들어 있을 경우 원본 데이터 접근에 유용합니다.

## 테이블 저장하기

예를 들어, 모델 예측값 테이블을 스크립트에서 생성했다면 결과를 W&B에 저장해 즉시 시각화할 수 있습니다.

### run에 테이블 기록하기

`wandb.Run.log()`를 사용해 테이블을 run에 기록할 수 있습니다:

```python
with wandb.init() as run:
    my_table = wandb.Table(columns=["a", "b"], data=[["1a", "1b"], ["2a", "2b"]])
    run.log({"table_key": my_table})
```

동일한 키로 테이블을 여러번 기록하면, 각 기록 시마다 새 버전의 테이블이 생성되어 백엔드에 저장됩니다. 이를 통해 여러 트레이닝 step에 걸쳐 예측값 변화 추이를 확인하거나, 서로 다른 run의 테이블을 비교할 수 있습니다(같은 키로 기록한 경우). 한번에 최대 20만 행까지 기록할 수 있습니다.

{{% alert %}}
20만 행(200,000 rows) 이상을 기록하려면, 다음과 같이 제한을 변경할 수 있습니다:

`wandb.Table.MAX_ARTIFACT_ROWS = X`

단, 이 경우 UI에서 쿼리가 느려지는 등 성능 이슈가 발생할 수 있습니다.
{{% /alert %}}

### 프로그래밍 방식으로 테이블 엑세스하기

백엔드에서 Table은 Artifacts로 저장됩니다. 특정 버전에 엑세스하려면 artifact API를 사용하세요:

```python
with wandb.init() as run:
    my_table = run.use_artifact("run-<run-id>-<table-name>:<tag>").get("<table-name>")
```

Artifacts에 관한 더 자세한 사항은 개발자 가이드의 [Artifacts 챕터]({{< relref path="/guides/core/artifacts/" lang="ko" >}})를 참고하세요.

### 테이블 시각화

이 방식으로 기록된 테이블은 Run Page와 Project Page 등 Workspace에서 확인할 수 있습니다. 자세한 내용은 [테이블 시각화 및 분석]({{< relref path="/guides/models/tables//visualize-tables.md" lang="ko" >}})을 참고하세요.

## Artifact Tables

`artifact.add()`를 사용하면 테이블을 Run의 Artifacts 섹션에 기록할 수 있습니다(Workspace가 아님). 데이터셋을 한 번만 기록하고, 이후 여러 Run에서 참고하고 싶을 때 유용합니다.

```python
with wandb.init(project="my_project") as run:
    # 각 의미 있는 스텝마다 wandb Artifact 생성
    test_predictions = wandb.Artifact("mnist_test_preds", type="predictions")

    # 위에서와 같이 예측 데이터 준비
    test_table = wandb.Table(data=data, columns=columns)
    test_predictions.add(test_table, "my_test_key")
    run.log_artifact(test_predictions)
```

이미지 데이터를 artifact.add()로 기록하는 [상세 예시는 이 Colab](https://wandb.me/dsviz-nature-colab)에서, Artifacts와 Tables를 통한 [테이블 데이터의 버전 관리와 중복 제거](https://wandb.me/TBV-Dedup) 예시는 이 Report에서 확인할 수 있습니다.

### Artifact 테이블 조인(Join)

로컬에서 생성한 Table이나, 다른 artifact에서 불러온 Table을 `wandb.JoinedTable(table_1, table_2, join_key)`를 통해 조인할 수 있습니다.

| 인수(Args)  |  설명                                                                                                  |
| --------- | ------------------------------------------------------------------------------------------------------ |
| table_1  | (str, `wandb.Table`, ArtifactEntry) artifact 내 table 경로, table 오브젝트, 혹은 ArtifactEntry               |
| table_2  | (str, `wandb.Table`, ArtifactEntry) artifact 내 table 경로, table 오브젝트, 혹은 ArtifactEntry               |
| join_key | (str, [str, str]) 조인 기준이 되는 키 (하나 또는 여러 개)                                                  |

Artifact에 기록해둔 두 Table을 조인하려면, 각각 Artifact에서 꺼내온 뒤, 조인 결과를 새 Table로 만들어 기록하면 됩니다.

예를 들어, 다음 코드는 원본 곡들의 Table('original_songs')과 해당 곡의 합성 버전을 보관한 Table('synth_songs')을 `"song_id"`를 기준으로 조인한 후, 결과를 새로운 W&B Table로 업로드합니다:

```python
import wandb

with wandb.init(project="my_project") as run:

    # 원본 곡 테이블 불러오기
    orig_songs = run.use_artifact("original_songs:latest")
    orig_table = orig_songs.get("original_samples")

    # 합성 곡 테이블 불러오기
    synth_songs = run.use_artifact("synth_songs:latest")
    synth_table = synth_songs.get("synth_samples")

    # "song_id" 컬럼으로 조인하기
    join_table = wandb.JoinedTable(orig_table, synth_table, "song_id")
    join_at = wandb.Artifact("synth_summary", "analysis")

    # 조인 테이블을 artifact에 추가하고 W&B에 기록
    join_at.add(join_table, "synth_explore")
    run.log_artifact(join_at)
```

[이 튜토리얼](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)을 참고하면 서로 다른 Artifact에 저장된 두 Table을 합치는 방법을 자세하게 배울 수 있습니다.