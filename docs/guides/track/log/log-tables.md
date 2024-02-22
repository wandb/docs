---
description: Log tables with W&B.
displayed_sidebar: default
---

# 로그 테이블

`wandb.Table`을 사용하여 W&B에서 시각화 및 쿼리할 수 있는 데이터를 로그합니다. 이 가이드에서는 다음을 배우게 됩니다:

1. [테이블 생성](./log-tables.md#create-tables)
2. [데이터 추가](./log-tables.md#add-data)
3. [데이터 검색](./log-tables.md#retrieve-data)
4. [테이블 저장](./log-tables.md#save-tables)

## 테이블 생성

테이블을 정의하려면 각 데이터 행에 대해 보고 싶은 열을 지정하세요. 각 행은 귀하의 학습 데이터세트의 단일 항목, 학습 중 특정 단계 또는 에포크, 테스트 항목에 대한 모델의 예측값, 모델이 생성한 개체 등이 될 수 있습니다. 각 열은 고정된 유형을 가지고 있습니다: 숫자, 텍스트, 부울, 이미지, 비디오, 오디오 등. 유형을 미리 지정할 필요는 없습니다. 각 열에 이름을 지정하고 해당 열 인덱스에 해당 유형의 데이터만 전달하도록 하세요. 좀 더 자세한 예제는 [이 리포트](https://wandb.ai/stacey/mnist-viz/reports/Guide-to-W-B-Tables--Vmlldzo2NTAzOTk#1.-how-to-log-a-wandb.table)를 참조하세요.

`wandb.Table` 생성자를 두 가지 방법 중 하나로 사용할 수 있습니다:

1. **행 목록:** 명명된 열과 데이터 행을 로그합니다. 예를 들어, 다음 코드 조각은 두 행과 세 열이 있는 테이블을 생성합니다:

```python
wandb.Table(columns=["a", "b", "c"], data=[["1a", "1b", "1c"], ["2a", "2b", "2c"]])
```


2. **Pandas DataFrame:** `wandb.Table(dataframe=my_df)`를 사용하여 DataFrame을 로그합니다. 열 이름은 DataFrame에서 추출됩니다.

#### 기존 배열 또는 데이터프레임에서

```python
# 모델이 네 개의 이미지에 대한 예측값을 반환했다고 가정합니다
# 다음 필드를 사용할 수 있습니다:
# - 이미지 ID
# - wandb.Image()에 래핑된 이미지 픽셀
# - 모델의 예측된 레이블
# - 실제값 레이블
my_data = [
    [0, wandb.Image("img_0.jpg"), 0, 0],
    [1, wandb.Image("img_1.jpg"), 8, 0],
    [2, wandb.Image("img_2.jpg"), 7, 1],
    [3, wandb.Image("img_3.jpg"), 1, 1],
]

# 해당 열이 있는 wandb.Table()을 생성합니다
columns = ["id", "image", "prediction", "truth"]
test_table = wandb.Table(data=my_data, columns=columns)
```

## 데이터 추가

테이블은 변경 가능합니다. 스크립트가 실행됨에 따라 테이블에 더 많은 데이터를 추가할 수 있으며, 최대 200,000행까지 추가할 수 있습니다. 테이블에 데이터를 추가하는 두 가지 방법은 다음과 같습니다:

1. **행 추가:** `table.add_data("3a", "3b", "3c")`. 새 행은 목록으로 표현되지 않습니다. 행이 목록 형식인 경우, 목록을 위치 인수로 확장하려면 별표 표기법 `*`을 사용하세요: `table.add_data(*my_row_list)`. 행은 테이블의 열 수와 동일한 수의 항목을 포함해야 합니다.
2. **열 추가:** `table.add_column(name="col_name", data=col_data)`. 여기서 `col_data`의 길이는 테이블의 현재 행 수와 같아야 합니다. `col_data`는 리스트 데이터 또는 NumPy NDArray일 수 있습니다.

#### 점진적으로 데이터 추가

```python
# 위와 같은 열을 가진 테이블을 생성하되,
# 모든 레이블에 대한 신뢰도 점수 추가
columns = ["id", "image", "guess", "truth"]
for digit in range(10):
    columns.append("score_" + str(digit))
test_table = wandb.Table(columns=columns)

# 모든 이미지에 대해 추론을 실행하고, my_model이 예측한
# 레이블을 반환하며, 실제값 레이블이 제공된다고 가정합니다
for img_id, img in enumerate(mnist_test_data):
    true_label = mnist_test_data_labels[img_id]
    guess_label = my_model.predict(img)
    test_table.add_data(img_id, wandb.Image(img), guess_label, true_label)
```

## 데이터 검색

테이블에 데이터가 있으면 열 또는 행별로 액세스할 수 있습니다:

1. **행 반복자:** 사용자는 `for ndx, row in table.iterrows(): ...`와 같은 테이블의 행 반복자를 사용하여 데이터의 행을 효율적으로 반복할 수 있습니다.
2. **열 가져오기:** 사용자는 `table.get_column("col_name")`을 사용하여 데이터의 열을 검색할 수 있습니다. 편의를 위해, 사용자는 `convert_to="numpy"`를 전달하여 열을 기본 데이터가 있는 NumPy NDArray로 변환할 수 있습니다. 이는 열에 `wandb.Image`과 같은 미디어 유형이 포함되어 있어 기본 데이터에 직접 액세스하려는 경우 유용합니다.

## 테이블 저장

스크립트에서 데이터 테이블을 생성한 후, 예를 들어 모델 예측값의 테이블인 경우, W&B에 저장하여 실시간으로 결과를 시각화합니다.

### 실행에 테이블 로그

`wandb.log()`를 사용하여 실행에 테이블을 저장하세요:

```python
run = wandb.init()
my_table = wandb.Table(columns=["a", "b"], data=[["1a", "1b"], ["2a", "2b"]])
run.log({"table_key": my_table})
```

같은 키에 대해 테이블이 로그될 때마다 테이블의 새 버전이 생성되어 백엔드에 저장됩니다. 이는 여러 학습 단계에 걸쳐 같은 테이블을 로그하여 모델 예측값이 시간에 따라 어떻게 개선되는지 보거나, 다른 실행과 테이블을 비교할 수 있다는 것을 의미합니다. 최대 200,000행까지 로그할 수 있습니다.

:::안내
200,000행 이상을 로그하려면 다음과 같이 제한을 무시할 수 있습니다:

`wandb.Table.MAX_ARTIFACTS_ROWS = X`

그러나, 이는 UI에서 느린 쿼리와 같은 성능 문제를 초래할 가능성이 있습니다.
:::

### 프로그래매틱으로 테이블 액세스

백엔드에서 테이블은 아티팩트로 유지됩니다. 특정 버전에 액세스하고 싶다면 아티팩트 API를 사용할 수 있습니다:

```python
with wandb.init() as run:
    my_table = run.use_artifact("run-<run-id>-<table-name>:<tag>").get("<table-name>")
```

아티팩트에 대한 자세한 내용은 개발자 가이드의 [아티팩트 챕터](../../artifacts/intro.md)를 참조하세요.

## 테이블 시각화

이러한 방식으로 로그된 모든 테이블은 실행 페이지와 프로젝트 페이지 모두에서 워크스페이스에 표시됩니다. 자세한 내용은 [테이블 시각화 및 분석](../../tables/visualize-tables.md)을 참조하세요.

## 고급: 아티팩트 테이블

`artifact.add()`를 사용하여 실행의 아티팩트 섹션에 테이블을 로그하세요. 워크스페이스가 아닙니다. 이는 미래의 실행을 위해 한 번 로그하고 참조하고 싶은 데이터세트가 있는 경우 유용할 수 있습니다.

```python
run = wandb.init(project="my_project")
# 의미 있는 각 단계마다 wandb 아티팩트 생성
test_predictions = wandb.Artifact("mnist_test_preds", type="predictions")

# [위에서 예측 데이터를 구축합니다]
test_table = wandb.Table(data=data, columns=columns)
test_predictions.add(test_table, "my_test_key")
run.log_artifact(test_predictions)
```

[image data와 함께 artifact.add()를 사용하는 자세한 예제](http://wandb.me/dsviz-nature-colab)에 대한 이 Colab을 참조하고, 아티팩트 및 테이블을 사용하여 [테이블 데이터의 버전 관리 및 중복 제거](http://wandb.me/TBV-Dedup)하는 방법에 대한 예제를 위한 이 리포트를 참조하세요.

### 아티팩트 테이블 결합

`wandb.JoinedTable(table_1, table_2, join_key)`를 사용하여 로컬에서 구성한 테이블이나 다른 아티팩트에서 검색한 테이블을 결합할 수 있습니다.

| Args      | 설명                                                                                                        |
| --------- | ------------------------------------------------------------------------------------------------------------------ |
| table_1  | (str, `wandb.Table`, ArtifactEntry) 아티팩트의 `wandb.Table`로의 경로, 테이블 객체, 또는 ArtifactEntry |
| table_2  | (str, `wandb.Table`, ArtifactEntry) 아티팩트의 `wandb.Table`로의 경로, 테이블 객체, 또는 ArtifactEntry |
| join_key | (str, [str, str]) 조인을 수행할 키 또는 키들                                                        |


아티팩트 컨텍스트에서 이전에 로그한 두 테이블을 결합하려면 아티팩트에서 가져온 다음 새 테이블로 결합 결과를 저장합니다.

예를 들어, `'original_songs'`라는 원래 노래의 테이블과 같은 노래의 합성 버전을 포함하는 `'synth_songs'`라는 다른 테이블을 읽는 방법을 보여줍니다. 다음 코드 예시는 두 테이블을 `"song_id"`에 대해 결합하고 결과 테이블을 새로운 W&B 테이블로 업로드합니다:

```python
import wandb

run = wandb.init(project="my_project")

# 원래 노래 테이블 가져오기
orig_songs = run.use_artifact("original_songs:latest")
orig_table = orig_songs.get("original_samples")

# 합성 노래 테이블 가져오기
synth_songs = run.use_artifact("synth_songs:latest")
synth_table = synth_songs.get("synth_samples")

# "song_id"에 대해 테이블 결합
join_table = wandb.JoinedTable(orig_table, synth_table, "song_id")
join_at = wandb.Artifact("synth_summary", "analysis")

# 테이블을 아티팩트에 추가하고 W&B에 로그
join_at.add(join_table, "synth_explore")
run.log_artifact(join_at)
```

다른 아티팩트 객체에 저장된 두 테이블을 결합하는 방법에 대한 예를 위해 이 [Colab 노트북](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)을 탐색하세요.