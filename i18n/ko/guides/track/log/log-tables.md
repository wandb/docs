---
description: Log tables with W&B.
displayed_sidebar: default
---

# 로그 테이블

`wandb.Table`을 사용하여 W&B에서 시각화하고 쿼리할 수 있는 데이터를 로그하세요. 이 가이드에서는 다음을 수행하는 방법을 배웁니다:

1. [테이블 생성](./log-tables.md#create-tables)
2. [데이터 추가](./log-tables.md#add-data)
3. [데이터 검색](./log-tables.md#retrieve-data)
4. [테이블 저장](./log-tables.md#save-tables)

## 테이블 생성

테이블을 정의하려면 각 데이터 행에 대해 보고 싶은 열을 지정하세요. 각 행은 교육 데이터셋의 단일 항목, 교육 중 특정 단계나 에포크, 테스트 항목에 대한 모델의 예측값, 모델이 생성한 오브젝트 등일 수 있습니다. 각 열은 고정 유형을 가지고 있습니다: 숫자, 텍스트, 불린, 이미지, 비디오, 오디오 등. 유형을 미리 지정할 필요는 없습니다. 각 열에 이름을 지정하고 해당 열 인덱스에 해당 유형의 데이터만 전달하세요. 더 자세한 예시는 [이 리포트](https://wandb.ai/stacey/mnist-viz/reports/Guide-to-W-B-Tables--Vmlldzo2NTAzOTk#1.-how-to-log-a-wandb.table)를 참조하세요.

`wandb.Table` 생성자를 두 가지 방법 중 하나로 사용하세요:

1. **행의 목록:** 이름이 지정된 열과 데이터 행을 로그합니다. 예를 들어 다음 코드 조각은 두 행과 세 열이 있는 테이블을 생성합니다:

```python
wandb.Table(columns=["a", "b", "c"], data=[["1a", "1b", "1c"], ["2a", "2b", "2c"]])
```


2. **Pandas DataFrame:** `wandb.Table(dataframe=my_df)`를 사용하여 DataFrame을 로그합니다. 열 이름은 DataFrame에서 추출됩니다.

#### 기존 배열이나 데이터프레임에서

```python
# 모델이 네 개의 이미지에 대한 예측값을 반환했다고 가정
# 다음과 같은 필드가 사용 가능합니다:
# - 이미지 ID
# - 이미지 픽셀, wandb.Image()로 래핑됨
# - 모델의 예측된 라벨
# - 그라운드 트루스 라벨
my_data = [
    [0, wandb.Image("img_0.jpg"), 0, 0],
    [1, wandb.Image("img_1.jpg"), 8, 0],
    [2, wandb.Image("img_2.jpg"), 7, 1],
    [3, wandb.Image("img_3.jpg"), 1, 1],
]

# 해당 열을 가진 wandb.Table() 생성
columns = ["id", "image", "prediction", "truth"]
test_table = wandb.Table(data=my_data, columns=columns)
```

## 데이터 추가

테이블은 변경 가능합니다. 스크립트가 실행됨에 따라 테이블에 데이터를 추가할 수 있습니다. 최대 200,000행까지 추가 가능합니다. 테이블에 데이터를 추가하는 두 가지 방법이 있습니다:

1. **행 추가**: `table.add_data("3a", "3b", "3c")`. 새 행은 리스트로 표현되지 않습니다. 행이 리스트 형식인 경우, 리스트를 위치 인수로 확장하려면 별표 표기법 `*`를 사용하세요: `table.add_data(*my_row_list)`. 행에는 테이블의 열 수와 동일한 수의 항목이 포함되어야 합니다.
2. **열 추가**: `table.add_column(name="col_name", data=col_data)`. 여기서 `col_data`의 길이는 테이블의 현재 행 수와 같아야 합니다. 여기서 `col_data`는 리스트 데이터 또는 NumPy NDArray일 수 있습니다.

#### 점진적으로 데이터 추가하기

```python
# 위와 동일한 열을 가진 테이블을 생성하되,
# 모든 라벨에 대한 확신 점수 추가
columns = ["id", "image", "guess", "truth"]
for digit in range(10):
    columns.append("score_" + str(digit))
test_table = wandb.Table(columns=columns)

# 모든 이미지에 대해 추론 실행, 가정: my_model은 예측된 라벨을 반환하고
# 그라운드 트루스 라벨이 사용 가능
for img_id, img in enumerate(mnist_test_data):
    true_label = mnist_test_data_labels[img_id]
    guess_label = my_model.predict(img)
    test_table.add_data(img_id, wandb.Image(img), guess_label, true_label)
```

## 데이터 검색

테이블에 데이터가 있으면 열이나 행별로 액세스할 수 있습니다:

1. **행 이터레이터**: 사용자는 `for ndx, row in table.iterrows(): ...`와 같은 테이블의 행 이터레이터를 사용하여 데이터의 행을 효율적으로 반복할 수 있습니다.
2. **열 가져오기**: 사용자는 `table.get_column("col_name")`을 사용하여 데이터 열을 검색할 수 있습니다. 편의를 위해, 사용자는 `convert_to="numpy"`를 전달하여 열을 원시 데이터의 NumPy NDArray로 변환할 수 있습니다. 이는 열에 미디어 유형(예: `wandb.Image`)이 포함된 경우 기본 데이터에 직접 액세스할 수 있도록 유용합니다.

## 테이블 저장

스크립트에서 데이터 테이블(예: 모델 예측의 테이블)을 생성한 후, W&B에 저장하여 실시간으로 결과를 시각화하세요.

### run에 테이블 로그하기

`wandb.log()`를 사용하여 run에 테이블을 저장하세요, 예를 들어:

```python
run = wandb.init()
my_table = wandb.Table(columns=["a", "b"], data=[["1a", "1b"], ["2a", "2b"]])
run.log({"table_key": my_table})
```

같은 키에 테이블이 로그될 때마다 테이블의 새 버전이 생성되어 백엔드에 저장됩니다. 이는 모델 예측이 시간에 따라 어떻게 개선되는지 보거나, 다른 run들과 테이블을 비교할 때 동일한 키에 로그된 경우 유용합니다. 최대 200,000행까지 로그할 수 있습니다.

:::안내
200,000행 이상을 로그하려면 다음과 같이 제한을 무시할 수 있습니다:

`wandb.Table.MAX_ARTIFACTS_ROWS = X`

그러나 이는 UI에서 더 느린 쿼리와 같은 성능 문제를 일으킬 수 있습니다.
:::

### 프로그래매틱하게 테이블 액세스하기

백엔드에서 테이블은 Artifacts로 유지됩니다. 특정 버전에 액세스하고 싶다면 아티팩트 API로 할 수 있습니다:

```python
with wandb.init() as run:
    my_table = run.use_artifact("run-<run-id>-<table-name>:<tag>").get("<table-name>")
```

Artifacts에 대한 자세한 내용은 개발자 가이드의 [Artifacts 챕터](../../artifacts/intro.md)를 참조하세요.

## 테이블 시각화하기

이와 같이 로그된 모든 테이블은 Run 페이지와 프로젝트 페이지 모두에서 워크스페이스에 표시됩니다. 자세한 정보는 [테이블 시각화 및 분석](../../tables/visualize-tables.md)을 참조하세요.

## 고급: 아티팩트 테이블

워크스페이스 대신 run의 Artifacts 섹션에 테이블을 로그하려면 `artifact.add()`를 사용하세요. 이는 나중에 참조하기 위해 한 번 로그하고 싶은 데이터셋이 있을 때 유용할 수 있습니다.

```python
run = wandb.init(project="my_project")
# 각 의미 있는 단계에 대한 wandb Artifact 생성
test_predictions = wandb.Artifact("mnist_test_preds", type="predictions")

# [위와 같이 예측 데이터 구축]
test_table = wandb.Table(data=data, columns=columns)
test_predictions.add(test_table, "my_test_key")
run.log_artifact(test_predictions)
```

[image data를 이용한 artifact.add()의 자세한 예제](http://wandb.me/dsviz-nature-colab)에 대해서는 이 Colab을 참조하고, Artifacts와 테이블을 사용하여 [탭 데이터의 버전 관리 및 중복 제거](http://wandb.me/TBV-Dedup)하는 방법에 대한 예시는 이 리포트를 참조하세요.

### 아티팩트 테이블 조인하기

로컬에서 구성하거나 다른 아티팩트에서 검색한 테이블을 `wandb.JoinedTable(table_1, table_2, join_key)`을 사용하여 조인할 수 있습니다.

| Args      | 설명                                                                                                       |
| --------- | --------------------------------------------------------------------------------------------------------- |
| table_1  | (str, `wandb.Table`, ArtifactEntry) 아티팩트에서 `wandb.Table`로의 경로, 테이블 객체, 또는 ArtifactEntry |
| table_2  | (str, `wandb.Table`, ArtifactEntry) 아티팩트에서 `wandb.Table`로의 경로, 테이블 객체, 또는 ArtifactEntry |
| join_key | (str, [str, str]) 조인을 수행할 키 또는 키들                                                             |

아티팩트 컨텍스트에서 이전에 로그한 두 테이블을 조인하려면, 아티팩트에서 두 테이블을 가져와 새로운 테이블로 조인 결과를 페치하세요.

예를 들어, `'original_songs'`라고 불리는 원래 노래들의 테이블과 같은 노래들의 합성 버전을 포함하는 `'synth_songs'`라는 또 다른 테이블을 읽는 방법을 보여줍니다. 다음 코드 예제는 두 테이블을 `"song_id"`에 대해 조인하고 결과 테이블을 새로운 W&B 테이블로 업로드합니다:

```python
import wandb

run = wandb.init(project="my_project")

# 원래 노래 테이블 가져오기
orig_songs = run.use_artifact("original_songs:latest")
orig_table = orig_songs.get("original_samples")

# 합성 노래 테이블 가져오기
synth_songs = run.use_artifact("synth_songs:latest")
synth_table = synth_songs.get("synth_samples")

# "song_id"를 기준으로 테이블 조인
join_table = wandb.JoinedTable(orig_table, synth_table, "song_id")
join_at = wandb.Artifact("synth_summary", "analysis")

# 아티팩트에 테이블 추가하고 W&B에 로그하기
join_at.add(join_table, "synth_explore")
run.log_artifact(join_at)
```

서로 다른 Artifact 객체에 저장된 두 테이블을 결합하는 방법에 대한 예시는 이 [Colab 노트북](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)을 탐색하세요.