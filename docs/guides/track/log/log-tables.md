---
title: Log tables
description: W&B로 테이블 로그.
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb'/>

`wandb.Table`을 사용하여 W&B에서 시각화하고 쿼리할 데이터 로그를 생성하세요. 이 가이드에서는 다음을 배웁니다:

1. [테이블 생성](./log-tables.md#create-tables)
2. [데이터 추가](./log-tables.md#add-data)
3. [데이터 검색](./log-tables.md#retrieve-data)
4. [테이블 저장](./log-tables.md#save-tables)

## 테이블 생성

테이블을 정의하려면 데이터의 각 행에서 보고 싶은 열을 지정하세요. 각 행은 트레이닝 데이터셋에 있는 단일 항목이거나, 트레이닝 중 특정 단계나 에포크, 테스트 항목에 대한 모델의 예측값, 모델에 의해 생성된 오브젝트일 수 있습니다. 각 열은 고정된 타입을 가집니다: 숫자, 텍스트, 불리언, 이미지, 비디오, 오디오 등. 미리 타입을 지정할 필요는 없습니다. 각 열에 이름을 부여하고 해당 타입의 데이터만 그 열 인덱스에 전달하십시오. 더 자세한 예시는 [이 리포트](https://wandb.ai/stacey/mnist-viz/reports/Guide-to-W-B-Tables--Vmlldzo2NTAzOTk#1.-how-to-log-a-wandb.table)를 참조하십시오.

`wandb.Table` 생성자를 두 가지 방법 중 하나로 사용하세요:

1. **행 목록:** 이름이 지정된 열과 데이터 행을 로그합니다. 예를 들어, 아래의 코드조각은 두 행과 세 열이 있는 테이블을 생성합니다:

```python
wandb.Table(columns=["a", "b", "c"], data=[["1a", "1b", "1c"], ["2a", "2b", "2c"]])
```

2. **Pandas DataFrame:** DataFrame을 `wandb.Table(dataframe=my_df)`를 사용해 로그하세요. 열 이름은 DataFrame에서 추출됩니다.

#### 기존 배열이나 데이터프레임에서

```python
# 네 장의 이미지에 대해 모델이 예측한 결과값이 있다고 가정합니다
# 다음 필드들이 가능하다고 가정합니다:
# - 이미지 ID
# - wandb.Image()로 감싸여진 이미지 픽셀
# - 모델이 예측한 레이블
# - 그라운드 트루스 레이블
my_data = [
    [0, wandb.Image("img_0.jpg"), 0, 0],
    [1, wandb.Image("img_1.jpg"), 8, 0],
    [2, wandb.Image("img_2.jpg"), 7, 1],
    [3, wandb.Image("img_3.jpg"), 1, 1],
]

# 해당 열을 이용해 wandb.Table()을 생성합니다
columns = ["id", "image", "prediction", "truth"]
test_table = wandb.Table(data=my_data, columns=columns)
```

## 데이터 추가

테이블은 변경 가능합니다. 스크립트가 실행되는 동안 최대 200,000개의 행까지 테이블에 더 많은 데이터를 추가할 수 있습니다. 테이블에 데이터를 추가하는 방법은 두 가지가 있습니다:

1. **행 추가**: `table.add_data("3a", "3b", "3c")`. 새 행은 리스트로 표현되지 않는 점을 주의하세요. 만약 행이 리스트 형식이라면, 리스트를 위치 인수로 확장하기 위해 별표(*) 표기법을 사용하세요: `table.add_data(*my_row_list)`. 행은 테이블에 있는 열의 수와 같은 수의 엔트리를 포함해야 합니다.
2. **열 추가**: `table.add_column(name="col_name", data=col_data)`. `col_data`의 길이는 테이블의 현재 행 수와 같아야 합니다. 여기서 `col_data`는 리스트 데이터나 NumPy NDArray일 수 있습니다.

#### 데이터 점진적 추가

```python
# 위와 같은 열로 테이블을 생성하되,
# 레이블에 대한 confidence 점수를 추가합니다
columns = ["id", "image", "guess", "truth"]
for digit in range(10):
    columns.append("score_" + str(digit))
test_table = wandb.Table(columns=columns)

# 모든 이미지에 대한 추론을 수행하며, my_model이
# 예측한 레이블과 그라운드 트루스 레이블이 있다는 가정하에
for img_id, img in enumerate(mnist_test_data):
    true_label = mnist_test_data_labels[img_id]
    guess_label = my_model.predict(img)
    test_table.add_data(img_id, wandb.Image(img), guess_label, true_label)
```

## 데이터 검색

데이터가 테이블에 들어가면 열이나 행으로 엑세스할 수 있습니다:

1. **행 반복자**: 사용자는 `for ndx, row in table.iterrows(): ...`와 같은 테이블의 행 반복자를 사용하여 데이터의 행을 효율적으로 반복할 수 있습니다.
2. **열 가져오기**: 사용자는 `table.get_column("col_name")`를 사용하여 열 데이터를 검색할 수 있습니다. 편의를 위해, 사용자는 `convert_to="numpy"`를 전달하여 열을 NumPy NDArray의 프리미티브로 변환할 수 있습니다. 이는 `wandb.Image`와 같은 미디어 타입이 포함된 열의 기본 데이터를 직접 엑세스할 수 있도록 유용합니다.

## 테이블 저장

스크립트에서 데이터 테이블을 생성한 후, 예를 들어 모델 예측의 테이블이라면, W&B에 저장하여 결과를 실시간으로 시각화하세요.

### 테이블을 run에 로그

`wandb.log()`를 사용하여 테이블을 run에 저장하세요:

```python
run = wandb.init()
my_table = wandb.Table(columns=["a", "b"], data=[["1a", "1b"], ["2a", "2b"]])
run.log({"table_key": my_table})
```

같은 키에 테이블이 로그될 때마다 새로운 버전의 테이블이 생성되어 백엔드에 저장됩니다. 이는 여러 트레이닝 단계에 걸쳐 모델 예측이 어떻게 개선되는지를 확인하거나, 서로 다른 run의 테이블을 비교할 수 있게 하며, 동일한 키로 로그된 경우 가능합니다. 최대 200,000개의 행까지 로그할 수 있습니다.

:::info
200,000개 이상의 행을 로그하려면 다음과 같이 제한을 초과할 수 있습니다:

`wandb.Table.MAX_ARTIFACTS_ROWS = X`

그러나 이는 UI에서 쿼리 속도 저하와 같은 성능 문제를 일으킬 수 있습니다.
:::

### 프로그래밍 방식으로 테이블에 엑세스

백엔드에서는 테이블이 Artifacts로 저장됩니다. 특정 버전에 엑세스하고 싶다면 Artifact API를 사용할 수 있습니다:

```python
with wandb.init() as run:
    my_table = run.use_artifact("run-<run-id>-<table-name>:<tag>").get("<table-name>")
```

Artifacts에 대한 더 많은 정보는 Developer Guide의 [Artifacts Chapter](../../artifacts/intro.md)를 참조하세요.

### 테이블 시각화

이렇게 로그된 모든 테이블은 Workspace 내의 Run 페이지와 Project 페이지 모두에 표시됩니다. 더 많은 정보는 [테이블 시각화 및 분석](../../tables/visualize-tables.md)를 참조하세요.

## 아티팩트 테이블

`artifact.add()`를 사용하여 작업공간 대신 run의 Artifacts 섹션에 테이블을 기록하세요. 이는 한 번만 기록하고, 나중에 참조하고 싶은 데이터셋이 있을 경우 유용합니다.

```python
run = wandb.init(project="my_project")
# 각 유의미한 단계에 대해 wandb Artifact를 생성합니다
test_predictions = wandb.Artifact("mnist_test_preds", type="predictions")

# [위에서처럼 예측 데이터 작성]
test_table = wandb.Table(data=data, columns=columns)
test_predictions.add(test_table, "my_test_key")
run.log_artifact(test_predictions)
```

이 Colab에서 이미지 데이터가 포함된 [artifact.add()의 자세한 예제](http://wandb.me/dsviz-nature-colab)와 Artifacts 및 Tables를 사용하여 [버전 관리를 하고 테이블 데이터를 중복 제거하는 방법](http://wandb.me/TBV-Dedup)에 대해 알아보세요.

### 아티팩트 테이블 조인

로컬에서 생성했거나 다른 아티팩트에서 가져온 테이블을 `wandb.JoinedTable(table_1, table_2, join_key)`를 사용하여 조인할 수 있습니다.

| Args      | Description                                                                                                        |
| --------- | ------------------------------------------------------------------------------------------------------------------ |
| table_1  | (str, `wandb.Table`, ArtifactEntry) 아티팩트 내의 `wandb.Table` 경로, 테이블 오브젝트, 또는 ArtifactEntry |
| table_2  | (str, `wandb.Table`, ArtifactEntry) 아티팩트 내의 `wandb.Table` 경로, 테이블 오브젝트, 또는 ArtifactEntry |
| join_key | (str, [str, str]) 연결을 수행할 키 또는 키들                                                             |

아티팩트 컨텍스트에 이전에 로그한 두 테이블을 조인하려면 아티팩트에서 해당 테이블을 가져와 결과를 새 테이블로 조인하세요.

예를 들어, `'original_songs'`이라는 원본 노래 테이블과 같은 노래의 합성 버전인 `'synth_songs'` 테이블을 읽는 방법을 보여줍니다. 진행 코드 예제는 두 테이블을 `"song_id"`로 조인하여 결과 테이블을 새로운 W&B 테이블로 업로드합니다:

```python
import wandb

run = wandb.init(project="my_project")

# 원본 노래 테이블 가져오기
orig_songs = run.use_artifact("original_songs:latest")
orig_table = orig_songs.get("original_samples")

# 합성된 노래 테이블 가져오기
synth_songs = run.use_artifact("synth_songs:latest")
synth_table = synth_songs.get("synth_samples")

# "song_id"로 테이블 조인
join_table = wandb.JoinedTable(orig_table, synth_table, "song_id")
join_at = wandb.Artifact("synth_summary", "analysis")

# 아티팩트에 테이블을 추가하고 W&B에 로그
join_at.add(join_table, "synth_explore")
run.log_artifact(join_at)
```

[이 튜토리얼을 읽어보세요](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) 두 개의 서로 다른 Artifact 오브젝트에 저장된 두 테이블을 결합하는 방법에 대한 예제를 보려면.