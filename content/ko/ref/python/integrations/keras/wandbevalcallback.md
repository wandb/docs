---
title: WandbEvalCallback
menu:
  reference:
    identifier: ko-ref-python-integrations-keras-wandbevalcallback
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L10-L228 >}}

모델 예측 시각화를 위한 Keras 콜백을 빌드하는 추상 기본 클래스입니다.

```python
WandbEvalCallback(
    data_table_columns: List[str],
    pred_table_columns: List[str],
    *args,
    **kwargs
) -> None
```

분류, 오브젝트 검출, 분할 등 작업을 위해 `model.fit()`에 전달될 수 있는 `on_epoch_end`에서 모델 예측을 시각화하기 위한 콜백을 빌드할 수 있습니다.

이를 사용하려면 이 기본 콜백 클래스에서 상속받아 `add_ground_truth` 및 `add_model_prediction` 메소드를 구현합니다.

기본 클래스는 다음 사항을 처리합니다.

- 그라운드 트루스 로깅을 위한 `data_table` 및 예측을 위한 `pred_table`을 초기화합니다.
- `data_table`에 업로드된 데이터는 `pred_table`에 대한 참조로 사용됩니다. 이는 메모리 공간을 줄이기 위함입니다. `data_table_ref`는 참조된 데이터에 액세스하는 데 사용할 수 있는 목록입니다. 아래 예제를 통해 수행 방법을 확인하십시오.
- 테이블을 W&B Artifacts로 W&B에 로그합니다.
- 각 새 `pred_table`은 에일리어스와 함께 새 버전으로 로그됩니다.

#### 예시:

```python
class WandbClfEvalCallback(WandbEvalCallback):
    def __init__(self, validation_data, data_table_columns, pred_table_columns):
        super().__init__(data_table_columns, pred_table_columns)

        self.x = validation_data[0]
        self.y = validation_data[1]

    def add_ground_truth(self):
        for idx, (image, label) in enumerate(zip(self.x, self.y)):
            self.data_table.add_data(idx, wandb.Image(image), label)

    def add_model_predictions(self, epoch):
        preds = self.model.predict(self.x, verbose=0)
        preds = tf.argmax(preds, axis=-1)

        data_table_ref = self.data_table_ref
        table_idxs = data_table_ref.get_index()

        for idx in table_idxs:
            pred = preds[idx]
            self.pred_table.add_data(
                epoch,
                data_table_ref.data[idx][0],
                data_table_ref.data[idx][1],
                data_table_ref.data[idx][2],
                pred,
            )


model.fit(
    x,
    y,
    epochs=2,
    validation_data=(x, y),
    callbacks=[
        WandbClfEvalCallback(
            validation_data=(x, y),
            data_table_columns=["idx", "image", "label"],
            pred_table_columns=["epoch", "idx", "image", "label", "pred"],
        )
    ],
)
```

보다 세분화된 제어를 위해 `on_train_begin` 및 `on_epoch_end` 메소드를 재정의할 수 있습니다. N개 배치 후 샘플을 기록하려면 `on_train_batch_end` 메소드를 구현하면 됩니다.

## Methods

### `add_ground_truth`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L117-L131)

```python
@abc.abstractmethod
add_ground_truth(
    logs: Optional[Dict[str, float]] = None
) -> None
```

그라운드 트루스 데이터를 `data_table`에 추가합니다.

이 메소드를 사용하여 `init_data_table` 메소드를 사용하여 초기화된 `data_table`에 유효성 검사/트레이닝 데이터를 추가하는 로직을 작성합니다.

#### 예시:

```python
for idx, data in enumerate(dataloader):
    self.data_table.add_data(idx, data)
```

이 메소드는 `on_train_begin` 또는 이와 동등한 훅에서 한 번 호출됩니다.

### `add_model_predictions`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L133-L155)

```python
@abc.abstractmethod
add_model_predictions(
    epoch: int,
    logs: Optional[Dict[str, float]] = None
) -> None
```

모델의 예측값을 `pred_table`에 추가합니다.

이 메소드를 사용하여 `init_pred_table` 메소드를 사용하여 초기화된 `pred_table`에 대한 유효성 검사/트레이닝 데이터에 대한 모델 예측을 추가하는 로직을 작성합니다.

#### 예시:

```python
# 데이터 로더가 샘플을 섞지 않는다고 가정합니다.
for idx, data in enumerate(dataloader):
    preds = model.predict(data)
    self.pred_table.add_data(
        self.data_table_ref.data[idx][0],
        self.data_table_ref.data[idx][1],
        preds,
    )
```

이 메소드는 `on_epoch_end` 또는 이와 동등한 훅에서 호출됩니다.

### `init_data_table`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L157-L166)

```python
init_data_table(
    column_names: List[str]
) -> None
```

유효성 검사 데이터를 위한 W&B Tables를 초기화합니다.

이 메소드를 `on_train_begin` 또는 이와 동등한 훅에서 호출합니다. 그 뒤에 테이블 행 또는 열 단위로 데이터를 추가합니다.

| Args |  |
| :--- | :--- |
|  `column_names` | (list) W&B Tables의 열 이름입니다. |

### `init_pred_table`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L168-L177)

```python
init_pred_table(
    column_names: List[str]
) -> None
```

모델 평가를 위한 W&B Tables를 초기화합니다.

이 메소드를 `on_epoch_end` 또는 이와 동등한 훅에서 호출합니다. 그 뒤에 테이블 행 또는 열 단위로 데이터를 추가합니다.

| Args |  |
| :--- | :--- |
|  `column_names` | (list) W&B Tables의 열 이름입니다. |

### `log_data_table`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L179-L205)

```python
log_data_table(
    name: str = "val",
    type: str = "dataset",
    table_name: str = "val_data"
) -> None
```

`data_table`을 W&B 아티팩트로 기록하고 그에 대해 `use_artifact`를 호출합니다.

이를 통해 평가 테이블은 이미 업로드된 데이터(이미지, 텍스트, 스칼라 등)의 참조를 다시 업로드하지 않고 사용할 수 있습니다.

| Args |  |
| :--- | :--- |
|  `name` | (str) 이 Artifacts에 대한 사람이 읽을 수 있는 이름입니다. UI에서 이 Artifacts를 식별하거나 use_artifact 호출에서 참조하는 방법입니다. (기본값은 'val') |
|  `type` | (str) Artifacts의 유형으로, Artifacts를 구성하고 차별화하는 데 사용됩니다. (기본값은 'dataset') |
|  `table_name` | (str) UI에 표시될 테이블의 이름입니다. (기본값은 'val_data') |

### `log_pred_table`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L207-L228)

```python
log_pred_table(
    type: str = "evaluation",
    table_name: str = "eval_data",
    aliases: Optional[List[str]] = None
) -> None
```

모델 평가를 위한 W&B Tables를 기록합니다.

테이블은 새 버전을 생성하여 여러 번 기록됩니다. 이를 사용하여 다른 간격으로 모델을 대화식으로 비교합니다.

| Args |  |
| :--- | :--- |
|  `type` | (str) Artifacts의 유형으로, Artifacts를 구성하고 차별화하는 데 사용됩니다. (기본값은 'evaluation') |
|  `table_name` | (str) UI에 표시될 테이블의 이름입니다. (기본값은 'eval_data') |
|  `aliases` | (List[str]) 예측 테이블의 에일리어스 목록입니다. |

### `set_model`

```python
set_model(
    model
)
```

### `set_params`

```python
set_params(
    params
)
```