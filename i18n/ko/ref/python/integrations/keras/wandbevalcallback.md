
# WandbEvalCallback

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/integration/keras/callbacks/tables_builder.py#L10-L226' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


모델 예측값 시각화를 위한 Keras 콜백을 구축하기 위한 추상 기본 클래스입니다.

```python
WandbEvalCallback(
    data_table_columns: List[str],
    pred_table_columns: List[str],
    *args,
    **kwargs
) -> None
```

분류, 오브젝트 검출, 분할 등의 작업을 위해 `on_epoch_end`에 시각화 콜백을 구축할 수 있습니다.
이를 사용하려면, 이 기본 콜백 클래스에서 상속받아 `add_ground_truth` 및 `add_model_prediction` 메소드를 구현하십시오.

기본 클래스는 다음을 처리합니다:

- 그라운드 트루스와 예측값을 로깅하기 위해 `data_table`과 `pred_table`을 초기화합니다.
- `data_table`에 업로드된 데이터는 `pred_table`의 참조로 사용됩니다. 이는 메모리 사용량을 줄이기 위함입니다. `data_table_ref`는 참조된 데이터에 엑세스하기 위해 사용할 수 있는 리스트입니다.
  아래 예제를 확인하여 그 방법을 확인하세요.
- W&B Artifacts로 테이블을 W&B에 로그합니다.
- 새로운 `pred_table`은 에일리어스와 함께 새로운 버전으로 로그됩니다.

#### 예제:

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

더 세밀한 제어를 원한다면, `on_train_begin` 및 `on_epoch_end` 메소드를 오버라이드할 수 있습니다. N 배치 후 샘플을 로그하려면 `on_train_batch_end` 메소드를 구현할 수 있습니다.

## 메소드

### `add_ground_truth`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/integration/keras/callbacks/tables_builder.py#L117-L131)

```python
@abc.abstractmethod
add_ground_truth(
    logs: Optional[Dict[str, float]] = None
) -> None
```

`data_table`에 그라운드 트루스 데이터를 추가합니다.

이 메소드를 사용하여 `init_data_table` 메소드를 사용하여 초기화된 `data_table`에 검증/트레이닝 데이터를 추가하기 위한 로직을 작성하십시오.

#### 예제:

```python
for idx, data in enumerate(dataloader):
    self.data_table.add_data(idx, data)
```

이 메소드는 `on_train_begin` 또는 동등한 훅에서 한 번 호출됩니다.

### `add_model_predictions`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/integration/keras/callbacks/tables_builder.py#L133-L153)

```python
@abc.abstractmethod
add_model_predictions(
    epoch: int,
    logs: Optional[Dict[str, float]] = None
) -> None
```

`pred_table`에 모델의 예측값을 추가합니다.

이 메소드를 사용하여 `init_pred_table` 메소드를 사용하여 초기화된 `pred_table`에 검증/트레이닝 데이터에 대한 모델 예측값을 추가하기 위한 로직을 작성하십시오.

#### 예제:

```python
# 데이터로더가 샘플을 섞지 않는다고 가정합니다.
for idx, data in enumerate(dataloader):
    preds = model.predict(data)
    self.pred_table.add_data(
        self.data_table_ref.data[idx][0], self.data_table_ref.data[idx][1], preds
    )
```

이 메소드는 `on_epoch_end` 또는 동등한 훅에서 호출됩니다.

### `init_data_table`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/integration/keras/callbacks/tables_builder.py#L155-L164)

```python
init_data_table(
    column_names: List[str]
) -> None
```

검증 데이터에 대한 W&B 테이블을 초기화합니다.

이 메소드는 `on_train_begin` 또는 동등한 훅에서 호출됩니다. 이는 테이블에 데이터를 행 또는 열 방식으로 추가하는 것을 따릅니다.

| Args |  |
| :--- | :--- |
|  `column_names` |  (list) W&B 테이블의 열 이름입니다. |

### `init_pred_table`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/integration/keras/callbacks/tables_builder.py#L166-L175)

```python
init_pred_table(
    column_names: List[str]
) -> None
```

모델 평가를 위한 W&B 테이블을 초기화합니다.

이 메소드는 `on_epoch_end` 또는 동등한 훅에서 호출됩니다. 이는 테이블에 데이터를 행 또는 열 방식으로 추가하는 것을 따릅니다.

| Args |  |
| :--- | :--- |
|  `column_names` |  (list) W&B 테이블의 열 이름입니다. |

### `log_data_table`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/integration/keras/callbacks/tables_builder.py#L177-L203)

```python
log_data_table(
    name: str = "val",
    type: str = "dataset",
    table_name: str = "val_data"
) -> None
```

`data_table`을 W&B 아티팩트로 로그하고 이에 대해 `use_artifact`를 호출합니다.

이를 통해 평가 테이블은 이미 업로드된 데이터(이미지, 텍스트, 스칼라 등)의 참조를 사용할 수 있게 되며, 데이터를 다시 업로드할 필요가 없습니다.

| Args |  |
| :--- | :--- |
|  `name` |  (str) 이 아티팩트에 대한 사람이 읽을 수 있는 이름으로, UI에서 이 아티팩트를 식별하거나 use_artifact 호출에서 참조하는 데 사용됩니다. (기본값은 'val') |
|  `type` |  (str) 아티팩트의 유형으로, 아티팩트를 조직하고 구별하는 데 사용됩니다. (기본값은 'dataset') |
|  `table_name` |  (str) UI에 표시될 테이블의 이름입니다. (기본값은 'val_data'). |

### `log_pred_table`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/integration/keras/callbacks/tables_builder.py#L205-L226)

```python
log_pred_table(
    type: str = "evaluation",
    table_name: str = "eval_data",
    aliases: Optional[List[str]] = None
) -> None
```

모델 평가를 위한 W&B 테이블을 로그합니다.

테이블은 여러 번 로그되어 새로운 버전을 생성합니다. 이를 사용하여 서로 다른 간격에서 모델을 대화식으로 비교할 수 있습니다.

| Args |  |
| :--- | :--- |
|  `type` |  (str) 아티팩트의 유형으로, 아티팩트를 조직하고 구별하는 데 사용됩니다. (기본값은 'evaluation') |
|  `table_name` |  (str) UI에 표시될 테이블의 이름입니다. (기본값은 'eval_data') |
|  `aliases` |  (List[str]) 예측 테이블의 에일리어스 목록입니다. |

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