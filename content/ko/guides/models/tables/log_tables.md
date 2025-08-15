---
title: 테이블 로그
menu:
  default:
    identifier: ko-guides-models-tables-log_tables
weight: 2
---

W&B Tables를 사용하여 표 형태의 데이터를 시각화하고 로깅하세요. W&B Table은 각 컬럼이 한 가지 타입의 데이터를 가지는 2차원 격자 데이터입니다. 각 행은 W&B [run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})에 기록된 하나 이상의 데이터 포인트를 나타냅니다. W&B Tables는 기본 및 숫자 타입은 물론 중첩된 리스트, 딕셔너리, 리치 미디어 타입도 지원합니다.

W&B Table은 W&B에서 [데이터 타입]({{< relref path="/ref/python/sdk/data-types/" lang="ko" >}}) 중 하나로, [artifact]({{< relref path="/guides/core/artifacts/" lang="ko" >}}) 오브젝트로 기록됩니다.

W&B Python SDK를 이용해 [테이블 오브젝트를 생성하고 로깅할 수 있습니다]({{< relref path="#create-and-log-a-new-table" lang="ko" >}}). 테이블 오브젝트를 생성할 때, 테이블의 컬럼과 데이터를 지정하고, [mode]({{< relref path="#table-logging-modes" lang="ko" >}})를 설정합니다. mode는 해당 테이블을 ML 실험 중 어떻게 기록하고 업데이트할지 방식을 정의합니다.

{{% alert %}}
`INCREMENTAL` 모드는 W&B Server v0.70.0 이상에서 지원됩니다.
{{% /alert %}}

## 테이블 생성 및 로깅

1. `wandb.init()`으로 새로운 run을 시작하세요. 
2. [`wandb.Table`]({{< relref path="/ref/python/sdk/data-types/table" lang="ko" >}}) 클래스를 사용해 Table 오브젝트를 생성하세요. 컬럼명은 `columns` 파라미터로, 데이터는 `data` 파라미터로 각각 지정합니다. 선택적인 `log_mode` 파라미터를 `IMMUTABLE`(기본값), `MUTABLE`, `INCREMENTAL` 세 가지 중 하나로 설정하는 것을 권장합니다. 자세한 내용은 아래 [Table Logging Modes]({{< relref path="#logging-modes" lang="ko" >}}) 섹션을 참고하세요.
3. run.log()로 테이블을 W&B에 기록하세요.

아래 예시는 `a`와 `b` 두 컬럼, 그리고 `["a1", "b1"]`, `["a2", "b2"]` 두 줄의 데이터로 테이블을 생성 및 로깅하는 방법을 보여줍니다:

```python
import wandb

# 새로운 run 시작
with wandb.init(project="table-demo") as run:

    # 두 개의 컬럼, 두 줄의 데이터로 테이블 오브젝트 생성
    my_table = wandb.Table(
        columns=["a", "b"],
        data=[["a1", "b1"], ["a2", "b2"]],
        log_mode="IMMUTABLE"
        )

    # W&B에 테이블 로깅
    run.log({"Table Name": my_table})
```

## Logging 모드

[`wandb.Table`]({{< relref path="/ref/python/sdk/data-types/table" lang="ko" >}})의 `log_mode` 파라미터는 ML 실험 중 테이블이 어떻게 기록·업데이트될지 정합니다. `log_mode`는 `IMMUTABLE`, `MUTABLE`, `INCREMENTAL` 중 하나의 값을 가집니다. 모드마다 테이블의 수정, 기록, W&B App에서 렌더링 방식에 차이가 있습니다.

아래 표는 세 가지 logging 모드와 주요 차이점, 일반적 유스 케이스를 설명합니다:

| Mode  | 정의 | 유스 케이스  | 장점  |
| ----- | --------- | ------------ | -------|
| `IMMUTABLE`   | 테이블이 W&B에 기록되면 수정할 수 없습니다. |- run 종료 시 생성된 테이블 데이터 저장 및 후속 분석                              | - run 끝에서 로그 시 오버헤드 최소<br>- 모든 행이 UI에서 렌더링됨 |
| `MUTABLE`     | 테이블을 기록한 후, 새 테이블로 덮어쓸 수 있습니다. | - 기존 테이블에 컬럼/행 추가<br>- 기존 결과에 새로운 정보 추가                        | - Table 변경 사항 누적<br>- 모든 행이 UI에서 렌더링됨                          |
| `INCREMENTAL` |  실험 중 새로운 데이터 묶음을 계속 테이블에 추가할 수 있습니다. | - 묶음(배치) 단위 행 추가<br> - 긴 러닝 시간의 트레이닝 작업<br>- 대규모 데이터셋을 배치로 처리<br>- 진행 중인 결과 모니터링 | - 트레이닝 중 실시간 UI 업데이트<br>- 증분별 데이터 탐색 가능    |

다음 섹션에서 각 모드의 코드 예시와, 언제 사용하는 것이 좋은지 안내합니다.

### MUTABLE 모드

`MUTABLE` 모드는 기존 테이블을 새 테이블로 바꾸며, 반복적이지 않은 작업에서 컬럼이나 행을 추가하고 싶을 때 유용합니다. UI에서는 모든 행과 컬럼(처음 기록 이후 추가된 것도 포함하여)이 함께 보여집니다.

{{% alert %}}
`MUTABLE` 모드에선 테이블을 기록할 때마다 테이블 오브젝트가 완전히 교체됩니다. 새 테이블로 덮어쓰는 작업은 연산적으로 비용이 크며, 큰 테이블일수록 느릴 수 있습니다.
{{% /alert %}}

아래 예시는 `MUTABLE` 모드로 테이블을 만들고 기록한 후, 컬럼을 추가하는 과정을 담았습니다. 테이블 오브젝트는 초기 데이터, confidence score 추가, 마지막 예측값 추가로 총 세 번 기록됩니다.

{{% alert %}}
아래 예시는 데이터를 불러오는 `load_eval_data()`, 예측을 하는 `model.predict()` 함수가 placeholder로 제공됩니다. 실제 사용 시 데이터와 예측용 함수로 대체하세요.
{{% /alert %}}

```python
import wandb
import numpy as np

with wandb.init(project="mutable-table-demo") as run:

    # MUTABLE 모드로 테이블 오브젝트 생성
    table = wandb.Table(columns=["input", "label", "prediction"],
                        log_mode="MUTABLE")

    # 데이터 불러오고 예측 수행
    inputs, labels = load_eval_data() # 예시 함수
    raw_preds = model.predict(inputs) # 예시 함수

    for inp, label, pred in zip(inputs, labels, raw_preds):
        table.add_data(inp, label, pred)

    # 1단계: 초기 예측값 기록
    run.log({"eval_table": table})

    # 2단계: confidence score(예. softmax 최대값) 컬럼 추가
    confidences = np.max(raw_preds, axis=1)
    table.add_column("confidence", confidences)
    run.log({"eval_table": table})

    # 3단계: 후처리된 최종 예측 컬럼 추가
    # (예: thresholding, smoothing 등)
    post_preds = (confidences > 0.7).astype(int)
    table.add_column("final_prediction", post_preds)
    run.log({"eval_table": table})
```

새 컬럼 없이 행만 배치로 점진적으로 추가하고 싶다면, [`INCREMENTAL` 모드]({{< relref path="#INCREMENTAL-mode" lang="ko" >}}) 사용을 고려해보세요.

### INCREMENTAL 모드

INCREMENTAL 모드에선 실험 수행 도중 여러 배치의 행을 테이블에 추가하며 기록합니다. 긴 시간이 필요한 작업, 업데이트 빈도가 많은 대용량 테이블에 효율적입니다. UI에서는 새 행이 기록될 때마다 테이블이 즉시 갱신되어, run이 끝나기 전에도 최신 데이터를 확인할 수 있습니다. 증분별로 시점을 이동하여 테이블을 탐색하는 것도 가능합니다.

{{% alert %}}
W&B App의 run 워크스페이스에는 최대 100개 이력(증분)만 보관합니다. 100개를 초과하면, 최신 100개만 워크스페이스에서 보여집니다.
{{% /alert %}}

아래 예시는 `INCREMENTAL` 모드로 테이블을 만들고, 트레이닝 step마다 새 행을 추가하며 로그하는 코드입니다.

{{% alert %}}
이번 예시에서는 데이터를 불러오는 `get_training_batch()`, 모델 트레이닝 `train_model_on_batch()`, 배치 예측 `predict_on_batch()` 함수가 placeholder로 제공됩니다. 실제 환경에 맞는 함수로 교체하세요.
{{% /alert %}}

```python
import wandb

with wandb.init(project="incremental-table-demo") as run:

    # INCREMENTAL 모드로 테이블 생성
    table = wandb.Table(columns=["step", "input", "label", "prediction"],
                        log_mode="INCREMENTAL")

    # 트레이닝 루프
    for step in range(get_num_batches()): # 예시 함수
        # 배치 데이터 불러오기
        inputs, labels = get_training_batch(step) # 예시 함수

        # 학습 및 예측수행
        train_model_on_batch(inputs, labels) # 예시 함수
        predictions = predict_on_batch(inputs) # 예시 함수

        # 배치 데이터를 테이블에 추가
        for input_item, label, prediction in zip(inputs, labels, predictions):
            table.add_data(step, input_item, label, prediction)

        # 증분 방식으로 테이블 로그
        run.log({"training_table": table}, step=step)
```

증분 로깅은 매번 테이블 전체를 기록하는 것(`log_mode=MUTABLE`)보다 메모리 및 연산 측면에서 효율적입니다. 단, 매우 많은 이력이 로그될 경우 일부 행이 UI에서 렌더링되지 않을 수 있습니다. 실험 중간 과정을 실시간으로 보고 싶고, 추후 모든 데이터를 분석에 사용하려면 두 개의 테이블을 쓰는 것이 좋습니다. 하나는 `INCREMENTAL`(진행 중 실시간 모니터링용), 다른 하나는 `IMMUTABLE`(최종 전체 데이터 용)로 두기입니다. 

다음은 `INCREMENTAL`와 `IMMUTABLE` 모드를 함께 사용하는 예시입니다.

```python
import wandb

with wandb.init(project="combined-logging-example") as run:

    # 트레이닝 도중 실시간 업데이트용 incremental 테이블 생성
    incr_table = wandb.Table(columns=["step", "input", "prediction", "label"],
                            log_mode="INCREMENTAL")

    # 트레이닝 루프
    for step in range(get_num_batches()):
        # 배치 처리
        inputs, labels = get_training_batch(step)
        predictions = model.predict(inputs)

        # 증분 테이블에 데이터 추가
        for inp, pred, label in zip(inputs, predictions, labels):
            incr_table.add_data(step, inp, pred, label)

        # 증분 업데이트(최종 테이블과 구분 위해 -incr 접미사)
        run.log({"table-incr": incr_table}, step=step)

    # 트레이닝 종료 시, 증분 테이블 전체 데이터를 IMMUTABLE 테이블로 복제
    final_table = wandb.Table(columns=incr_table.columns, data=incr_table.data, log_mode="IMMUTABLE")
    run.log({"table": final_table})
```

위 예시에서, `incr_table`은 트레이닝 중(`log_mode="INCREMENTAL"`)에 실시간으로 로그됩니다. 새 데이터가 처리될 때마다 테이블 갱신 및 UI에서 확인이 가능합니다. 트레이닝 종료 시 증분 테이블의 모든 데이터로 immutable 테이블(`final_table`)을 만듭니다. 이 immutable 테이블은 전체 데이터셋 보존 및 분석, W&B App에서의 전체 행 탐색에 활용할 수 있습니다.

## 예시

### 평가 결과를 MUTABLE로 확장하기

```python
import wandb
import numpy as np

with wandb.init(project="mutable-logging") as run:

    # 1단계: 초기 예측값 로깅
    table = wandb.Table(columns=["input", "label", "prediction"], log_mode="MUTABLE")
    inputs, labels = load_eval_data()
    raw_preds = model.predict(inputs)

    for inp, label, pred in zip(inputs, labels, raw_preds):
        table.add_data(inp, label, pred)

    run.log({"eval_table": table})  # 원본 예측값 기록

    # 2단계: confidence score (예: softmax 최대값) 추가
    confidences = np.max(raw_preds, axis=1)
    table.add_column("confidence", confidences)
    run.log({"eval_table": table})

    # 3단계: 후처리 예측값 추가
    # (예: thresholding, smoothing 등)
    post_preds = (confidences > 0.7).astype(int)
    table.add_column("final_prediction", post_preds)
    run.log({"eval_table": table})
```

### INCREMENTAL 테이블로 run 이어서 기록하기

run을 이어받아 incremental 테이블에 계속 로그할 수 있습니다:

```python
# run 시작 또는 재개
resumed_run = wandb.init(project="resume-incremental", id="your-run-id", resume="must")

# 증분 테이블 생성; 이전에 기록된 테이블 데이터를 미리 채울 필요가 없음
# 새 이력(증분)은 계속 Table artifact에 추가됨
table = wandb.Table(columns=["step", "metric"], log_mode="INCREMENTAL")

# 계속 로깅
for step in range(resume_step, final_step):
    metric = compute_metric(step)
    table.add_data(step, metric)
    resumed_run.log({"metrics": table}, step=step)

resumed_run.finish()
```

{{% alert %}}
`wandb.Run.define_metric("<table_key>", summary="none")` 또는 `wandb.Run.define_metric("*", summary="none")`을 사용해 incremental 테이블로 쓰는 키의 summary를 끄면, 새 테이블로 increments가 기록됩니다.
{{% /alert %}}


### INCREMENTAL 배치 트레이닝으로 실험하기

```python

with wandb.init(project="batch-training-incremental") as run:

    # 증분 방식 테이블 생성
    table = wandb.Table(columns=["step", "input", "label", "prediction"], log_mode="INCREMENTAL")

    # 트레이닝 루프(예시)
    for step in range(get_num_batches()):
        # 배치 데이터 불러오기
        inputs, labels = get_training_batch(step)

        # 배치 학습
        train_model_on_batch(inputs, labels)

        # 모델 추론
        predictions = predict_on_batch(inputs)

        # 테이블에 데이터 추가
        for input_item, label, prediction in zip(inputs, labels, predictions):
            table.add_data(step, input_item, label, prediction)

        # 현재 테이블 상태 증분 로그
        run.log({"training_table": table}, step=step)
```