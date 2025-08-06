---
title: 단일 프로세스에서 여러 run을 생성하고 관리하기
description: W&B의 reinit 기능을 사용하여 하나의 Python 프로세스에서 여러 run 을 관리하세요
menu:
  default:
    identifier: ko-guides-models-track-runs-multiple-runs-per-process
    parent: what-are-runs
---

하나의 Python 프로세스에서 여러 개의 run 을 관리할 수 있습니다. 이 기능은 주요 프로세스를 활성 상태로 유지하면서, 짧게 실행되는 보조 프로세스를 만들어 여러 서브 태스크를 처리해야 하는 워크플로우에 유용합니다. 주요 유스 케이스 예시는 다음과 같습니다.

- 스크립트 전체에서 하나의 "주" run 을 계속 활성화하고, 평가나 서브 태스크를 위해 짧게 실행되는 "보조" run 들을 생성
- 하나의 파일에서 여러 서브 실험을 오케스트레이션  
- 하나의 "메인" 프로세스에서 여러 run 에 로그를 남기며, 각각의 run 이 서로 다른 태스크나 시간 구간을 나타내는 경우

기본적으로 W&B는 하나의 Python 프로세스에서 `wandb.init()`을 호출할 때마다 단 한 개의 run 만 활성화되어 있음을 전제로 합니다. 만약 `wandb.init()`을 다시 호출하면, 설정에 따라 이전 run 을 종료하고 새로운 run 을 시작하거나 같은 run 을 반환하게 됩니다. 이 가이드에서는 `reinit` 옵션을 사용하여 `wandb.init()`의 동작을 변경하고, 하나의 Python 프로세스에서 여러 run 을 활성화하는 방법을 설명합니다.

{{% alert title="Requirements" %}}
하나의 Python 프로세스에서 여러 run 을 관리하려면, W&B Python SDK `v0.19.10` 이상 버전이 필요합니다.
{{% /alert  %}}

## `reinit` 옵션

`reinit` 파라미터로 여러 번의 `wandb.init()` 호출 시 W&B의 동작 방식을 설정할 수 있습니다. 아래 표는 각 인수와 그 효과를 설명합니다:

| | 설명 | run 생성 여부 | 대표 유스 케이스 |
|----------------|----------------|----------------| -----------------|
| `create_new` | 기존에 활성화된 run 을 종료하지 않고, `wandb.init()`으로 새로운 run 을 생성합니다. W&B는 전역 `wandb.Run`을 자동으로 신규 run 으로 변경하지 않습니다. 각 run 오브젝트를 직접 관리해야 합니다. 자세한 활용 방법은 아래 [한 프로세스에서 여러 run 사용 예시]({{< relref path="multiple-runs-per-process/#example-multiple-runs-in-one-process" lang="ko" >}})를 참고하세요.  | 네 | 동시 프로세스 관리가 필요한 경우 적합합니다. 예를 들어, "주" run 을 계속 유지하면서 "보조" run 을 시작하거나 종료할 때 사용합니다.|
| `finish_previous` | 새로운 run 을 생성하기 전, 활성화된 모든 run 을 `run.finish()`로 종료합니다. 노트북이 아닌 환경에서 기본 동작입니다. | 네 | 순차적으로 실행되는 서브 프로세스를 각각 개별 run 으로 나누고자 할 때 적합합니다. |
| `return_previous` | 가장 최근의 종료되지 않은 run 을 반환합니다. 노트북 환경에서 기본 동작입니다. | 아니요 | |

{{% alert  %}}
W&B는 Hugging Face Trainer, Keras 콜백, PyTorch Lightning 등 단일 전역 run 을 전제로 설계된 [W&B 인테그레이션]({{< relref path="/guides/integrations/" lang="ko" >}})에서 `create_new` 모드를 지원하지 않습니다. 이러한 인테그레이션을 사용하는 경우, 각 서브 실험을 별도의 프로세스에서 실행하는 것이 권장됩니다.
{{% /alert %}}

## `reinit` 지정 방법



- `wandb.init()`에 직접 `reinit` 인수를 넘기는 방법:
   ```python
   import wandb
   wandb.init(reinit="<create_new|finish_previous|return_previous>")
   ```
- `wandb.Settings` 오브젝트를 생성해 `settings` 파라미터로 전달하는 방법. `reinit` 값을 `Settings` 오브젝트에서 지정할 수 있습니다:

   ```python
   import wandb
   wandb.init(settings=wandb.Settings(reinit="<create_new|finish_previous|return_previous>"))
   ```

- `wandb.setup()` 으로 현재 프로세스 내 모든 run 에 글로벌하게 `reinit` 옵션을 지정할 수 있습니다. 한 번만 지정하고 이후의 모든 `wandb.init()` 호출에 적용하려는 경우 유용합니다.

   ```python
   import wandb
   wandb.setup(wandb.Settings(reinit="<create_new|finish_previous|return_previous>"))
   ```

- 환경 변수 `WANDB_REINIT` 에 원하는 값을 설정하여 적용할 수도 있습니다. 환경 변수를 지정하면 `wandb.init()` 호출에 해당 `reinit` 옵션이 적용됩니다.

   ```bash
   export WANDB_REINIT="<create_new|finish_previous|return_previous>"
   ```

아래 코드조각은 `wandb.init()`을 호출할 때마다 새로운 run 이 생성되도록 높은 수준에서 설정하는 예시입니다:

```python
import wandb

wandb.setup(wandb.Settings(reinit="create_new"))

with wandb.init() as experiment_results_run:
    # 이 run 은 각 실험 결과를 기록하는 데 사용됩니다.
    # 'parent' run 으로 볼 수 있으며, 여러 실험의 결과를 모아서 관리합니다.
      with wandb.init() as run:
         # do_experiment() 함수는 상세 메트릭을 해당 run 에 기록하고,
         # 별도로 관리하고 싶은 결과 메트릭을 반환합니다.
         experiment_results = do_experiment(run)

         # 각 실험 후, 결과를 parent run 에 기록합니다.
         # parent run 의 차트 각 포인트는 한 실험의 결과를 의미합니다.
         experiment_results_run.log(experiment_results)
```

## 예시: 동시 프로세스 관리

스크립트가 실행되는 내내 열려 있는 주요 프로세스를 하나 만들고, 필요에 따라 짧게 생성 및 종료되는 보조 프로세스를 여러 번 만들고 싶다고 가정해 봅시다. 예를 들어, 주요 run 에서는 모델 트레이닝을 수행하면서, 평가 등 개별 작업은 별도의 run 에서 진행하는 패턴입니다.

이런 경우엔 `reinit="create_new"` 옵션을 이용해 여러 run 을 초기화하면 됩니다. 여기서 예시로, "Run A"는 전체 스크립트 동안 열려 있는 주요 프로세스이며, "Run B1", "Run B2" 등은 평가 등의 작업을 위한 짧게 실행되는 보조 run 으로 가정해보겠습니다.

전반적인 워크플로우는 아래와 같을 수 있습니다:

1. 주요 프로세스 Run A를 `wandb.init()`으로 초기화하고 트레이닝 메트릭 기록  
2. Run B1(`wandb.init()`), 데이터 기록 후 종료  
3. Run A에 추가 데이터 기록  
4. Run B2 초기화, 데이터 기록 후 종료  
5. Run A에 계속 기록  
6. 마지막에 Run A 종료

아래 Python 예시는 위 워크플로우를 보여줍니다:

```python
import wandb

def train(name: str) -> None:
    """각각의 W&B run 에서 하나의 트레이닝 이터레이션을 실행합니다.

    'with wandb.init()' 블록에서 `reinit="create_new"`를 사용하면,
    이미 다른 run(예: 메인 추적 run)이 활성 상태여도 이 트레이닝 서브-run 을 생성할 수 있습니다.
    """
    with wandb.init(
        project="my_project",
        name=name,
        reinit="create_new"
    ) as run:
        # 실제 스크립트라면, 이 블록 내에서 트레이닝 스텝을 실행합니다.
        run.log({"train_loss": 0.42})  # 실제 메트릭으로 대체하세요

def evaluate_loss_accuracy() -> (float, float):
    """현재 모델의 loss 및 accuracy를 반환합니다.
    
    실제 평가 로직으로 내용을 채우세요.
    """
    return 0.27, 0.91  # 예시 메트릭 값

# 여러 번의 train/eval 단계 동안 계속 유지되는 'primary' run 을 생성합니다.
with wandb.init(
    project="my_project",
    name="tracking_run",
    reinit="create_new"
) as tracking_run:
    # 1) 'training_1'이라는 서브-run 에서 트레이닝
    train("training_1")
    loss, accuracy = evaluate_loss_accuracy()
    tracking_run.log({"eval_loss": loss, "eval_accuracy": accuracy})

    # 2) 'training_2'라는 서브-run 에서 추가 트레이닝
    train("training_2")
    loss, accuracy = evaluate_loss_accuracy()
    tracking_run.log({"eval_loss": loss, "eval_accuracy": accuracy})
    
    # 'tracking_run'은 이 'with' 블록이 끝날 때 자동으로 종료됩니다.
```

위 예시에서 주목해야 할 점은 다음과 같습니다:

1. `reinit="create_new"`를 지정하면 `wandb.init()`을 호출할 때마다 새로운 run 이 생성됩니다.
2. 각 run 을 변수에 저장해 직접 참조해야 합니다. `wandb.run`은 `reinit="create_new"`로 생성된 run 으로 자동 전환되지 않으므로, `run_a`, `run_b1` 등과 같은 변수에 새 run 을 저장하고, `.log()`나 `.finish()`를 해당 오브젝트에서 호출해야 합니다.
3. 주요 run 을 열린 채로 두면서 , 원할 때 서브 run 들을 종료할 수 있습니다.
4. 각 run 에 대한 로그가 끝나면 `run.finish()`로 명확하게 종료하세요. 그래야 모든 데이터가 업로드되고 run 이 정상적으로 종료됩니다.