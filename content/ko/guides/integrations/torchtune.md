---
title: Pytorch torchtune
menu:
  default:
    identifier: ko-guides-integrations-torchtune
    parent: integrations
weight: 320
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/torchtune/torchtune_and_wandb.ipynb" >}}

[torchtune](https://pytorch.org/torchtune/stable/index.html)는 PyTorch 기반 라이브러리로, 대형 언어 모델(LLM)의 작성, 파인튜닝, 실험 과정을 간단하게 만들어줍니다. 또한, torchtune은 [W&B를 활용한 로깅](https://pytorch.org/torchtune/stable/deep_dives/wandb_logging.html)을 기본적으로 지원하여, 트레이닝 과정의 추적과 시각화를 쉽게 할 수 있습니다.

{{< img src="/images/integrations/torchtune_dashboard.png" alt="TorchTune training dashboard" >}}

[Fine-tuning Mistral 7B using torchtune](https://wandb.ai/capecape/torchtune-mistral/reports/torchtune-The-new-PyTorch-LLM-fine-tuning-library---Vmlldzo3NTUwNjM0)에 대한 W&B 블로그 포스트도 참고해보세요.

## 손쉽게 시작하는 W&B 로깅

{{< tabpane text=true >}}
{{% tab header="Command line" value="cli" %}}

런 시점에서 커맨드라인 인수를 오버라이드하기:

```bash
tune run lora_finetune_single_device --config llama3/8B_lora_single_device \
  metric_logger._component_=torchtune.utils.metric_logging.WandBLogger \
  metric_logger.project="llama3_lora" \
  log_every_n_steps=5
```

{{% /tab %}}
{{% tab header="Recipe's config" value="config" %}}

레시피의 설정 파일에서 W&B 로깅 활성화:

```yaml
# llama3/8B_lora_single_device.yaml 파일 내부
metric_logger:
  _component_: torchtune.utils.metric_logging.WandBLogger
  project: llama3_lora
log_every_n_steps: 5
```

{{% /tab %}}
{{< /tabpane >}}

## W&B metric logger 사용하기

레시피의 설정 파일에서 `metric_logger` 섹션을 수정하여 W&B 로깅을 활성화할 수 있습니다. `_component_`를 `torchtune.utils.metric_logging.WandBLogger` 클래스로 변경하세요. 또한 `project` 이름과 `log_every_n_steps`를 지정하여 로깅 행동을 원하는 대로 설정할 수 있습니다.

그리고 [wandb.init()]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}}) 메소드에 전달하는 것처럼 다른 키워드 인수(`kwargs`)도 함께 전달할 수 있습니다. 예를 들어, 팀 작업 중이라면 `entity` 인수를 `WandBLogger` 클래스에 넘겨 팀 이름을 지정할 수 있습니다.

{{< tabpane text=true >}}
{{% tab header="Recipe's Config" value="config" %}}

```yaml
# llama3/8B_lora_single_device.yaml 파일 내부
metric_logger:
  _component_: torchtune.utils.metric_logging.WandBLogger
  project: llama3_lora
  entity: my_project
  job_type: lora_finetune_single_device
  group: my_awesome_experiments
log_every_n_steps: 5
```

{{% /tab %}}

{{% tab header="Command Line" value="cli" %}}

```shell
tune run lora_finetune_single_device --config llama3/8B_lora_single_device \
  metric_logger._component_=torchtune.utils.metric_logging.WandBLogger \
  metric_logger.project="llama3_lora" \
  metric_logger.entity="my_project" \
  metric_logger.job_type="lora_finetune_single_device" \
  metric_logger.group="my_awesome_experiments" \
  log_every_n_steps=5
```

{{% /tab %}}
{{< /tabpane >}}

## 무엇이 기록되나요?

W&B 대시보드에서 기록된 메트릭을 탐색할 수 있습니다. 기본적으로, W&B는 설정 파일과 런 오버라이드에서 모든 하이퍼파라미터를 기록합니다.

W&B는 최종적으로 적용된 설정을 **Overview** 탭에 보여줍니다. YAML 형식의 설정 파일은 [Files 탭](https://wandb.ai/capecape/torchtune/runs/joyknwwa/files)에도 저장됩니다.

{{< img src="/images/integrations/torchtune_config.png" alt="TorchTune configuration" >}}

### 기록되는 메트릭

각 레시피는 고유의 트레이닝 루프를 가집니다. 각각의 레시피에서 기록되는 메트릭을 확인하세요. 기본으로 포함되는 항목은 다음과 같습니다:

| Metric | 설명 |
| --- | --- |
| `loss` | 모델의 손실(loss) |
| `lr` | 러닝레이트 |
| `tokens_per_second` | 초당 처리된 토큰 수 |
| `grad_norm` | 모델의 그레이디언트 노름(norm) |
| `global_step` | 트레이닝 루프의 현재 스텝. 그레이디언트 어큐뮬레이션을 반영하며, 옵티마이저 스텝이 실행될 때마다 모델이 업데이트되고 그레이디언트가 누적되어, `gradient_accumulation_steps`마다 한 번씩 모델이 업데이트됩니다. |

{{% alert %}}
`global_step`은 트레이닝 스텝 수와 동일하지 않습니다. 트레이닝 루프 내의 현재 스텝에 대응하며, 그레이디언트 어큐뮬레이션을 고려합니다. 기본적으로 옵티마이저 스텝이 한 번 실행될 때마다 `global_step`이 1씩 증가합니다. 예를 들어, 데이터로더에 10개 배치가 있고, 그레이디언트 어큐뮬레이션 스텝이 2이고, 3 에포크 동안 실행하면, 옵티마이저는 총 15번 스텝을 밟게 되며, 이 경우 `global_step`은 1부터 15까지의 값을 가집니다.
{{% /alert %}}

torchtune의 간결한 설계 덕분에 커스텀 메트릭을 쉽게 추가하거나 기존 메트릭을 수정할 수 있습니다. 단순하게 [레시피 파일](https://github.com/pytorch/torchtune/tree/main/recipes)을 수정하면 되며, 예를 들어 에포크 진행률을 백분율로 나타내는 `current_epoch`를 추가하려면 아래와 같이 구현할 수 있습니다:

```python
# 레시피 파일의 `train.py` 함수 내부
self._metric_logger.log_dict(
    {"current_epoch": self.epochs * self.global_step / self._steps_per_epoch},
    step=self.global_step,
)
```

{{% alert %}}
이 라이브러리는 빠르게 발전하고 있으므로, 현재 제공되는 메트릭은 변경될 수 있습니다. 커스텀 메트릭을 추가하고 싶다면, 레시피 파일을 수정하고 적절한 `self._metric_logger.*` 함수를 호출하면 됩니다.
{{% /alert %}}

## 체크포인트 저장 및 불러오기

torchtune 라이브러리는 다양한 [체크포인트 포맷](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html)을 지원합니다. 사용하는 모델에 맞는 [checkpointer class](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html)로 전환해 주세요.

모델 체크포인트를 [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})에 저장하려면, 해당 레시피의 `save_checkpoint` 함수를 오버라이드하는 것이 가장 간단한 방법입니다.

아래는 모델 체크포인트를 W&B Artifacts에 저장하는 `save_checkpoint` 함수 오버라이드 예시입니다.

```python
def save_checkpoint(self, epoch: int) -> None:
    ...
    ## 체크포인트를 W&B에 저장해봅시다
    ## Checkpointer Class에 따라 파일 이름이 달라질 수 있습니다
    ## 아래는 full_finetune 케이스 예시입니다
    checkpoint_file = Path.joinpath(
        self._checkpointer._output_dir, f"torchtune_model_{epoch}"
    ).with_suffix(".pt")
    wandb_artifact = wandb.Artifact(
        name=f"torchtune_model_{epoch}",
        type="model",
        # 모델 체크포인트에 대한 설명
        description="Model checkpoint",
        # 원하는 메타데이터를 dict 형태로 추가 가능
        metadata={
            utils.SEED_KEY: self.seed,
            utils.EPOCHS_KEY: self.epochs_run,
            utils.TOTAL_EPOCHS_KEY: self.total_epochs,
            utils.MAX_STEPS_KEY: self.max_steps_per_epoch,
        },
    )
    wandb_artifact.add_file(checkpoint_file)
    wandb.log_artifact(wandb_artifact)
```
