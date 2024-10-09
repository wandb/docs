---
title: Pytorch torchtune
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/torchtune/torchtune_and_wandb.ipynb"></CTAButtons>

[torchtune](https://pytorch.org/torchtune/stable/index.html)는 대형 언어 모델(LLM)의 작성, 미세 조정 및 실험 프로세스를 간소화하기 위해 설계된 PyTorch 기반 라이브러리입니다. 또한, torchtune은 [W&B로 로그하는 것](https://pytorch.org/torchtune/stable/deep_dives/wandb_logging.html)에 대한 지원을 내장하고 있어 트레이닝 프로세스의 추적 및 시각화를 강화합니다.

![](/images/integrations/torchtune_dashboard.png)

[Fine-tuning Mistral 7B using torchtune](https://wandb.ai/capecape/torchtune-mistral/reports/torchtune-The-new-PyTorch-LLM-fine-tuning-library---Vmlldzo3NTUwNjM0)에 대한 W&B 블로그 글을 확인하세요.

## 손끝에서 제어하는 W&B 로그

<Tabs
  defaultValue="config"
  values={[
    {label: 'Recipe\'s config', value: 'config'},
    {label: 'Command line', value: 'cli'},
  ]}>
  <TabItem value="cli">

런치 시 커맨드라인 인수 재정의:

```bash
tune run lora_finetune_single_device --config llama3/8B_lora_single_device \
  metric_logger._component_=torchtune.utils.metric_logging.WandBLogger \
  metric_logger.project="llama3_lora" \
  log_every_n_steps=5
```

  </TabItem>
  <TabItem value="config">

Recipe의 config에서 W&B 로그 활성화
```yaml
# inside llama3/8B_lora_single_device.yaml
metric_logger:
  _component_: torchtune.utils.metric_logging.WandBLogger
  project: llama3_lora
log_every_n_steps: 5
```

  </TabItem>
</Tabs>

## W&B 메트릭 로거 사용하기

`metric_logger` 섹션을 수정하여 Recipe의 config 파일에서 W&B 로그를 활성화하세요. `_component_`를 `torchtune.utils.metric_logging.WandBLogger` 클래스로 변경하면 됩니다. `project` 이름을 전달하고 `log_every_n_steps`를 전달하여 로그 행동을 커스터마이즈할 수 있습니다.

또한, [wandb.init](../../ref/python/init.md) 메소드에 전달할 수 있는 다른 `kwargs`도 전달할 수 있습니다. 팀 작업을 진행 중이라면 `WandBLogger` 클래스에 `entity` 인수를 전달하여 팀 이름을 지정할 수 있습니다.

<Tabs
  defaultValue="config"
  values={[
    {label: 'Recipe\'s Config', value: 'config'},
    {label: 'Command Line', value: 'cli'},
  ]}>
  <TabItem value="cli">

```shell
tune run lora_finetune_single_device --config llama3/8B_lora_single_device \
  metric_logger._component_=torchtune.utils.metric_logging.WandBLogger \
  metric_logger.project="llama3_lora" \
  metric_logger.entity="my_project" \
  metric_logger.job_type="lora_finetune_single_device" \
  metric_logger.group="my_awesome_experiments" \
  log_every_n_steps=5
```
  
  </TabItem>
  <TabItem value="config">

```yaml
# inside llama3/8B_lora_single_device.yaml
metric_logger:
  _component_: torchtune.utils.metric_logging.WandBLogger
  project: llama3_lora
  entity: my_project
  job_type: lora_finetune_single_device
  group: my_awesome_experiments
log_every_n_steps: 5
```

  </TabItem>
</Tabs>

## 무엇이 로그되나요?

위의 명령을 실행한 후, W&B 대시보드를 탐색하여 로그된 메트릭을 확인할 수 있습니다. 기본적으로 W&B는 config 파일과 런치 재정의로부터 모든 하이퍼파라미터를 가져옵니다.

W&B는 **개요** 탭에서 해결된 config를 캡처해 줍니다. 또한, W&B는 config를 [파일 탭](https://wandb.ai/capecape/torchtune/runs/joyknwwa/files)에 YAML 형식으로 저장합니다.

![](/images/integrations/torchtune_config.png)

### 로그된 메트릭

각 Recipe는 자체적인 트레이닝 루프를 가지므로, 각 Recipe를 개별적으로 확인하여 어떤 메트릭이 로그되는지 확인하세요. 기본적으로 로그되는 메트릭은 다음과 같습니다:

| Metric | 설명 |
| --- | --- |
| `loss` | 모델의 손실 |
| `lr` | 학습률 |
| `tokens_per_second` | 모델의 초당 토큰 수 |
| `grad_norm` | 모델의 그레이디언트 노름 |
| `global_step` | 트레이닝 루프의 현재 스텝에 해당합니다. 그레이디언트 누적을 고려하며, 기본적으로 옵티마이저 스텝이 수행될 때마다 모델이 업데이트되고, 그레이디언트가 누적되며, `gradient_accumulation_steps`마다 모델이 업데이트됩니다. |

:::info
`global_step`은 트레이닝 스텝 수와 동일하지 않습니다. 트레이닝 루프의 현재 스텝에 해당합니다. 그레이디언트 누적을 고려하며, 기본적으로 옵티마이저 스텝이 수행될 때마다 `global_step`이 1씩 증가합니다. 예를 들어, 데이터로더에 10개의 배치가 있고, 그레이디언트 누적 스텝이 2이며 3 에포크를 실행할 경우, 옵티마이저는 15회 스텝을 수행하며, 이 경우 `global_step`은 1에서 15까지 범위를 가집니다.
:::

torchtune의 간소화된 디자인은 사용자 정의 메트릭을 쉽게 추가하거나 기존 메트릭을 수정할 수 있게 해 줍니다. 해당하는 [recipe 파일](https://github.com/pytorch/torchtune/tree/main/recipes)을 수정하여 특정 에포크를 총 에포크 수의 비율로 로그하거나 등 다양한 설정을 추가할 수 있습니다:

```python
# inside `train.py` function in the recipe file
self._metric_logger.log_dict(
    {"current_epoch": self.epochs * self.global_step / self._steps_per_epoch},
    step=self.global_step,
)
```

:::info
이 라이브러리는 빠르게 발전하는 중이며, 현재의 메트릭은 변경될 수 있습니다. 사용자 정의 메트릭을 추가하려면 Recipe를 수정하고 해당하는 `self._metric_logger.*` 함수를 호출해야 합니다.
:::

## 체크포인트 저장 및 로드

torchtune 라이브러리는 다양한 [체크포인트 형식](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html)을 지원합니다. 사용하는 모델의 출처에 따라 적절한 [체크포인터 클래스](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html)로 전환해야 합니다.

모델 체크포인트를 [W&B Artifacts](../artifacts/intro.md)로 저장하려면, 해당 Recipe 내부의 `save_checkpoint` 함수를 재정의하는 가장 간단한 방법이 있습니다.

모델 체크포인트를 W&B Artifacts로 저장하기 위해 `save_checkpoint` 함수를 재정의하는 방법의 예는 다음과 같습니다:

```python
def save_checkpoint(self, epoch: int) -> None:
    ...
    ## W&B에 체크포인트 저장하기
    ## 체크포인터 클래스에 따라 파일 이름이 다르게 지정됩니다
    ## full_finetune 경우의 예시입니다
    checkpoint_file = Path.joinpath(
        self._checkpointer._output_dir, f"torchtune_model_{epoch}"
    ).with_suffix(".pt")
    wandb_artifact = wandb.Artifact(
        name=f"torchtune_model_{epoch}",
        type="model",
        # 모델 체크포인트의 설명
        description="Model checkpoint",
        # 원하는 메타데이터를 사전 형식으로 추가할 수 있습니다
        metadata={
            utils.SEED_KEY: self.seed,
            utils.EPOCHS_KEY: self.epochs_run,
            utils.TOTAL_EPOCHS_KEY: self.total_epochs,
            utils.MAX_STEPS_KEY: self.max_steps_per_epoch,
        }
    )
    wandb_artifact.add_file(checkpoint_file)
    wandb.log_artifact(wandb_artifact)
```