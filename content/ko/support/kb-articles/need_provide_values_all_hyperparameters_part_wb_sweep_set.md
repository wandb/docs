---
title: Do I need to provide values for all hyperparameters as part of the W&B Sweep.
  Can I set defaults?
menu:
  support:
    identifier: ko-support-kb-articles-need_provide_values_all_hyperparameters_part_wb_sweep_set
support:
- sweeps
toc_hide: true
type: docs
url: /ko/support/:filename
---

`wandb.config`를 사용하여 스윕 구성에서 하이퍼파라미터 이름과 값에 엑세스하세요. `wandb.config`는 사전처럼 작동합니다.

스윕 외부의 run의 경우, `wandb.init`에서 `config` 인수에 사전을 전달하여 `wandb.config` 값을 설정합니다. 스윕에서 `wandb.init`에 제공된 모든 설정은 스윕이 재정의할 수 있는 기본값으로 사용됩니다.

명시적인 행동을 위해 `config.setdefaults`를 사용하세요. 다음 코드 조각은 두 가지 방법을 모두 보여줍니다.

{{< tabpane text=true >}}
{{% tab "wandb.init()" %}}
```python
# Set default values for hyperparameters
config_defaults = {"lr": 0.1, "batch_size": 256}

# Start a run and provide defaults
# that a sweep can override
with wandb.init(config=config_defaults) as run:
    # Add training code here
    ...
```
{{% /tab %}}
{{% tab "config.setdefaults()" %}}
```python
# Set default values for hyperparameters
config_defaults = {"lr": 0.1, "batch_size": 256}

# Start a run
with wandb.init() as run:
    # Update any values not set by the sweep
    run.config.setdefaults(config_defaults)

    # Add training code here
```
{{% /tab %}}
{{< /tabpane >}}
