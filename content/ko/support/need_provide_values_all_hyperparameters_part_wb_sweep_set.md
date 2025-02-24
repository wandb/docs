---
title: Do I need to provide values for all hyperparameters as part of the W&B Sweep.
  Can I set defaults?
menu:
  support:
    identifier: ko-support-need_provide_values_all_hyperparameters_part_wb_sweep_set
tags:
- sweeps
toc_hide: true
type: docs
---

`wandb.config`를 사용하여 스윕 구성에서 하이퍼파라미터 이름과 값에 엑세스하세요. `wandb.config`는 사전처럼 작동합니다.

스윕 외부의 run의 경우, `wandb.init`에서 `config` 인수에 사전을 전달하여 `wandb.config` 값을 설정합니다. 스윕에서 `wandb.init`에 제공된 모든 설정은 스윕이 재정의할 수 있는 기본값으로 사용됩니다.

명시적인 행동을 위해 `config.setdefaults`를 사용하세요. 다음 코드 조각은 두 가지 방법을 모두 보여줍니다.

{{< tabpane text=true >}}
{{% tab "wandb.init()" %}}
```python
# 하이퍼파라미터의 기본값 설정
config_defaults = {"lr": 0.1, "batch_size": 256}

# run을 시작하고 스윕이 재정의할 수 있는 기본값 제공
with wandb.init(config=config_defaults) as run:
    # 여기에 트레이닝 코드 추가
    ...
```
{{% /tab %}}
{{% tab "config.setdefaults()" %}}
```python
# 하이퍼파라미터의 기본값 설정
config_defaults = {"lr": 0.1, "batch_size": 256}

# run 시작
with wandb.init() as run:
    # 스윕에 의해 설정되지 않은 모든 값 업데이트
    run.config.setdefaults(config_defaults)

    # 여기에 트레이닝 코드 추가
```
{{% /tab %}}
{{< /tabpane >}}
