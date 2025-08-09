---
title: W&B Sweep에서 모든 하이퍼파라미터의 값을 반드시 제공해야 하나요? 기본값을 설정할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-need_provide_values_all_hyperparameters_part_wb_sweep_set
support:
- 스윕
toc_hide: true
type: docs
url: /support/:filename
---

스윕 구성에서 하이퍼파라미터 이름과 값을 `(run.config())`을 사용해 엑세스할 수 있으며, 이는 사전처럼 동작합니다.

스윕 외부의 run에서는, `wandb.init()`에서 `config` 인수에 사전을 전달하여 `wandb.Run.config()` 값을 설정할 수 있습니다. 스윕 내에서는, `wandb.init()`에 제공된 구성 정보가 기본값이 되며, 스윕이 이를 덮어쓸 수 있습니다.

명시적인 행동이 필요할 때는 `wandb.Run.config.setdefaults()`를 사용하세요. 아래 코드조각에서 두 가지 방법 모두를 보여줍니다.

{{< tabpane text=true >}}
{{% tab "wandb.init()" %}}
```python
# 하이퍼파라미터 기본값 설정
config_defaults = {"lr": 0.1, "batch_size": 256}

# run을 시작하면서
# 스윕에서 덮어쓸 수 있는 기본값 제공
with wandb.init(config=config_defaults) as run:
    # 이곳에 트레이닝 코드를 추가하세요
    ...
```
{{% /tab %}}
{{% tab "config.setdefaults()" %}}
```python
# 하이퍼파라미터 기본값 설정
config_defaults = {"lr": 0.1, "batch_size": 256}

# run을 시작
with wandb.init() as run:
    # 스윕에서 지정하지 않은 값을 업데이트합니다
    run.config.setdefaults(config_defaults)

    # 이곳에 트레이닝 코드를 추가하세요
```
{{% /tab %}}
{{< /tabpane >}}