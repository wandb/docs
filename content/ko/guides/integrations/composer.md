---
title: MosaicML Composer
description: 최신 알고리즘으로 신경망을 효과적으로 학습하세요
menu:
  default:
    identifier: ko-guides-integrations-composer
    parent: integrations
weight: 230
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/mosaicml/MosaicML_Composer_and_wandb.ipynb" >}}

[Composer](https://github.com/mosaicml/composer)는 신경망을 더 빠르고, 더 저렴하며, 더 잘 트레이닝할 수 있게 해주는 라이브러리입니다. 최신 신경망 트레이닝 가속화 및 일반화 기법을 다수 포함하고 있으며, 다양한 향상 방법을 _조합_ 해서 쉽게 쓸 수 있게 해주는 [Trainer](https://docs.mosaicml.com/projects/composer/en/stable/trainer/using_the_trainer.html) API도 제공합니다.

W&B는 ML 실험의 로그를 손쉽게 남길 수 있는 경량 래퍼를 제공합니다. 직접 두 라이브러리를 합칠 필요 없이, Composer 라이브러리에는 [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts)가 바로 내장되어 있습니다.

## W&B로 로그 남기기 시작하기

```python
from composer import Trainer
from composer.loggers import WandBLogger

trainer = Trainer(..., logger=WandBLogger())
```

{{< img src="/images/integrations/n6P7K4M.gif" alt="인터랙티브 대시보드" >}}

## Composer의 `WandBLogger` 사용하기

Composer 라이브러리는 `Trainer`에서 [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) 클래스를 사용하여 W&B에 메트릭을 기록합니다. 이 로거를 인스턴스화해서 `Trainer`에 전달하기만 하면 됩니다.

```python
wandb_logger = WandBLogger(project="gpt-5", log_artifacts=True)
trainer = Trainer(logger=wandb_logger)
```

## Logger 인수

아래는 `WandbLogger`에서 설정할 수 있는 파라미터입니다. 전체 목록과 설명은 [Composer 문서](https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.loggers.WandBLogger.html)를 참고하세요.

| 파라미터                       | 설명                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `project`                 | W&B Project 이름 (str, 선택)
| `group`                   | W&B group 이름 (str, 선택)
| `name`                   |  W&B Run 이름. 생략 시 State.run_name이 사용됩니다 (str, 선택)
| `entity`                   | W&B entity 이름(예: 사용자 이름 혹은 W&B Team 이름) (str, 선택)
| `tags`                   | W&B tags (List[str], 선택)
| `log_artifacts`                 | 체크포인트를 wandb에 로그할지 여부, 기본값: `false` (bool, 선택)|
| `rank_zero_only`         | rank 0 프로세스에서만 로그를 남길지 여부. Artifacts를 기록할 때는 모든 rank에서 로그를 남기길 권장합니다. rank가 1 이상인 곳에서의 Artifacts가 저장되지 않으므로 중요한 정보가 누락될 수 있습니다. 예를 들어, Deepspeed ZeRO를 사용할 때 모든 rank의 체크포인트가 없다면 복원이 불가능합니다. 기본값: `True` (bool, 선택)
| `init_kwargs`                   | `wandb.init()`에 전달할 파라미터 (`config` 등 가능). `wandb.init()`에서 사용할 수 있는 파라미터는 [`wandb.init()` 파라미터]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}})를 참고하세요.

일반적인 사용 예시는 다음과 같습니다.

```
init_kwargs = {"notes":"이번 실험에서 학습률을 높여 테스트합니다", 
               "config":{"arch":"Llama",
                         "use_mixed_precision":True
                         }
               }

wandb_logger = WandBLogger(log_artifacts=True, init_kwargs=init_kwargs)
```

## 예측 샘플 로그 남기기

[Composer의 Callbacks](https://docs.mosaicml.com/projects/composer/en/stable/trainer/callbacks.html) 시스템을 활용해 `WandBLogger`를 통해 언제 W&B에 로그를 남길지 제어할 수 있습니다. 아래 예시에서는 검증 이미지 및 예측 샘플의 일부를 로그로 남깁니다.

```python
import wandb
from composer import Callback, State, Logger

class LogPredictions(Callback):
    def __init__(self, num_samples=100, seed=1234):
        super().__init__()
        self.num_samples = num_samples
        self.data = []
        
    def eval_batch_end(self, state: State, logger: Logger):
        """배치별 예측값을 계산해서 self.data에 저장합니다"""
        
        if state.timer.epoch == state.max_duration: # 마지막 val epoch에서만
            if len(self.data) < self.num_samples:
                n = self.num_samples
                x, y = state.batch_pair
                outputs = state.outputs.argmax(-1)
                data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
                self.data += data
            
    def eval_end(self, state: State, logger: Logger):
        with wandb.init() as run:
            "wandb.Table을 만들고 로그로 남깁니다"
            columns = ['image', 'ground truth', 'prediction']
            table = wandb.Table(columns=columns, data=self.data[:self.num_samples])
            run.log({'sample_table':table}, step=int(state.timer.batch))         
...

trainer = Trainer(
    ...
    loggers=[WandBLogger()],
    callbacks=[LogPredictions()]
)
```