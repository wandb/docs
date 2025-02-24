---
title: MosaicML Composer
description: 신경망 을 학습시키는 최첨단 알고리즘
menu:
  default:
    identifier: ko-guides-integrations-composer
    parent: integrations
weight: 230
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/mosaicml/MosaicML_Composer_and_wandb.ipynb" >}}

[Composer](https://github.com/mosaicml/composer)는 신경망을 더 좋고, 더 빠르고, 더 저렴하게 트레이닝하기 위한 라이브러리입니다. 여기에는 신경망 트레이닝을 가속화하고 일반화를 개선하기 위한 최첨단 메소드와 다양한 개선 사항을 쉽게 _구성_할 수 있는 선택적 [Trainer](https://docs.mosaicml.com/projects/composer/en/stable/trainer/using_the_trainer.html) API가 포함되어 있습니다.

Weights & Biases는 ML Experiments 로깅을 위한 가벼운 래퍼를 제공합니다. 하지만 직접 결합할 필요가 없습니다. Weights & Biases는 [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts)를 통해 Composer 라이브러리에 직접 통합됩니다.

## Weights & Biases에 로깅 시작하기

```python
from composer import Trainer
from composer.loggers import WandBLogger
﻿
trainer = Trainer(..., logger=WandBLogger())
```

{{< img src="/images/integrations/n6P7K4M.gif" alt="Interactive dashboards accessible anywhere, and more!" >}}

## Composer의 `WandBLogger` 사용

Composer 라이브러리는 `Trainer`에서 [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) 클래스를 사용하여 메트릭을 Weights & Biases에 로깅합니다. 로거를 인스턴스화하고 `Trainer`에 전달하는 것만큼 간단합니다.

```python
wandb_logger = WandBLogger(project="gpt-5", log_artifacts=True)
trainer = Trainer(logger=wandb_logger)
```

## 로거 인수

아래에서 WandbLogger에 대한 파라미터를 확인하세요. 전체 목록 및 설명은 [Composer documentation](https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.loggers.WandBLogger.html)을 참조하세요.

| 파라미터                       | 설명                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `project`                 | W&B 프로젝트 이름 (str, optional)
| `group`                   | W&B 그룹 이름 (str, optional)
| `name`                   |  W&B run 이름. 지정하지 않으면 State.run_name이 사용됩니다 (str, optional)
| `entity`                   | 사용자 이름 또는 W&B Team 이름과 같은 W&B entity 이름 (str, optional)
| `tags`                   | W&B 태그 (List[str], optional)
| `log_artifacts`                 | 체크포인트를 wandb에 로깅할지 여부, 기본값: `false` (bool, optional)|
| `rank_zero_only`         | rank-zero 프로세스에서만 로깅할지 여부. 아티팩트를 로깅할 때는 모든 순위에서 로깅하는 것이 좋습니다. 순위 ≥1의 아티팩트는 저장되지 않으므로 관련 정보가 삭제될 수 있습니다. 예를 들어 Deepspeed ZeRO를 사용하는 경우 모든 순위의 아티팩트 없이는 체크포인트에서 복원할 수 없습니다. 기본값: `True` (bool, optional)
| `init_kwargs`                   | wandb `config` 등과 같이 `wandb.init`에 전달할 파라미터 [전체 목록은 여기]({{< relref path="/ref/python/init" lang="ko" >}})에서 `wandb.init`이 허용하는 파라미터를 참조하세요.                                                                                                                                                                                   


일반적인 사용법은 다음과 같습니다.

```
init_kwargs = {"notes":"이 실험에서 더 높은 학습률 테스트", 
               "config":{"arch":"Llama",
                         "use_mixed_precision":True
                         }
               }

wandb_logger = WandBLogger(log_artifacts=True, init_kwargs=init_kwargs)
```

## 예측 샘플 로깅

[Composer의 콜백](https://docs.mosaicml.com/projects/composer/en/stable/trainer/callbacks.html) 시스템을 사용하여 WandBLogger를 통해 Weights & Biases에 로깅하는 시기를 제어할 수 있습니다. 이 예에서는 유효성 검사 이미지 및 예측 샘플이 로깅됩니다.

```python
import wandb
from composer import Callback, State, Logger

class LogPredictions(Callback):
    def __init__(self, num_samples=100, seed=1234):
        super().__init__()
        self.num_samples = num_samples
        self.data = []
        
    def eval_batch_end(self, state: State, logger: Logger):
        """배치당 예측을 계산하고 self.data에 저장합니다."""
        
        if state.timer.epoch == state.max_duration: #on last val epoch
            if len(self.data) < self.num_samples:
                n = self.num_samples
                x, y = state.batch_pair
                outputs = state.outputs.argmax(-1)
                data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
                self.data += data
            
    def eval_end(self, state: State, logger: Logger):
        "wandb.Table을 만들고 로깅합니다."
        columns = ['image', 'ground truth', 'prediction']
        table = wandb.Table(columns=columns, data=self.data[:self.num_samples])
        wandb.log({'sample_table':table}, step=int(state.timer.batch))         
...

trainer = Trainer(
    ...
    loggers=[WandBLogger()],
    callbacks=[LogPredictions()]
)
```