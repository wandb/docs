---
description: State of the art algorithms to train your neural networks
slug: /guides/integrations/composer
displayed_sidebar: default
---

# MosaicML Composer

[**Colab 노트북에서 시도해 보세요 →**](https://github.com/wandb/examples/blob/master/colabs/mosaicml/MosaicML_Composer_and_wandb.ipynb)

[Composer](https://github.com/mosaicml/composer)는 신경망을 더 잘, 더 빠르고, 더 저렴하게 훈련시키기 위한 라이브러리입니다. 신경망 트레이닝을 가속화하고 일반화를 개선하는 다양한 최신 방법들을 포함하고 있으며, 다양한 개선 사항을 쉽게 _조합_할 수 있는 선택적 [Trainer](https://docs.mosaicml.com/projects/composer/en/stable/trainer/using_the_trainer.html) API도 포함되어 있습니다.

Weights & Biases는 ML 실험을 로깅하기 위한 가벼운 래퍼를 제공합니다. 하지만 두 가지를 직접 결합할 필요는 없습니다: Weights & Biases는 [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts)를 통해 Composer 라이브러리에 직접 통합되어 있습니다.

## 1줄의 코드로 W&B에 로깅 시작하기

```python
from composer import Trainer
from composer.loggers import WandBLogger
﻿
trainer = Trainer(..., logger=WandBLogger())
```

![어디에서나 접근 가능한 인터랙티브 대시보드 등!](@site/static/images/integrations/n6P7K4M.gif)

## Composer의 `WandBLogger` 사용하기

Composer 라이브러리는 `Trainer`에서 메트릭을 Weights and Biases에 로깅하기 위해 [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) 클래스를 사용합니다. 로거를 인스턴스화하고 `Trainer`에 전달하는 것만큼 간단합니다.

```
wandb_logger = WandBLogger(project="gpt-5", log_artifacts=True)
trainer = Trainer(logger=wandb_logger)
```

### 로거 인수

WandbLogger의 파라미터는 아래와 같으며, 전체 목록과 설명은 [Composer 문서](https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.loggers.WandBLogger.html)를 참조하세요.

| 파라미터                       | 설명                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `project`                 | W&B 프로젝트 이름 (str, 선택 사항)
| `group`                   | W&B 그룹 이름 (str, 선택 사항)
| `name`                   |  W&B run 이름. 지정되지 않으면 State.run_name이 사용됩니다 (str, 선택 사항)
| `entity`                   | W&B 엔티티 이름, 예를 들어 사용자 이름이나 W&B 팀 이름 (str, 선택 사항)
| `tags`                   | W&B 태그 (List[str], 선택 사항)
| `log_artifacts`                 | Wandb에 체크포인트를 로깅할지 여부, 기본값: `false` (bool, 선택 사항)|
| `rank_zero_only`         | 로그를 rank-zero 프로세스에서만 로깅할지 여부. 아티팩트 로깅 시, 모든 랭크에서 로깅하는 것이 매우 권장됩니다. 랭크 ≥1의 아티팩트는 저장되지 않으며, 예를 들어 Deepspeed ZeRO를 사용할 때 모든 랭크의 아티팩트 없이는 체크포인트에서 복원할 수 없어 중요한 정보가 손실될 수 있습니다, 기본값: `True` (bool, 선택 사항)
| `init_kwargs`                   | `wandb.init`에 전달할 파라미터들, 예를 들어 wandb `config` 등 [여기를 참조하세요](https://docs.wandb.ai/ref/python/init)에서 `wandb.init`이 받는 전체 목록을 볼 수 있습니다.                                                                                                                                                                                   


일반적인 사용 예는 다음과 같습니다.

```
init_kwargs = {"notes":"이 실험에서는 높은 학습률을 테스트합니다.", 
               "config":{"arch":"Llama",
                         "use_mixed_precision":True
                         }
               }

wandb_logger = WandBLogger(log_artifacts=True, init_kwargs=init_kwargs)
```

### 예측 샘플 로깅하기

WandBLogger를 통해 Weights & Biases에 언제 로깅할지 제어할 수 있는 [Composer의 콜백](https://docs.mosaicml.com/projects/composer/en/stable/trainer/callbacks.html) 시스템을 사용할 수 있습니다. 이 예제에서는 검증 이미지와 예측값의 샘플이 로깅됩니다:

```python
import wandb
from composer import Callback, State, Logger

class LogPredictions(Callback):
    def __init__(self, num_samples=100, seed=1234):
        super().__init__()
        self.num_samples = num_samples
        self.data = []
        
    def eval_batch_end(self, state: State, logger: Logger):
        """배치 당 예측을 계산하고 self.data에 저장합니다"""
        
        if state.timer.epoch == state.max_duration: #마지막 val 에포크에서
            if len(self.data) < self.num_samples:
                n = self.num_samples
                x, y = state.batch_pair
                outputs = state.outputs.argmax(-1)
                data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
                self.data += data
            
    def eval_end(self, state: State, logger: Logger):
        "wandb.Table을 생성하고 로깅합니다"
        columns = ['image', '그라운드 트루스', '예측값']
        table = wandb.Table(columns=columns, data=self.data[:self.num_samples])
        wandb.log({'sample_table':table}, step=int(state.timer.batch))         
...

trainer = Trainer(
    ...
    loggers=[WandBLogger()],
    callbacks=[LogPredictions()]
)
```