---
title: MosaicML Composer
description: 최신 알고리즘으로 신경망을 훈련하세요
slug: /guides/integrations/composer
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://github.com/wandb/examples/blob/master/colabs/mosaicml/MosaicML_Composer_and_wandb.ipynb"></CTAButtons>

[Composer](https://github.com/mosaicml/composer)는 신경망 트레이닝을 더 잘, 빠르게, 저렴하게 수행하기 위한 라이브러리입니다. 신경망 트레이닝을 가속하고 일반화를 개선하기 위한 최신 메소드를 많이 포함하고 있으며, 여러 가지 개선을 손쉽게 _구성하는_ 것을 가능하게 하는 선택적 [Trainer](https://docs.mosaicml.com/projects/composer/en/stable/trainer/using_the_trainer.html) API도 포함하고 있습니다.

W&B는 ML 실험을 로깅하는 경량 래퍼를 제공합니다. 하지만 두 가지를 직접 결합할 필요는 없습니다. W&B는 [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts)를 통해 Composer 라이브러리에 직접 통합되어 있습니다.

## 1줄의 코드로 W&B에 로깅 시작하기

```python
from composer import Trainer
from composer.loggers import WandBLogger

trainer = Trainer(..., logger=WandBLogger())
```

![어디서나 엑세스할 수 있는 대화형 대시보드, 그 외에도 더 많은 기능!](/images/integrations/n6P7K4M.gif)

## Composer의 `WandBLogger` 사용하기

Composer 라이브러리는 `Trainer`에서 [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) 클래스를 사용하여 Weights and Biases에 메트릭을 로그합니다. 로거를 인스턴스화하고 이를 `Trainer`에 전달하기만 하면 됩니다.

```
wandb_logger = WandBLogger(project="gpt-5", log_artifacts=True)
trainer = Trainer(logger=wandb_logger)
```

### Logger 인수

WandbLogger의 파라미터는 아래에서 확인할 수 있으며, 전체 목록과 설명은 [Composer documentation](https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.loggers.WandBLogger.html)에서 확인하세요.

| 파라미터                       | 설명                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `project`                 | W&B 프로젝트 이름 (str, 선택 사항)
| `group`                   | W&B 그룹 이름 (str, 선택 사항)
| `name`                   |  W&B run 이름. 지정하지 않으면 State.run_name이 사용됨 (str, 선택 사항)
| `entity`                 | W&B 엔티티 이름, 사용자 이름 또는 W&B 팀 이름 (str, 선택 사항)
| `tags`                   | W&B 태그 (List[str], 선택 사항)
| `log_artifacts`                 | wandb에 체크포인트를 로그할지 여부, 기본값: `false` (bool, 선택 사항)|
| `rank_zero_only`         | 랭크 제로 프로세스에서만 로그할지 여부. 아티팩트를 로그할 때는 모든 랭크에서 로그하는 것이 강력히 권장됩니다. 랭크 ≥1의 아티팩트는 저장되지 않아 관련 정보를 버릴 수 있습니다. 예를 들어, Deepspeed ZeRO를 사용할 경우, 모든 랭크의 아티팩트 없이는 체크포인트에서 복원할 수 없습니다. 기본값: `True` (bool, 선택 사항)
| `init_kwargs`                   | `wandb.init`에 전달할 파라미터, 예를 들면 wandb `config` 등 [여기](/ref/python/init)에서 `wandb.init`이 수용하는 전체 목록을 참조하십시오.                                                                                                                                                                                   

일반적인 사용 예시는 다음과 같습니다:

```
init_kwargs = {"notes":"이 실험에서 더 높은 학습률 테스트", 
               "config":{"arch":"Llama",
                         "use_mixed_precision":True
                         }
               }

wandb_logger = WandBLogger(log_artifacts=True, init_kwargs=init_kwargs)
```

### 예측 샘플 로그하기

[Composer의 콜백](https://docs.mosaicml.com/projects/composer/en/stable/trainer/callbacks.html) 시스템을 사용하여 WandBLogger를 통해 Weights & Biases에 언제 로그할지 제어할 수 있으며, 이 예에서는 검증 이미지와 예측 샘플을 로그합니다:

```python
import wandb
from composer import Callback, State, Logger

class LogPredictions(Callback):
    def __init__(self, num_samples=100, seed=1234):
        super().__init__()
        self.num_samples = num_samples
        self.data = []
        
    def eval_batch_end(self, state: State, logger: Logger):
        """배치마다 예측값을 계산하고 self.data에 저장합니다"""
        
        if state.timer.epoch == state.max_duration: #마지막 검증 에포크에서
            if len(self.data) < self.num_samples:
                n = self.num_samples
                x, y = state.batch_pair
                outputs = state.outputs.argmax(-1)
                data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
                self.data += data
            
    def eval_end(self, state: State, logger: Logger):
        "wandb.Table을 생성하고 이를 로그합니다"
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