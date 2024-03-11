---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# PyTorch Lightning

[**Colab 노트북에서 시도해 보세요 →**](https://wandb.me/lightning)

PyTorch Lightning은 PyTorch 코드를 조직화하고 분산 트레이닝, 16비트 정밀도와 같은 고급 기능을 쉽게 추가할 수 있는 경량 래퍼를 제공합니다. W&B는 ML 실험을 로깅하기 위한 경량 래퍼를 제공합니다. 하지만 두 가지를 직접 결합할 필요는 없습니다: Weights & Biases는 [**`WandbLogger`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)를 통해 PyTorch Lightning 라이브러리에 직접 통합되어 있습니다.

## ⚡ 몇 줄만으로 번개처럼 빠르게 시작하세요.

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch 로거", value: "pytorch"},
    {label: "Fabric 로거", value: "fabric"},
]}>

<TabItem value="pytorch">

```python
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer

wandb_logger = WandbLogger(log_model="all")
trainer = Trainer(logger=wandb_logger)
```

:::info
**wandb.log() 사용하기:** `WandbLogger`는 Trainer의 `global_step`을 사용하여 W&B에 로그를 기록합니다. 코드에서 직접 `wandb.log`를 추가로 호출하는 경우, `wandb.log()`에서 `step` 인수를 **사용하지 마세요.**

대신 Trainer의 `global_step`을 다른 메트릭처럼 로그하세요:

`wandb.log({"accuracy":0.99, "trainer/global_step": step})`
:::

</TabItem>

<TabItem value="fabric">

```python
import lightning as L
from wandb.integration.lightning.fabric import WandbLogger

wandb_logger = WandbLogger(log_model="all")
fabric = L.Fabric(loggers=[wandb_logger])
fabric.launch()
fabric.log_dict({"important_metric": important_metric})
```

</TabItem>

</Tabs>


![어디서나 엑세스 가능한 인터랙티브 대시보드 및 기타!](@site/static/images/integrations/n6P7K4M.gif)

## wandb에 가입하고 로그인하기

a) [**가입하기**](https://wandb.ai/site) 무료 계정을 생성하세요

b) `wandb` 라이브러리를 Pip으로 설치하세요

c) 트레이닝 스크립트에서 로그인하려면, www.wandb.ai에서 계정에 로그인해 있어야 하며, **API 키는** [**인증 페이지**](https://wandb.ai/authorize)**에서 찾을 수 있습니다.**

처음으로 Weights and Biases를 사용한다면 [**퀵스타트**](../../quickstart.md)를 확인하세요.

<Tabs
  defaultValue="cli"
  values={[
    {label: '커맨드라인', value: 'cli'},
    {label: '노트북', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```bash
pip install wandb

wandb login
```

</TabItem>
  <TabItem value="notebook">

```notebook
!pip install wandb

import wandb
wandb.login()
```

  </TabItem>
</Tabs>

## PyTorch Lightning의 `WandbLogger` 사용하기

PyTorch Lightning에는 메트릭, 모델 가중치, 미디어 등을 원활하게 로그할 수 있는 여러 `WandbLogger`([**`Pytorch`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb))([**`Fabric`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)) 클래스가 있습니다. WandbLogger를 인스턴스화하고 Lightning의 `Trainer` 또는 `Fabric`에 전달하기만 하면 됩니다.

```
wandb_logger = WandbLogger()
```

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch 로거", value: "pytorch"},
    {label: "Fabric 로거", value: "fabric"},
]}>

<TabItem value="pytorch">

```
trainer = Trainer(logger=wandb_logger)
```

</TabItem>

<TabItem value="fabric">

```
fabric = L.Fabric(loggers=[wandb_logger])
fabric.launch()
fabric.log_dict({
    "important_metric": important_metric
})
```

</TabItem>

</Tabs>

### 로거 인수

아래는 WandbLogger에서 가장 많이 사용되는 매개변수입니다. 전체 목록과 설명을 보려면 PyTorch Lightning을 참조하세요.

- ([**`Pytorch`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb))
- ([**`Fabric`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb))

| 매개변수     | 설명                                                                      |
| ----------- | ------------------------------------------------------------------------ |
| `project`   | 어떤 wandb 프로젝트에 로그를 기록할지 정의합니다.                                         |
| `name`      | wandb 실행에 이름을 부여합니다.                                                   |
| `log_model` | `log_model="all"`인 경우 모든 모델을 로그하거나 `log_model=True`인 경우 트레이닝 종료 시 로그합니다. |
| `save_dir`  | 데이터가 저장되는 경로입니다.                                                     |

### 하이퍼파라미터 로깅하기

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch 로거", value: "pytorch"},
    {label: "Fabric 로거", value: "fabric"},
]}>

<TabItem value="pytorch">

```python
class LitModule(LightningModule):
    def __init__(self, *args, **kwarg):
        self.save_hyperparameters()
```

</TabItem>

<TabItem value="fabric">

```python
wandb_logger.log_hyperparams(
    {
        "hyperparameter_1": hyperparameter_1,
        "hyperparameter_2": hyperparameter_2,
    }
)
```

</TabItem>

</Tabs>

### 추가 구성 매개변수 로깅하기

```python
# 하나의 매개변수 추가
wandb_logger.experiment.config["key"] = value

# 여러 매개변수 추가
wandb_logger.experiment.config.update({key1: val1, key2: val2})

# wandb 모듈을 직접 사용
wandb.config["key"] = value
wandb.config.update()
```

### 그레이디언트, 매개변수 히스토그램 및 모델 토폴로지 로깅하기

`wandblogger.watch()`에 모델 오브젝트를 전달하여 트레이닝하는 동안 모델의 그레이디언트와 매개변수를 모니터링할 수 있습니다. PyTorch Lightning `WandbLogger` 문서를 참조하세요.

### 메트릭 로깅하기

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch 로거", value: "pytorch"},
    {label: "Fabric 로거", value: "fabric"},
]}>

<TabItem value="pytorch">

`WandbLogger`를 사용하여 W&B에 메트릭을 로그하려면 `LightningModule` 내에서 `self.log('my_metric_name', metric_vale)`를 호출하세요. 예를 들어, `training_step` 또는 `validation_step methods.`에서 사용할 수 있습니다.

아래 코드 조각은 메트릭과 `LightningModule` 하이퍼파라미터를 로깅하기 위해 `LightningModule`을 정의하는 방법을 보여줍니다. 이 예제에서는 메트릭을 계산하기 위해 [`torchmetrics`](https://github.com/PyTorchLightning/metrics) 라이브러리를 사용합니다.

```python
import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy
from lightning.pytorch import LightningModule


class My_LitModule(LightningModule):
    def __init__(self, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-3):
        """모델 매개변수를 정의하는 데 사용되는 메소드"""
        super().__init__()

        # mnist 이미지는 (1, 28, 28) (채널, 너비, 높이)입니다
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.layer_3 = Linear(n_layer_2, n_classes)

        self.loss = CrossEntropyLoss()
        self.lr = lr

        # 하이퍼-파라미터를 self.hparams에 저장합니다 (W&B에서 자동으로 로그됨)
        self.save_hyperparameters()

    def forward(self, x):
        """추론을 위해 사용되는 메소드 입력 -> 출력"""

        # (b, 1, 28, 28) -> (b, 1*28*28)
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        # 3 x (linear + relu)를 해봅시다
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        """단일 배치에서 손실을 반환해야 합니다"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # 손실과 메트릭 로그
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """메트릭 로깅에 사용됩니다"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # 손실과 메트릭 로그
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds

    def configure_optimizers(self):
        """모델 옵티마이저를 정의합니다"""
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        """train/valid/test 단계가 비슷하기 때문에 편리한 함수입니다"""
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y)
        return preds, loss, acc
```

</TabItem>

<TabItem value="fabric">

```python
import lightning as L
import torch
import torchvision as tv
from wandb.integration.lightning.fabric import WandbLogger
import wandb

fabric = L.Fabric(loggers=[wandb_logger])
fabric.launch()

model = tv.models.resnet18()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
model, optimizer = fabric.setup(model, optimizer)

train_dataloader = fabric.setup_dataloaders(
    torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
)

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        fabric.log_dict({"loss": loss})
```

</TabItem>

</Tabs>

### 메트릭의 최소/최대 로깅하기

wandb의 [`define_metric`](https://docs.wandb.ai/ref/python/run#define\_metric) 함수를 사용하면 W&B 요약 메트릭에 해당 메트릭의 최소, 최대, 평균 또는 최적값을 표시할지 정의할 수 있습니다. `define_metric`을 사용하지 않으면 로그된 마지막 값이 요약 메트릭에 표시됩니다. `define_metric` [참조 문서](https://docs.wandb.ai/ref/python/run#define\_metric)와 [가이드](https://docs.wandb.ai/guides/track/log#customize-axes-and-summaries-with-define\_metric)를 참조하세요.

W&B 요약 메트릭에서 검증 정확도의 최대값을 추적하려면 `wandb.define_metric`을 한 번만 호출하면 됩니다. 예를 들어, 트레이닝 시작 시 다음과 같이 호출할 수 있습니다:

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch 로거", value: "pytorch"},
    {label: "Fabric 로거", value: "fabric"},
]}>

<TabItem value="pytorch">

```python
class My_LitModule(LightningModule):
    ...

    def validation_step(self, batch, batch_idx):
        if trainer.global_step == 0:
            wandb.define_metric("val_accuracy", summary="max")

        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # 손실과 메트릭 로그
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds
```

</TabItem>

<TabItem value="fabric">

```python
wandb.define_metric("val_accuracy", summary="max")
fabric = L.Fabric(loggers=[wandb_logger])
fabric.launch()
fabric.log_dict({"val_accuracy": val_accuracy})
```

</TabItem>

</Tabs>

### 모델 체크포인트하기

W&B [아티팩트](https://docs.wandb.ai/guides/data-and-model-versioning)로 모델 체크포인트를 저장하려면, Lightning [`ModelCheckpoint`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch\_lightning.callbacks.ModelCheckpoint.html#pytorch\_lightning.callbacks.ModelCheckpoint) 콜백을 사용하고 `WandbLogger`에서 `log_model` 인수를 설정하세요:

```python
# `val_accuracy`가 증가할 때만 모델 로깅
wandb_logger = WandbLogger(log_model="all")
checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
```

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch 로거", value: "pytorch"},
    {label: "Fabric 로거", value: "fabric"},
]}>

<TabItem value="pytorch">

```python
trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback])
```

</TabItem>

<TabItem value="fabric">

```python
fabric = L.Fabric(loggers=[wandb_logger], callbacks=[checkpoint_callback])
```

</TabItem>

</Tabs>


최신 및 최고 별칭은 W&B [아티팩트](https://docs.wandb.ai/guides/data-and-model-versioning)에서 모델 체크포인트를 쉽게 검색할 수 있도록 자동으로 설정됩니다:

```python
# 아티팩트 패널에서 참조를 검색할 수 있습니다
# "VERSION"은 버전(예: "v2") 또는 별칭("latest" 또는 "best")일 수 있습니다
checkpoint_reference = "USER/PROJECT/MODEL-RUN_ID:VERSION"
```

<Tabs
  defaultValue="logger"
  values={[
    {label: "로거를 통해", value: "logger"},
    {label: "wandb를 통해", value: "wandb"},
]}>

<TabItem value="logger">

```python
# 로컬에 체크포인트 다운로드(캐시되지 않은 경우)
wandb_logger.download_artifact(checkpoint_reference, artifact_type="model")
```

</TabItem>

<TabItem value="wandb">

```python
# 로컬에 체크포인트 다운로드(캐시되지 않은 경우)
run = wandb.init(project="MNIST")
artifact = run.use_artifact(checkpoint_reference, type="model")
artifact_dir = artifact.download()
```

</TabItem>

</Tabs>


<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch 로거", value: "pytorch"},
    {label: "Fabric 로거", value: "fabric"},
]}>

<TabItem value="pytorch">

```python
# 체크포인트 로드
model = LitModule.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
```

</TabItem>

<TabItem value="fabric">



### Lightning과 W&B를 사용하여 여러 GPU를 사용하는 방법은?

PyTorch Lightning은 DDP 인터페이스를 통해 Multi-GPU 지원을 합니다. 그러나, PyTorch Lightning의 설계는 우리가 GPU를 인스턴스화하는 방법에 대해 주의를 기울여야 합니다.

Lightning은 트레이닝 루프의 각 GPU(또는 랭크)가 동일한 초기 조건으로 정확히 동일한 방식으로 인스턴스화되어야 한다고 가정합니다. 그러나, 0번 랭크 프로세스만 `wandb.run` 오브젝트에 엑세스할 수 있으며, 0이 아닌 랭크 프로세스의 경우: `wandb.run = None`입니다. 이로 인해 0이 아닌 랭크 프로세스가 실패할 수 있습니다. 이러한 상황은 0번 랭크 프로세스가 이미 충돌한 0이 아닌 랭크 프로세스가 조인할 때까지 기다리게 되는 **데드락**에 빠질 수 있습니다.

이러한 이유로, 우리는 트레이닝 코드를 설정하는 방법에 대해 주의를 기울여야 합니다. 권장되는 방법은 `wandb.run` 오브젝트와 독립적인 코드를 가지는 것입니다.

```python
class MNISTClassifier(pl.LightningModule):
    def __init__(self):
        super(MNISTClassifier, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        self.log("train/loss", loss)
        return {"train_loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        self.log("val/loss", loss)
        return {"val_loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def main():
    # 모든 랜덤 시드를 동일한 값으로 설정합니다.
    # 이것은 분산 트레이닝 설정에서 중요합니다.
    # 각 랭크는 자체의 초기 가중치 세트를 받습니다.
    # 일치하지 않으면, 그레이디언트도 일치하지 않을 것이며,
    # 수렴하지 않을 수 있는 트레이닝으로 이어질 수 있습니다.
    pl.seed_everything(1)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    model = MNISTClassifier()
    wandb_logger = WandbLogger(project="<project_name>")
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            every_n_train_steps=100,
        ),
    ]
    trainer = pl.Trainer(
        max_epochs=3, gpus=2, logger=wandb_logger, strategy="ddp", callbacks=callbacks
    )
    trainer.fit(model, train_loader, val_loader)
```

## 인터랙티브한 예제를 확인하세요!

우리의 비디오 튜토리얼과 함께하는 튜토리얼 콜랩을 [여기](https://wandb.me/lit-colab)에서 따라할 수 있습니다.

## 자주 묻는 질문들

### W&B는 Lightning과 어떻게 통합되나요?

핵심 통합은 [Lightning `loggers` API](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html)를 기반으로 하며, 이를 통해 많은 로깅 코드를 프레임워크에 구애받지 않는 방식으로 작성할 수 있습니다. `Logger`들은 [Lightning `Trainer`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)에 전달되며, 그 API의 풍부한 [훅-앤-콜백 시스템](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html)에 기반하여 트리거됩니다. 이는 연구 코드를 엔지니어링 및 로깅 코드와 잘 분리시킵니다.

### 추가 코드 없이 통합은 무엇을 로깅하나요?

W&B에 모델 체크포인트를 저장하며, 여기서 모델 체크포인트를 보거나 미래의 run에서 사용하기 위해 다운로드할 수 있습니다. GPU 사용량 및 네트워크 I/O와 같은 [시스템 메트릭](../app/features/system-metrics.md), 하드웨어 및 OS 정보와 같은 환경 정보, [코드 상태](../app/features/panels/code.md)(git 커밋 및 diff 패치, 노트북 내용 및 세션 이력 포함), 그리고 표준 출력으로 인쇄된 모든 것을 포착합니다.

### 트레이닝 설정에서 정말로 `wandb.run`을 사용해야 한다면?

본질적으로 엑세스해야 할 변수의 범위를 스스로 확장해야 합니다. 즉, 모든 프로세스에서 초기 조건이 동일하도록 보장해야 합니다.

```python
if os.environ.get("LOCAL_RANK", None) is None:
    os.environ["WANDB_DIR"] = wandb.run.dir
```

그런 다음, `os.environ["WANDB_DIR"]`를 사용하여 모델 체크포인트 디렉토리를 설정할 수 있습니다. 이 방법으로, `wandb.run.dir`은 0이 아닌 랭크 프로세스에서도 사용될 수 있습니다.