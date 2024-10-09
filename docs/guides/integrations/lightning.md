---
title: PyTorch Lightning
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Optimize_Pytorch_Lightning_models_with_Weights_%26_Biases.ipynb"></CTAButtons>

PyTorch Lightning은 PyTorch 코드를 구성하고 분산 트레이닝 및 16비트 정밀도와 같은 고급 기능을 쉽게 추가할 수 있는 경량 래퍼를 제공합니다. W&B는 ML 실험을 기록하는 경량 래퍼를 제공합니다. 그러나 두 가지를 스스로 결합할 필요는 없습니다: Weights & Biases는 PyTorch Lightning 라이브러리에 [**`WandbLogger`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)로 직접 통합되어 있습니다.

## ⚡ 몇 줄의 코드로 빠르게 시작하세요.

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch Logger", value: "pytorch"},
    {label: "Fabric Logger", value: "fabric"},
]}>

<TabItem value="pytorch">

```python
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer

wandb_logger = WandbLogger(log_model="all")
trainer = Trainer(logger=wandb_logger)
```

:::info
**wandb.log() 사용:** `WandbLogger`는 Trainer의 `global_step`을 사용하여 W&B에 기록합니다. 코드에서 `wandb.log`를 추가로 호출하는 경우, `wandb.log()`의 `step` 인수를 사용하지 **마세요.**

대신, 다른 메트릭처럼 Trainer의 `global_step`을 기록하세요. 다음과 같이 하세요:

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

![대시보드를 어디서든 접근 가능하며, 그 외 더 많은 기능!](/images/integrations/n6P7K4M.gif)

## wandb에 가입하고 로그인하기

a) [**무료 계정 가입**](https://wandb.ai/site)

b) `wandb` 라이브러리를 pip로 설치

c) 트레이닝 스크립트에서 로그인하려면, www.wandb.ai에 접속하여 계정에 로그인해야 하며, **`API 키`를 [**인증 페이지**](https://wandb.ai/authorize)에서 찾을 수 있습니다.**

Weights and Biases를 처음 사용하는 경우, [**퀵스타트**](../../quickstart.md) 가이드를 참조하세요.

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
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

PyTorch Lightning에는 메트릭, 모델 가중치, 미디어 등을 원활하게 기록할 수 있는 다양한 `WandbLogger` ([**`Pytorch`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)) ([**`Fabric`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)) 클래스가 있습니다. WandbLogger를 생성하고 이를 Lightning의 `Trainer` 또는 `Fabric`에 전달하기만 하면 됩니다.

```
wandb_logger = WandbLogger()
```

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch Logger", value: "pytorch"},
    {label: "Fabric Logger", value: "fabric"},
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

아래는 WandbLogger에서 자주 사용되는 파라미터들입니다. 전체 목록과 설명은 PyTorch Lightning 문서를 참조하세요

- ([**`Pytorch`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb))
- ([**`Fabric`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb))

| 파라미터   | 설명                                                                           |
| ----------- | ----------------------------------------------------------------------------- |
| `project`   | 로깅할 wandb 프로젝트를 정의합니다                                        |
| `name`      | wandb run의 이름을 지정합니다                                               |
| `log_model` | `log_model="all"`일 경우 모든 모델을 기록하거나, `log_model=True`일 경우 트레이닝 종료 시 기록합니다 |
| `save_dir`  | 데이터가 저장될 경로입니다                                                     |

### 하이퍼파라미터 기록

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch Logger", value: "pytorch"},
    {label: "Fabric Logger", value: "fabric"},
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

### 추가 구성 파라미터 기록

```python
# 하나의 파라미터 추가
wandb_logger.experiment.config["key"] = value

# 여러 파라미터 추가
wandb_logger.experiment.config.update({key1: val1, key2: val2})

# wandb 모듈을 직접 사용
wandb.config["key"] = value
wandb.config.update()
```

### 그레이디언트, 파라미터 히스토그램 및 모델 토폴로지 기록

`wandblogger.watch()`에 여러분의 모델 오브젝트를 전달하여 트레이닝 중 모델의 그레이디언트와 파라미터를 모니터링할 수 있습니다. PyTorch Lightning `WandbLogger` 문서를 참조하세요.

### 메트릭 기록

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch Logger", value: "pytorch"},
    {label: "Fabric Logger", value: "fabric"},
]}>

<TabItem value="pytorch">

`LightningModule` 내의 `training_step` 또는 `validation_step 메소드`에서 `self.log('my_metric_name', metric_vale)`을 호출하여 메트릭을 W&B에 기록할 수 있습니다.

아래 코드조각은 메트릭과 LightningModule 하이퍼파라미터를 기록하는 방법을 보여줍니다. 이 예시에서는 [`torchmetrics`](https://github.com/PyTorchLightning/metrics) 라이브러리를 사용하여 메트릭을 계산합니다.

```python
import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy
from lightning.pytorch import LightningModule


class My_LitModule(LightningModule):
    def __init__(self, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-3):
        """모델 파라미터를 정의하는 메소드"""
        super().__init__()

        # mnist 이미지 크기 (1, 28, 28) (채널 수, 폭, 높이)
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.layer_3 = Linear(n_layer_2, n_classes)

        self.loss = CrossEntropyLoss()
        self.lr = lr

        # 하이퍼파라미터를 self.hparams에 저장 (W&B에 자동 기록)
        self.save_hyperparameters()

    def forward(self, x):
        """추론 입력 -> 출력에 사용되는 메소드"""

        # (b, 1, 28, 28) -> (b, 1*28*28)
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        # 3번의 (선형 + relu) 연산 수행
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        """단일 배치로부터 손실을 반환해야 합니다"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # 손실과 메트릭 기록
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """메트릭 기록에 사용됩니다"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # 손실과 메트릭 기록
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds

    def configure_optimizers(self):
        """모델 옵티마이저 정의"""
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        """train/valid/test 단계가 유사하기 때문에 편리한 함수"""
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

### 메트릭의 최소/최대 값 기록

wandb의 [`define_metric`](/ref/python/run#define_metric) 함수를 사용하여 W&B 요약 메트릭이 해당 메트릭의 최소, 최대, 평균 혹은 최상의 값을 표시하도록 정의할 수 있습니다. `define_metric`이 사용되지 않으면 마지막으로 기록된 값이 요약 메트릭에 표시됩니다. `define_metric` [참조 문서](/ref/python/run#define_metric) 및 [가이드](/guides/track/log/customize-logging-axes)를 참조하세요.

W&B 요약 메트릭에서 최대 검증 정확도를 추적하려면, 트레이닝 시작 시 `wandb.define_metric`을 한 번 호출하면 됩니다. 예를 들어, 트레이닝 시작 시 다음과 같이 호출할 수 있습니다:

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch Logger", value: "pytorch"},
    {label: "Fabric Logger", value: "fabric"},
]}>

<TabItem value="pytorch">

```python
class My_LitModule(LightningModule):
    ...

    def validation_step(self, batch, batch_idx):
        if trainer.global_step == 0:
            wandb.define_metric("val_accuracy", summary="max")

        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # 손실과 메트릭 기록
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

### 모델 체크포인팅

모델 체크포인트를 W&B [Artifacts](/guides/artifacts/)로 저장하려면, Lightning [`ModelCheckpoint`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) 콜백을 사용하고 `WandbLogger`에서 `log_model` 인수를 설정하세요:

```python
# `val_accuracy`가 증가할 때 모델만 기록
wandb_logger = WandbLogger(log_model="all")
checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
```

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch Logger", value: "pytorch"},
    {label: "Fabric Logger", value: "fabric"},
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

_최신_ 및 _최고_ 에일리어스는 W&B [Artifact](/guides/artifacts/)에서 모델 체크포인트를 쉽게 검색하기 위해 자동으로 설정됩니다:

```python
# 아티팩트 패널에서 참조를 검색할 수 있습니다
# "VERSION"은 버전(ex: "v2")이나 에일리어스("latest" 또는 "best")가 될 수 있습니다
checkpoint_reference = "USER/PROJECT/MODEL-RUN_ID:VERSION"
```

<Tabs
  defaultValue="logger"
  values={[
    {label: "Via Logger", value: "logger"},
    {label: "Via wandb", value: "wandb"},
]}>

<TabItem value="logger">

```python
# 로컬에 체크포인트 다운로드 (이미 캐시되지 않은 경우)
wandb_logger.download_artifact(checkpoint_reference, artifact_type="model")
```

</TabItem>

<TabItem value="wandb">

```python
# 로컬에 체크포인트 다운로드 (이미 캐시되지 않은 경우)
run = wandb.init(project="MNIST")
artifact = run.use_artifact(checkpoint_reference, type="model")
artifact_dir = artifact.download()
```

</TabItem>

</Tabs>

<Tabs
  defaultValue="pytorch"
  values={[
    {label: "Pytorch Logger", value: "pytorch"},
    {label: "Fabric Logger", value: "fabric"},
]}>

<TabItem value="pytorch">

```python
# 체크포인트 로드
model = LitModule.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
```

</TabItem>

<TabItem value="fabric">

```python
# 원시 체크포인트 요청
full_checkpoint = fabric.load(Path(artifact_dir) / "model.ckpt")

model.load_state_dict(full_checkpoint["model"])
optimizer.load_state_dict(full_checkpoint["optimizer"])
```

</TabItem>

</Tabs>

로그된 모델 체크포인트는 [W&B Artifacts](/guides/artifacts) UI를 통해 확인할 수 있으며, 전체 모델 계보를 포함합니다 (UI에서의 모델 체크포인트 예시 [여기](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..)에서 볼 수 있습니다).

W&B 모델 레지스트리를 통해 팀 전체에 최고의 모델 체크포인트를 북마크하고 중앙 집중화할 수 있습니다. 여기서 작업별 최고의 모델을 조직하고, 모델 수명주기를 관리하며, ML 라이프사이클 전반에 걸쳐 추적 및 감사를 용이하게 하며, 웹훅이나 작업으로 후속 작업을 [자동화](/guides/artifacts/project-scoped-automations/#create-a-webhook-automation)할 수 있습니다.

### 이미지, 텍스트 및 기타 로그하기

`WandbLogger`에는 미디어를 기록하기 위한 `log_image`, `log_text` 및 `log_table` 메소드가 있습니다.

또한 `wandb.log` 또는 `trainer.logger.experiment.log`를 직접 호출하여 Audio, Molecules, Point Clouds, 3D Objects 등과 같은 다른 미디어 유형을 기록할 수 있습니다.

<Tabs
  defaultValue="images"
  values={[
    {label: '이미지 로그', value: 'images'},
    {label: '텍스트 로그', value: 'text'},
    {label: '테이블 로그', value: 'tables'},
  ]}>
  <TabItem value="images">

```python
# 텐서, numpy 배열 또는 PIL 이미지를 사용
wandb_logger.log_image(key="samples", images=[img1, img2])

# 캡션 추가
wandb_logger.log_image(key="samples", images=[img1, img2], caption=["tree", "person"])

# 파일 경로 사용
wandb_logger.log_image(key="samples", images=["img_1.jpg", "img_2.jpg"])

# 트레이너에서 .log 사용
trainer.logger.experiment.log(
    {"samples": [wandb.Image(img, caption=caption) for (img, caption) in my_images]},
    step=current_trainer_global_step,
)
```
</TabItem>
<TabItem value="text">

```python
# 데이터는 리스트의 리스트여야 합니다
columns = ["input", "label", "prediction"]
my_data = [["cheese", "english", "english"], ["fromage", "french", "spanish"]]

# 열과 데이터를 사용
wandb_logger.log_text(key="my_samples", columns=columns, data=my_data)

# 판다스 DataFrame 사용
wandb_logger.log_text(key="my_samples", dataframe=my_dataframe)
```

</TabItem>
<TabItem value="tables">

```python
# 텍스트 캡션, 이미지, 오디오가 포함된 W&B 테이블 로그
columns = ["caption", "image", "sound"]

# 데이터는 리스트의 리스트여야 합니다.
my_data = [
    ["cheese", wandb.Image(img_1), wandb.Audio(snd_1)],
    ["wine", wandb.Image(img_2), wandb.Audio(snd_2)],
]

# 테이블 로그
wandb_logger.log_table(key="my_samples", columns=columns, data=data)
```

</TabItem>
</Tabs>

Lightning의 콜백 시스템을 사용하여 WandbLogger를 통해 Weights & Biases에 언제 로그를 할지 제어할 수 있습니다. 이 예시는 검증 이미지와 예측을 샘플로 로그하는 방법을 보여줍니다:

```python
import torch
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

# 또는
# from wandb.integration.lightning.fabric import WandbLogger


class LogPredictionSamplesCallback(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """검증 배치가 끝날 때 호출됩니다."""

        # `outputs`는 모델 예측에 해당하는 `LightningModule.validation_step`에서 나옵니다.

        # 첫 번째 배치에서 20개의 샘플 이미지 예측을 로그합시다.
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [
                f"Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(y[:n], outputs[:n])
            ]

            # 옵션 1: `WandbLogger.log_image`으로 이미지 로그
            wandb_logger.log_image(key="sample_images", images=images, caption=captions)

            # 옵션 2: 이미지와 예측을 W&B 테이블로 로그
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in zip(x[:n], y[:n], outputs[:n])
            ]
            wandb_logger.log_table(key="sample_table", columns=columns, data=data)


trainer = pl.Trainer(callbacks=[LogPredictionSamplesCallback()])
```

### Lightning과 W&B를 사용하여 다중 GPU를 사용하는 방법은 무엇인가요?

PyTorch Lightning은 DDP 인터페이스를 통해 다중 GPU 지원을 제공합니다. 그러나 PyTorch Lightning의 설계는 GPU를 인스턴스화하는 방법에 대해 주의가 필요합니다.

Lightning은 트레이닝 루프 내의 각 GPU(또는 Rank)가 동일한 초기 조건으로 인스턴스화되기를 가정합니다. 하지만, 0이 아닌 랭크 프로세스는 `wandb.run` 객체에 엑세스할 수 없습니다. 이러한 상황은 0이 아닌 랭크 프로세스가 이미 충돌했으므로, 0 랭크 프로세스가 난처하게도 **교착상태**에 빠질 수 있습니다.

이 때문에 트레이닝 코드를 설정할 때 주의해야 합니다. 코드가 `wandb.run` 객체에 독립적이도록 설정하는 것이 권장됩니다.

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
    # 모든 랜덤 시드를 같은 값으로 설정합니다.
    # 분산 학습 환경에서 이것은 중요합니다.
    # 각 랭크는 자체 초기 가중치를 갖게 됩니다.
    # 일치하지 않으면, 그레이디언트도 불일치하여 학습이 수렴하지 않을 수 있습니다.
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

## 인터랙티브 예제를 확인하세요!

우리의 비디오 튜토리얼과 함께 이 콜랩에서 튜토리얼을 따를 수 있습니다 [여기](https://wandb.me/lit-colab)

## 자주 묻는 질문

### W&B는 어떻게 Lightning과 통합되나요?

핵심 통합은 프레임워크에 구애받지 않는 방식으로 많은 로깅 코드를 작성할 수 있게 해주는 [Lightning `loggers` API](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html)를 기반으로 합니다. `Logger`는 [Lightning `Trainer`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)에 전달되며, 이 API의 풍부한 [hook-and-callback system](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html)에 의해 트리거됩니다. 이를 통해 연구 코드가 엔지니어링 및 로깅 코드와 잘 분리됩니다.

### 추가 코드 없이 통합은 무엇을 기록하나요?

모델 체크포인트를 W&B에 저장하여, 이후 run에서 사용하거나 다운로드할 수 있습니다. 또한 [시스템 메트릭](../app/features/system-metrics.md), 예를 들어 GPU 사용량 및 네트워크 I/O, 하드웨어 및 OS 정보와 같은 환경 정보를 캡처하고, 코드 상태(예: git 커밋 및 diff 패치, 노트북 콘텐츠 및 세션 히스토리), 표준 출력에 인쇄된 모든 것을 포착합니다.

### 트레이닝 설정에서 `wandb.run`을 사용해야 한다면 어떻게 하나요?

필요한 변수를 스스로 확장해야 합니다. 즉, 모든 프로세스에서 초기 조건이 동일한지 확인하는 것입니다.

```python
if os.environ.get("LOCAL_RANK", None) is None:
    os.environ["WANDB_DIR"] = wandb.run.dir
```

그러면 `os.environ["WANDB_DIR"]`을 사용하여 모델 체크포인트 디렉토리를 설정할 수 있습니다. 이를 통해, 비-제로 랭크 프로세스에서도 `wandb.run.dir`을 사용할 수 있습니다.