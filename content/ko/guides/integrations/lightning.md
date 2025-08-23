---
title: PyTorch Lightning
menu:
  default:
    identifier: ko-guides-integrations-lightning
    parent: integrations
weight: 340
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Optimize_PyTorch_Lightning_models_with_Weights_%26_Biases.ipynb" >}}

PyTorch Lightning은 PyTorch 코드를 구조화하고 분산 트레이닝, 16비트 정밀도 등과 같은 고급 기능을 손쉽게 추가할 수 있도록 가벼운 래퍼를 제공합니다. W&B도 ML 실험 로그를 위한 경량 래퍼를 제공하지만, 두 가지를 직접 결합할 필요 없이 [`WandbLogger`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)가 PyTorch Lightning 라이브러리에 직접 통합되어 있습니다.

## Lightning과 통합하기

{{< tabpane text=true >}}
{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer

wandb_logger = WandbLogger(log_model="all")
trainer = Trainer(logger=wandb_logger)
```

{{% alert %}}
**wandb.log() 사용 시:** `WandbLogger`는 Trainer의 `global_step`을 기준으로 W&B에 로그를 남깁니다. 코드에서 직접 `wandb.log`를 추가 호출할 경우, `wandb.log()`에서 `step` 인수를 **사용하지 마세요**.

대신 Trainer의 `global_step`과 함께 다른 메트릭처럼 로그하면 됩니다:

```python
wandb.log({"accuracy":0.99, "trainer/global_step": step})
```
{{% /alert %}}

{{% /tab %}}

{{% tab header="Fabric Logger" value="fabric" %}}

```python
import lightning as L
from wandb.integration.lightning.fabric import WandbLogger

wandb_logger = WandbLogger(log_model="all")
fabric = L.Fabric(loggers=[wandb_logger])
fabric.launch()
fabric.log_dict({"important_metric": important_metric})
```

{{% /tab %}}

{{< /tabpane >}}

{{< img src="/images/integrations/n6P7K4M.gif" alt="Interactive dashboards" >}}

### 가입 및 API 키 생성하기

API 키는 사용자의 머신이 W&B에 인증하는 데 사용됩니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
더 간편하게 API 키를 생성하려면 [W&B 인증 페이지](https://wandb.ai/authorize)로 바로 이동하세요. 표시되는 API 키를 복사해 비밀번호 관리 프로그램 등 안전한 위치에 저장합니다.
{{% /alert %}}

1. 오른쪽 상단의 사용자 프로필 아이콘을 클릭하세요.
1. **User Settings**를 선택하고 **API Keys** 섹션까지 스크롤합니다.
1. **Reveal**을 클릭하여 API 키를 복사하세요. 키를 숨기려면 페이지를 새로 고치세요.

### `wandb` 라이브러리 설치 및 로그인

로컬에 `wandb` 라이브러리를 설치하고 로그인하려면:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 본인의 API 키로 설정하세요.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` 라이브러리를 설치하고 로그인합니다.

    ```shell
    pip install wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}


## PyTorch Lightning의 `WandbLogger` 사용하기

PyTorch Lightning에는 다양한 `WandbLogger` 클래스가 있어 메트릭, 모델 가중치, 이미지 등 다양한 정보를 로그할 수 있습니다.

- [`PyTorch`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)
- [`Fabric`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)

Lightning과 통합하기 위해서는 `WandbLogger`를 인스턴스화하고, 이를 Lightning의 `Trainer`나 `Fabric`에 전달하면 됩니다.

{{< tabpane text=true >}}

{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
trainer = Trainer(logger=wandb_logger)
```

{{% /tab %}}

{{% tab header="Fabric Logger" value="fabric" %}}

```python
fabric = L.Fabric(loggers=[wandb_logger])
fabric.launch()
fabric.log_dict({
    "important_metric": important_metric
})
```

{{% /tab %}}

{{< /tabpane >}}


### 자주 쓰이는 Logger 인수들

다음은 `WandbLogger`에서 자주 사용하는 주요 파라미터들입니다. 전체 인수 설명은 PyTorch Lightning 공식 문서를 참고하세요.

- [`PyTorch`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)
- [`Fabric`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)

| 파라미터      | 설명                                                                               |
| ------------ | --------------------------------------------------------------------------------- |
| `project`     | 로그를 남길 wandb Project 지정                                                        |
| `name`        | wandb run에 이름 부여                                                                |
| `log_model`   | `log_model="all"`이면 모든 모델 저장, `log_model=True`이면 트레이닝 마지막에 저장                                      |
| `save_dir`    | 데이터가 저장되는 경로 지정                                                           |

## 하이퍼파라미터 로깅하기

{{< tabpane text=true >}}

{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
class LitModule(LightningModule):
    def __init__(self, *args, **kwarg):
        self.save_hyperparameters()
```

{{% /tab %}}

{{% tab header="Fabric Logger" value="fabric" %}}

```python
wandb_logger.log_hyperparams(
    {
        "hyperparameter_1": hyperparameter_1,
        "hyperparameter_2": hyperparameter_2,
    }
)
```

{{% /tab %}}

{{< /tabpane >}}

## 추가 config 파라미터 로깅

```python
# 파라미터 하나 추가
wandb_logger.experiment.config["key"] = value

# 여러 파라미터 추가
wandb_logger.experiment.config.update({key1: val1, key2: val2})

# wandb 모듈 직접 사용
wandb.config["key"] = value
wandb.config.update()
```

## gradient, 파라미터 히스토그램, 모델 구조 로깅

`wandblogger.watch()`에 모델 오브젝트를 넘기면 트레이닝 중 gradient와 파라미터를 모니터링 할 수 있습니다. 자세한 내용은 PyTorch Lightning의 `WandbLogger` 문서를 참고하세요.

## 메트릭 로깅

{{< tabpane text=true >}}

{{% tab header="PyTorch Logger" value="pytorch" %}}

`WandbLogger`를 사용할 때는 `LightningModule`의 `training_step` 또는 `validation_step` 메서드 같은 곳에서 `self.log('my_metric_name', metric_value)`로 메트릭을 기록할 수 있습니다.

아래 코드는 `LightningModule`에서 메트릭 및 하이퍼파라미터를 로깅하는 예시입니다. [`torchmetrics`](https://github.com/PyTorchLightning/metrics) 라이브러리를 사용하여 메트릭을 계산합니다.

```python
import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy
from lightning.pytorch import LightningModule


class My_LitModule(LightningModule):
    def __init__(self, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-3):
        """모델 파라미터 정의용 메소드"""
        super().__init__()

        # mnist 이미지는 (1, 28, 28) (채널, 너비, 높이)
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.layer_3 = Linear(n_layer_2, n_classes)

        self.loss = CrossEntropyLoss()
        self.lr = lr

        # 하이퍼파라미터를 self.hparams에 저장 (W&B에서 자동으로 로깅됨)
        self.save_hyperparameters()

    def forward(self, x):
        """입력 -> 출력 추론 메소드"""

        # (b, 1, 28, 28) -> (b, 1*28*28)
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        # 3번 (linear + relu)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        """각 배치마다 loss 반환"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # loss와 메트릭 로깅
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """메트릭 로깅용"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # loss와 메트릭 로깅
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds

    def configure_optimizers(self):
        """모델 옵티마이저 정의"""
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        """train/valid/test 단계별 편의 함수"""
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y)
        return preds, loss, acc
```

{{% /tab %}}

{{% tab header="Fabric Logger" value="fabric" %}}

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

{{% /tab %}}

{{< /tabpane >}}

## 메트릭의 최소/최대 값 로깅

wandb의 [`define_metric`]({{< relref path="/ref/python/sdk/classes/run.md#define_metric" lang="ko" >}}) 함수를 사용하면 W&B summary metric에 min, max, mean, best 등 다양한 통계를 표시할 수 있습니다. `define_metric`을 사용하지 않으면 마지막에 로깅된 값이 summary metric에 표시됩니다. 자세한 내용은 [`define_metric` 레퍼런스 문서]({{< relref path="/ref/python/sdk/classes/run.md#define_metric" lang="ko" >}})와 [가이드]({{< relref path="/guides/models/track/log/customize-logging-axes" lang="ko" >}})를 참고하세요.

W&B summary에서 최대 validation accuracy를 기록하려면, 트레이닝 시작 시 `wandb.define_metric`을 한 번만 호출하면 됩니다.

{{< tabpane text=true >}}
{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
class My_LitModule(LightningModule):
    ...

    def validation_step(self, batch, batch_idx):
        if trainer.global_step == 0:
            wandb.define_metric("val_accuracy", summary="max")

        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # loss와 메트릭 로깅
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds
```

{{% /tab %}}
{{% tab header="Fabric Logger" value="fabric" %}}

```python
wandb.define_metric("val_accuracy", summary="max")
fabric = L.Fabric(loggers=[wandb_logger])
fabric.launch()
fabric.log_dict({"val_accuracy": val_accuracy})
```

{{% /tab %}}
{{< /tabpane >}}

## 모델 체크포인트 저장

모델 체크포인트를 W&B [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})로 저장하려면,
Lightning의 [`ModelCheckpoint`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) 콜백과 `WandbLogger`의 `log_model` 인수를 설정하세요.

{{< tabpane text=true >}}

{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback])
```

{{% /tab %}}

{{% tab header="Fabric Logger" value="fabric" %}}

```python
fabric = L.Fabric(loggers=[wandb_logger], callbacks=[checkpoint_callback])
```

{{% /tab %}}

{{< /tabpane >}}

_최신(latest)_ 및 _최고(best)_ 에일리어스는 W&B [Artifact]({{< relref path="/guides/core/artifacts/" lang="ko" >}})로부터 손쉽게 모델 체크포인트를 불러올 수 있도록 자동으로 설정됩니다:

```python
# artifacts 패널에서 reference를 확인할 수 있습니다
# "VERSION"에는 버전(ex: "v2")이나 에일리어스("latest" 또는 "best")를 사용할 수 있습니다
checkpoint_reference = "USER/PROJECT/MODEL-RUN_ID:VERSION"
```

{{< tabpane text=true >}}
{{% tab header="Via Logger" value="logger" %}}

```python
# 체크포인트를 로컬에 다운로드 (이미 캐시된 경우는 건너뜀)
wandb_logger.download_artifact(checkpoint_reference, artifact_type="model")
```

{{% /tab %}}

{{% tab header="Via wandb" value="wandb" %}}

```python
# 체크포인트를 로컬에 다운로드 (이미 캐시된 경우는 건너뜀)
run = wandb.init(project="MNIST")
artifact = run.use_artifact(checkpoint_reference, type="model")
artifact_dir = artifact.download()
```

{{% /tab %}}
{{< /tabpane >}}

{{< tabpane text=true >}}
{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
# 체크포인트 불러오기
model = LitModule.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
```

{{% /tab %}}

{{% tab header="Fabric Logger" value="fabric" %}}

```python
# 원본 체크포인트 불러오기
full_checkpoint = fabric.load(Path(artifact_dir) / "model.ckpt")

model.load_state_dict(full_checkpoint["model"])
optimizer.load_state_dict(full_checkpoint["optimizer"])
```

{{% /tab %}}
{{< /tabpane >}}

로그한 모델 체크포인트는 [W&B Artifacts]({{< relref path="/guides/core/artifacts" lang="ko" >}}) UI를 통해 확인할 수 있습니다. 여기엔 전체 모델 계보가 포함되어 있습니다(예시: [여기에서 확인](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..)).

최고의 모델 체크포인트를 북마크하고 팀 전체에서 중앙 집중식으로 관리하려면, [W&B Model Registry]({{< relref path="/guides/models" lang="ko" >}})에 연결하세요.

여기서 작업별로 모델을 정리하고, 라이프사이클을 관리하며, ML 라이프사이클 전반에 걸쳐 손쉬운 추적 및 감사가 가능하고, 웹훅이나 잡을 통해 [자동화]({{< relref path="/guides/core/automations/" lang="ko" >}})도 할 수 있습니다.

## 이미지, 텍스트 등 다양한 미디어 로깅

`WandbLogger`에는 `log_image`, `log_text`, `log_table` 등 미디어 로그를 위한 메소드가 제공됩니다.

또한 `wandb.log`나 `trainer.logger.experiment.log`를 직접 호출하여 오디오, 분자, 포인트 클라우드, 3D 오브젝트 등 다양한 미디어를 기록할 수 있습니다.

{{< tabpane text=true >}}

{{% tab header="Log Images" value="images" %}}

```python
# tensor, numpy array, PIL 이미지 모두 사용 가능
wandb_logger.log_image(key="samples", images=[img1, img2])

# 캡션 추가
wandb_logger.log_image(key="samples", images=[img1, img2], caption=["tree", "person"])

# 파일 경로 사용
wandb_logger.log_image(key="samples", images=["img_1.jpg", "img_2.jpg"])

# trainer에서 .log 사용
trainer.logger.experiment.log(
    {"samples": [wandb.Image(img, caption=caption) for (img, caption) in my_images]},
    step=current_trainer_global_step,
)
```

{{% /tab %}}

{{% tab header="Log Text" value="text" %}}

```python
# 데이터는 리스트의 리스트 형식이어야 함
columns = ["input", "label", "prediction"]
my_data = [["cheese", "english", "english"], ["fromage", "french", "spanish"]]

# columns와 data로 로그
wandb_logger.log_text(key="my_samples", columns=columns, data=my_data)

# pandas DataFrame 사용
wandb_logger.log_text(key="my_samples", dataframe=my_dataframe)
```

{{% /tab %}}

{{% tab header="Log Tables" value="tables" %}}

```python
# W&B Table을 로그 (캡션, 이미지, 오디오 포함)
columns = ["caption", "image", "sound"]

# 데이터는 리스트의 리스트 형식이어야 함
my_data = [
    ["cheese", wandb.Image(img_1), wandb.Audio(snd_1)],
    ["wine", wandb.Image(img_2), wandb.Audio(snd_2)],
]

# Table 로그
wandb_logger.log_table(key="my_samples", columns=columns, data=data)
```

{{% /tab %}}

{{< /tabpane >}}

Lightning의 Callbacks 시스템을 사용해 `WandbLogger`로 W&B에 로그를 남기는 시점을 직접 제어할 수 있습니다. 아래는 검증 이미지 및 예측 샘플을 로그하는 예시입니다:

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
        """validation 배치가 끝날 때 호출됩니다."""

        # `outputs`는 `LightningModule.validation_step`에서 반환된 값으로
        # 여기서는 모델의 예측값에 해당합니다

        # 첫 번째 배치에서 샘플 이미지 20개와 예측값을 로그
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [
                f"Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(y[:n], outputs[:n])
            ]

            # 옵션 1: `WandbLogger.log_image` 사용
            wandb_logger.log_image(key="sample_images", images=images, caption=captions)

            # 옵션 2: 이미지와 예측을 W&B Table로 로그
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred] or x_i,
                y_i,
                y_pred in list(zip(x[:n], y[:n], outputs[:n])),
            ]
            wandb_logger.log_table(key="sample_table", columns=columns, data=data)


trainer = pl.Trainer(callbacks=[LogPredictionSamplesCallback()])
```

## Lightning과 W&B를 활용한 멀티-GPU 사용하기

PyTorch Lightning은 DDP 인터페이스를 통해 멀티-GPU를 지원합니다. 하지만 PyTorch Lightning의 구조상 GPU 인스턴스화 방식에 주의해야 합니다.

Lightning은 트레이닝 루프 내의 각 GPU(=Rank)가 동일한 초기 조건으로 인스턴스화되어야 한다고 가정합니다. 그러나 rank 0 프로세스만이 `wandb.run` 오브젝트에 접근할 수 있고, 나머지 rank는 `wandb.run = None`이 됩니다. 이로 인해 rank가 0이 아닌 프로세스가 실패할 수 있습니다. 이런 상황에서는 rank 0 프로세스가 나머지 프로세스의 조인을 기다리다가 **데드락**에 빠질 수 있습니다.

따라서 트레이닝 코드를 구성할 때 `wandb.run` 오브젝트에 의존하지 않도록 주의하세요.

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
    # 모든 랜덤 시드를 동일하게 설정
    # 분산 트레이닝 환경에서 중요합니다.
    # 각 rank는 고유의 초기 가중치를 받으며,
    # 일치하지 않으면 gradient 역시 불일치해 학습이 잘 되지 않을 수 있습니다.
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

## 예시

[Colab 노트북과 비디오 튜토리얼](https://wandb.me/lit-colab)에서 직접 따라해 보실 수 있습니다.

## 자주 묻는 질문(FAQ)

### W&B는 Lightning과 어떻게 통합되나요?

핵심 통합은 [Lightning `loggers` API](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html)를 기반으로 하며, 이를 통해 프레임워크에 종속적이지 않은 방식으로 많은 로깅 코드를 작성할 수 있습니다. `Logger`는 [Lightning `Trainer`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)에 전달되며, 이 API의 강력한 [hook-및-콜백 시스템](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html)을 기반으로 동작합니다. 이를 통해 연구 코드와 엔지니어링/로깅 코드를 분리할 수 있습니다.

### 별도의 코드 추가 없이도 통합에서 어떤 정보를 로깅하나요?

모델 체크포인트를 W&B에 저장해서 나중에 확인하거나 다른 run에서 다운로드할 수 있습니다. 또한 [시스템 메트릭]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ko" >}})(GPU 사용량, 네트워크 I/O 등), 하드웨어/운영체제 등 환경 정보, [코드 상태]({{< relref path="/guides/models/app/features/panels/code.md" lang="ko" >}}) (git 커밋/패치, 노트북 내용/세션 히스토리 등), 표준 출력에 인쇄된 모든 내용까지 자동으로 기록합니다.

### 트레이닝 코드에서 꼭 `wandb.run`을 써야 할 경우에는 어떻게 하나요?

엑세스하려는 변수를 직접 스코프 확장해서 사용해야 합니다. 즉, 모든 프로세스에서 초기 조건이 동일하도록 만들어 주세요.

```python
if os.environ.get("LOCAL_RANK", None) is None:
    os.environ["WANDB_DIR"] = wandb.run.dir
```

이렇게 하면 `os.environ["WANDB_DIR"]`를 사용해 모델 체크포인트 디렉토리를 지정할 수 있습니다. 이 방식으로는 rank가 0이 아닌 프로세스도 `wandb.run.dir`에 엑세스할 수 있습니다.