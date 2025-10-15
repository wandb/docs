---
title: PyTorch Lightning
menu:
  default:
    identifier: ko-guides-integrations-lightning
    parent: integrations
weight: 340
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Optimize_PyTorch_Lightning_models_with_Weights_%26_Biases.ipynb" >}}

PyTorch Lightning 은 PyTorch 코드를 구성하고 분산 트레이닝 및 16비트 정밀도와 같은 고급 기능을 쉽게 추가할 수 있는 가벼운 래퍼를 제공합니다. W&B는 ML Experiments을 로깅하기 위한 가벼운 래퍼를 제공합니다. 하지만 이 둘을 직접 결합할 필요는 없습니다. Weights & Biases는 [**`WandbLogger`**](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)를 통해 PyTorch Lightning 라이브러리에 직접 통합됩니다.

## Lightning 과 통합

{{< tabpane text=true >}}
{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer

wandb_logger = WandbLogger(log_model="all")
trainer = Trainer(logger=wandb_logger)
```

{{% alert %}}
**wandb.log() 사용:** `WandbLogger`는 Trainer의 `global_step`을 사용하여 W&B에 로깅합니다. 코드에서 `wandb.log`를 직접 추가로 호출하는 경우 `wandb.log()`에서 `step` 인수를 **사용하지 마십시오**.

대신 다른 메트릭과 마찬가지로 Trainer의 `global_step`을 로깅합니다.

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

{{< img src="/images/integrations/n6P7K4M.gif" alt="Interactive dashboards accessible anywhere, and more!" >}}

### 가입 및 API 키 생성

API 키는 사용자의 머신을 W&B에 인증합니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
보다 간소화된 접근 방식을 위해 [https://wandb.ai/authorize](https://wandb.ai/authorize)로 직접 이동하여 API 키를 생성할 수 있습니다. 표시된 API 키를 복사하여 비밀번호 관리자와 같은 안전한 위치에 저장하십시오.
{{% /alert %}}

1. 오른쪽 상단에서 사용자 프로필 아이콘을 클릭합니다.
2. **사용자 설정**을 선택한 다음 **API 키** 섹션으로 스크롤합니다.
3. **Reveal**을 클릭합니다. 표시된 API 키를 복사합니다. API 키를 숨기려면 페이지를 새로 고칩니다.

### `wandb` 라이브러리 설치 및 로그인

`wandb` 라이브러리를 로컬에 설치하고 로그인하려면:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. API 키로 `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 설정합니다.

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


## PyTorch Lightning의 `WandbLogger` 사용

PyTorch Lightning에는 메트릭 및 모델 weights, 미디어 등을 로깅하는 여러 `WandbLogger` 클래스가 있습니다.

- [`PyTorch`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)
- [`Fabric`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)

Lightning과 통합하려면 WandbLogger를 인스턴스화하고 Lightning의 `Trainer` 또는 `Fabric`에 전달합니다.

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


### 일반적인 로거 인수

다음은 WandbLogger에서 가장 많이 사용되는 파라미터 중 일부입니다. 모든 로거 인수에 대한 자세한 내용은 PyTorch Lightning 설명서를 검토하십시오.

- [`PyTorch`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)
- [`Fabric`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)

| 파라미터   | 설명                                                                           |
| ----------- | ------------------------------------------------------------------------------ |
| `project`   | 로깅할 wandb Project를 정의합니다.                                                |
| `name`      | wandb run에 이름을 지정합니다.                                                  |
| `log_model` | `log_model="all"`인 경우 모든 모델을 로깅하고 `log_model=True`인 경우 트레이닝 종료 시 로깅합니다. |
| `save_dir`  | 데이터가 저장되는 경로입니다.                                                     |

## 하이퍼파라미터 로깅

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

## 추가 구성 파라미터 로깅

```python
# 파라미터 하나 추가
wandb_logger.experiment.config["key"] = value

# 여러 파라미터 추가
wandb_logger.experiment.config.update({key1: val1, key2: val2})

# wandb 모듈을 직접 사용
wandb.config["key"] = value
wandb.config.update()
```

## 그레이디언트, 파라미터 히스토그램 및 모델 토폴로지 로깅

모델 오브젝트를 `wandblogger.watch()`에 전달하여 트레이닝하는 동안 모델의 그레이디언트와 파라미터를 모니터링할 수 있습니다. PyTorch Lightning `WandbLogger` 설명서를 참조하십시오.

## 메트릭 로깅

{{< tabpane text=true >}}

{{% tab header="PyTorch Logger" value="pytorch" %}}

`training_step` 또는 `validation_step methods`와 같이 `LightningModule` 내에서 `self.log('my_metric_name', metric_vale)`을 호출하여 `WandbLogger`를 사용할 때 메트릭을 W&B에 로깅할 수 있습니다.

아래 코드 조각은 메트릭과 `LightningModule` 하이퍼파라미터를 로깅하도록 `LightningModule`을 정의하는 방법을 보여줍니다. 이 예제에서는 [`torchmetrics`](https://github.com/PyTorchLightning/metrics) 라이브러리를 사용하여 메트릭을 계산합니다.

```python
import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy
from lightning.pytorch import LightningModule


class My_LitModule(LightningModule):
    def __init__(self, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-3):
        """모델 파라미터를 정의하는 데 사용되는 메소드"""
        super().__init__()

        # mnist 이미지는 (1, 28, 28) (채널, 너비, 높이)입니다.
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.layer_3 = Linear(n_layer_2, n_classes)

        self.loss = CrossEntropyLoss()
        self.lr = lr

        # 하이퍼파라미터를 self.hparams에 저장합니다(W&B에서 자동 로깅).
        self.save_hyperparameters()

    def forward(self, x):
        """추론 입력 -> 출력에 사용되는 메소드"""

        # (b, 1, 28, 28) -> (b, 1*28*28)
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        # 3 x (linear + relu)를 수행해 보겠습니다.
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        """단일 배치에서 손실을 반환해야 합니다."""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # 손실 및 메트릭 로깅
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """메트릭 로깅에 사용"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # 손실 및 메트릭 로깅
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds

    def configure_optimizers(self):
        """모델 옵티마이저를 정의합니다."""
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        """train/valid/test 단계가 유사하므로 편의 함수"""
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

## 메트릭의 최소/최대값 로깅

wandb의 [`define_metric`]({{< relref path="/ref/python/run#define_metric" lang="ko" >}}) 함수를 사용하여 W&B 요약 메트릭에 해당 메트릭의 최소, 최대, 평균 또는 최적값을 표시할지 여부를 정의할 수 있습니다. `define`_`metric` _이(가) 사용되지 않으면 로깅된 마지막 값이 요약 메트릭에 나타납니다. `define_metric` [참조 문서(여기)]({{< relref path="/ref/python/run#define_metric" lang="ko" >}})와 [가이드(여기)]({{< relref path="/guides/models/track/log/customize-logging-axes" lang="ko" >}})를 참조하십시오.

W&B가 W&B 요약 메트릭에서 최대 검증 정확도를 추적하도록 지시하려면 트레이닝 시작 시 `wandb.define_metric`을 한 번만 호출하십시오.

{{< tabpane text=true >}}
{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
class My_LitModule(LightningModule):
    ...

    def validation_step(self, batch, batch_idx):
        if trainer.global_step == 0:
            wandb.define_metric("val_accuracy", summary="max")

        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # 손실 및 메트릭 로깅
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

## 모델 체크포인트

모델 체크포인트를 W&B [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})로 저장하려면 Lightning [`ModelCheckpoint`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) 콜백을 사용하고 `WandbLogger`에서 `log_model` 인수를 설정합니다.

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

_latest_ 및 _best_ 에일리어스는 W&B [Artifact]({{< relref path="/guides/core/artifacts/" lang="ko" >}})에서 모델 체크포인트를 쉽게 검색할 수 있도록 자동으로 설정됩니다.

```python
# 아티팩트 패널에서 레퍼런스를 검색할 수 있습니다.
# "VERSION"은 버전("v2" 등) 또는 에일리어스("latest" 또는 "best")일 수 있습니다.
checkpoint_reference = "USER/PROJECT/MODEL-RUN_ID:VERSION"
```

{{< tabpane text=true >}}
{{% tab header="Via Logger" value="logger" %}}

```python
# 체크포인트를 로컬로 다운로드합니다(이미 캐시되지 않은 경우).
wandb_logger.download_artifact(checkpoint_reference, artifact_type="model")
```

{{% /tab %}}

{{% tab header="Via wandb" value="wandb" %}}

```python
# 체크포인트를 로컬로 다운로드합니다(이미 캐시되지 않은 경우).
run = wandb.init(project="MNIST")
artifact = run.use_artifact(checkpoint_reference, type="model")
artifact_dir = artifact.download()
```

{{% /tab %}}
{{< /tabpane >}}

{{< tabpane text=true >}}
{{% tab header="PyTorch Logger" value="pytorch" %}}

```python
# 체크포인트 로드
model = LitModule.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
```

{{% /tab %}}

{{% tab header="Fabric Logger" value="fabric" %}}

```python
# 원시 체크포인트 요청
full_checkpoint = fabric.load(Path(artifact_dir) / "model.ckpt")

model.load_state_dict(full_checkpoint["model"])
optimizer.load_state_dict(full_checkpoint["optimizer"])
```

{{% /tab %}}
{{< /tabpane >}}

로깅하는 모델 체크포인트는 [W&B Artifacts]({{< relref path="/guides/core/artifacts" lang="ko" >}}) UI를 통해 볼 수 있으며 전체 모델 계보를 포함합니다(UI의 예제 모델 체크포인트는 [여기](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..) 참조).

최고의 모델 체크포인트를 북마크하고 팀 전체에서 중앙 집중화하려면 [W&B Model Registry]({{< relref path="/guides/models" lang="ko" >}})에 연결하면 됩니다.

여기에서 작업별로 최고의 모델을 구성하고, 모델 수명 주기를 관리하고, ML 수명 주기 전반에 걸쳐 쉬운 추적 및 감사를 용이하게 하고, 웹후크 또는 작업을 통해 다운스트림 작업을 [자동화]({{< relref path="/guides/core/automations/" lang="ko" >}})할 수 있습니다.

## 이미지, 텍스트 등 로깅

`WandbLogger`에는 미디어를 로깅하기 위한 `log_image`, `log_text` 및 `log_table` 메소드가 있습니다.

`wandb.log` 또는 `trainer.logger.experiment.log`를 직접 호출하여 오디오, 분자, 포인트 클라우드, 3D 오브젝트 등과 같은 다른 미디어 유형을 로깅할 수도 있습니다.

{{< tabpane text=true >}}

{{% tab header="Log Images" value="images" %}}

```python
# 텐서, numpy 어레이 또는 PIL 이미지 사용
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

{{% /tab %}}

{{% tab header="Log Text" value="text" %}}

```python
# 데이터는 목록 목록이어야 합니다.
columns = ["input", "label", "prediction"]
my_data = [["cheese", "english", "english"], ["fromage", "french", "spanish"]]

# 열 및 데이터 사용
wandb_logger.log_text(key="my_samples", columns=columns, data=my_data)

# pandas DataFrame 사용
wandb_logger.log_text(key="my_samples", dataframe=my_dataframe)
```

{{% /tab %}}

{{% tab header="Log Tables" value="tables" %}}

```python
# 텍스트 캡션, 이미지 및 오디오가 있는 W&B Table 로깅
columns = ["caption", "image", "sound"]

# 데이터는 목록 목록이어야 합니다.
my_data = [
    ["cheese", wandb.Image(img_1), wandb.Audio(snd_1)],
    ["wine", wandb.Image(img_2), wandb.Audio(snd_2)],
]

# Table 로깅
wandb_logger.log_table(key="my_samples", columns=columns, data=data)
```

{{% /tab %}}

{{< /tabpane >}}

Lightning의 콜백 시스템을 사용하여 WandbLogger를 통해 Weights & Biases에 로깅하는 시점을 제어할 수 있습니다. 이 예에서는 검증 이미지 및 예측 샘플을 로깅합니다.


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
        """검증 배치가 종료되면 호출됩니다."""

        # `outputs`는 `LightningModule.validation_step`에서 가져옵니다.
        # 이 경우 모델 예측에 해당합니다.

        # 첫 번째 배치에서 20개의 샘플 이미지 예측을 로깅해 보겠습니다.
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [
                f"Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(y[:n], outputs[:n])
            ]

            # 옵션 1: `WandbLogger.log_image`로 이미지 로깅
            wandb_logger.log_image(key="sample_images", images=images, caption=captions)

            # 옵션 2: 이미지 및 예측을 W&B Table로 로깅
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred] or x_i,
                y_i,
                y_pred in list(zip(x[:n], y[:n], outputs[:n])),
            ]
            wandb_logger.log_table(key="sample_table", columns=columns, data=data)


trainer = pl.Trainer(callbacks=[LogPredictionSamplesCallback()])
```

## Lightning 및 W&B로 여러 GPU 사용

PyTorch Lightning은 DDP 인터페이스를 통해 Multi-GPU를 지원합니다. 그러나 PyTorch Lightning의 디자인에서는 GPU를 인스턴스화하는 방법에 주의해야 합니다.

Lightning은 트레이닝 루프의 각 GPU(또는 순위)가 동일한 초기 조건으로 정확히 동일한 방식으로 인스턴스화되어야 한다고 가정합니다. 그러나 순위 0 프로세스만 `wandb.run` 오브젝트에 엑세스할 수 있으며 0이 아닌 순위 프로세스의 경우 `wandb.run = None`입니다. 이로 인해 0이 아닌 프로세스가 실패할 수 있습니다. 이러한 상황은 순위 0 프로세스가 이미 충돌한 0이 아닌 순위 프로세스가 조인될 때까지 대기하기 때문에 **교착 상태**에 놓일 수 있습니다.

이러한 이유로 트레이닝 코드를 설정하는 방법에 주의하십시오. 설정하는 데 권장되는 방법은 코드가 `wandb.run` 오브젝트와 독립적이도록 하는 것입니다.

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
    # 이는 분산 트레이닝 설정에서 중요합니다.
    # 각 순위는 자체 초기 weights 집합을 가져옵니다.
    # 일치하지 않으면 그레이디언트도 일치하지 않습니다.
    # 수렴하지 않을 수 있는 트레이닝으로 이어집니다.
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



## 예제

Colab [여기](https://wandb.me/lit-colab)에서 비디오 튜토리얼을 따라갈 수 있습니다.

## 자주 묻는 질문

### W&B는 Lightning과 어떻게 통합됩니까?

코어 통합은 프레임워크에 구애받지 않는 방식으로 대부분의 로깅 코드를 작성할 수 있도록 해주는 [Lightning `loggers` API](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html)를 기반으로 합니다. `Logger`는 [Lightning `Trainer`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)에 전달되고 해당 API의 풍부한 [훅 및 콜백 시스템](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html)을 기반으로 트리거됩니다. 이를 통해 연구 코드는 엔지니어링 및 로깅 코드와 잘 분리됩니다.

### 추가 코드 없이 통합 로깅은 무엇입니까?

모델 체크포인트를 W&B에 저장하여 볼 수 있거나 향후 run에서 사용하기 위해 다운로드할 수 있습니다. 또한 GPU 사용량 및 네트워크 I/O와 같은 [시스템 메트릭](/ref/system-metrics), 하드웨어 및 OS 정보와 같은 환경 정보, [코드 상태]({{< relref path="/guides/models/app/features/panels/code.md" lang="ko" >}}) (git 커밋 및 차이 패치, 노트북 콘텐츠 및 세션 기록 포함) 및 표준 출력에 인쇄된 모든 항목을 캡처합니다.

### 트레이닝 설정에서 `wandb.run`을 사용해야 하는 경우는 어떻게 해야 합니까?

엑세스해야 하는 변수의 범위를 직접 확장해야 합니다. 즉, 모든 프로세스에서 초기 조건이 동일한지 확인하십시오.

```python
if os.environ.get("LOCAL_RANK", None) is None:
    os.environ["WANDB_DIR"] = wandb.run.dir
```

그렇다면 `os.environ["WANDB_DIR"]`을 사용하여 모델 체크포인트 디렉토리를 설정할 수 있습니다. 이렇게 하면 0이 아닌 모든 순위 프로세스가 `wandb.run.dir`에 엑세스할 수 있습니다.
```