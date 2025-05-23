---
title: PyTorch Lightning
menu:
  tutorials:
    identifier: ko-tutorials-integration-tutorials-lightning
    parent: integration-tutorials
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Image_Classification_using_PyTorch_Lightning.ipynb" >}}
PyTorch Lightning를 사용하여 이미지 분류 파이프라인을 구축합니다. 코드의 가독성과 재현성을 높이기 위해 이 [스타일 가이드](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html)를 따릅니다. 이에 대한 멋진 설명은 [여기](https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY)에서 확인할 수 있습니다.

## PyTorch Lightning 및 W&B 설정

이 튜토리얼에서는 PyTorch Lightning와 Weights & Biases가 필요합니다.

```shell
pip install lightning -q
pip install wandb -qU
```

```python
import lightning.pytorch as pl

# 당신이 가장 선호하는 기계 학습 추적 툴
from lightning.pytorch.loggers import WandbLogger

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader

from torchmetrics import Accuracy

from torchvision import transforms
from torchvision.datasets import CIFAR10

import wandb
```

이제 wandb 계정에 로그인해야 합니다.

```
wandb.login()
```

## DataModule - 우리가 원하는 데이터 파이프라인

DataModule은 데이터 관련 훅을 LightningModule에서 분리하여 데이터셋에 구애받지 않는 모델을 개발할 수 있도록 하는 방법입니다.

데이터 파이프라인을 하나의 공유 및 재사용 가능한 클래스로 구성합니다. DataModule은 PyTorch에서 데이터 처리와 관련된 5단계를 캡슐화합니다.
- 다운로드 / 토큰화 / 처리.
- 정리하고 (선택적으로) 디스크에 저장.
- 데이터셋 내부에 로드.
- 변환 적용 (회전, 토큰화 등...).
- DataLoader 내부에 래핑.

DataModule에 대해 자세히 알아보려면 [여기](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)를 참조하세요. Cifar-10 데이터셋을 위한 DataModule을 구축해 보겠습니다.

```
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.num_classes = 10
    
    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        # dataloader에서 사용할 train/val 데이터셋 할당
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # dataloader에서 사용할 테스트 데이터셋 할당
        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)
```

## 콜백

콜백은 프로젝트 전반에서 재사용할 수 있는 독립적인 프로그램입니다. PyTorch Lightning에는 정기적으로 사용되는 몇 가지 [기본 제공 콜백](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html#built-in-callbacks)이 함께 제공됩니다. PyTorch Lightning의 콜백에 대해 자세히 알아보려면 [여기](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html)를 참조하세요.

### 내장 콜백

이 튜토리얼에서는 [Early Stopping](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.callbacks.EarlyStopping) 및 [Model Checkpoint](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) 내장 콜백을 사용합니다. 이는 `Trainer`에 전달될 수 있습니다.

### 사용자 지정 콜백
사용자 지정 Keras 콜백에 익숙하다면 PyTorch 파이프라인에서 동일한 작업을 수행할 수 있다는 것은 금상첨화입니다.

이미지 분류를 수행하므로 이미지 샘플에 대한 모델의 예측을 시각화할 수 있는 기능이 유용할 수 있습니다. 콜백 형태의 이러한 기능은 초기 단계에서 모델을 디버그하는 데 도움이 될 수 있습니다.

```
class ImagePredictionLogger(pl.callbacks.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # 텐서를 CPU로 가져오기
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # 모델 예측 가져오기
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # 이미지를 wandb Image로 기록
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
            })
        
```

## LightningModule - 시스템 정의

LightningModule은 모델이 아닌 시스템을 정의합니다. 여기서 시스템은 모든 연구 코드를 단일 클래스로 그룹화하여 독립적으로 만듭니다. `LightningModule`은 PyTorch 코드를 5개의 섹션으로 구성합니다.
- 계산 (`__init__`).
- 트레이닝 루프 (`training_step`)
- 검증 루프 (`validation_step`)
- 테스트 루프 (`test_step`)
- 옵티마이저 (`configure_optimizers`)

따라서 쉽게 공유할 수 있는 데이터셋에 구애받지 않는 모델을 구축할 수 있습니다. Cifar-10 분류를 위한 시스템을 구축해 보겠습니다.

```
class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4):
        super().__init__()
        
        # 하이퍼파라미터 기록
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)

        self.pool1 = torch.nn.MaxPool2d(2)
        self.pool2 = torch.nn.MaxPool2d(2)
        
        n_sizes = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_sizes, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)

    # conv 블록에서 Linear 레이어로 들어가는 출력 텐서의 크기를 반환합니다.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # conv 블록에서 특징 텐서를 반환합니다.
    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x
    
    # 추론 중에 사용됩니다.
    def forward(self, x):
       x = self._forward_features(x)
       x = x.view(x.size(0), -1)
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = F.log_softmax(self.fc3(x), dim=1)
       
       return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        # 트레이닝 메트릭
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # 검증 메트릭
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        # 검증 메트릭
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

```

## 트레이닝 및 평가

이제 `DataModule`을 사용하여 데이터 파이프라인을 구성하고 `LightningModule`을 사용하여 모델 아키텍처 + 트레이닝 루프를 구성했으므로 PyTorch Lightning `Trainer`가 다른 모든 것을 자동화합니다.

Trainer는 다음을 자동화합니다.
- 에포크 및 배치 반복
- `optimizer.step()`, `backward`, `zero_grad()` 호출
- `.eval()` 호출, grads 활성화/비활성화
- 가중치 저장 및 로드
- Weights & Biases 로깅
- 다중 GPU 트레이닝 지원
- TPU 지원
- 16비트 트레이닝 지원

```
dm = CIFAR10DataModule(batch_size=32)
# x_dataloader에 액세스하려면 prepare_data 및 setup을 호출해야 합니다.
dm.prepare_data()
dm.setup()

# 이미지 예측을 기록하기 위해 사용자 지정 ImagePredictionLogger 콜백에 필요한 샘플입니다.
val_samples = next(iter(dm.val_dataloader()))
val_imgs, val_labels = val_samples[0], val_samples[1]
val_imgs.shape, val_labels.shape
```

```
model = LitModel((3, 32, 32), dm.num_classes)

# wandb 로거 초기화
wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

# 콜백 초기화
early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
checkpoint_callback = pl.callbacks.ModelCheckpoint()

# 트레이너 초기화
trainer = pl.Trainer(max_epochs=2,
                     logger=wandb_logger,
                     callbacks=[early_stop_callback,
                                ImagePredictionLogger(val_samples),
                                checkpoint_callback],
                     )

# 모델 트레이닝
trainer.fit(model, dm)

# 보류된 테스트 세트에서 모델 평가 ⚡⚡
trainer.test(dataloaders=dm.test_dataloader())

# wandb run 닫기
wandb.finish()
```

## 마지막 생각
저는 TensorFlow/Keras 에코시스템에서 왔으며 PyTorch가 우아한 프레임워크임에도 불구하고 약간 부담스럽다고 생각합니다. 그냥 제 개인적인 경험입니다. PyTorch Lightning를 탐색하면서 저를 PyTorch에서 멀어지게 했던 거의 모든 이유가 해결되었다는 것을 깨달았습니다. 제 흥분에 대한 간략한 요약은 다음과 같습니다.
- 이전: 기존 PyTorch 모델 정의는 모든 곳에 흩어져 있었습니다. 일부 `model.py` 스크립트의 모델과 `train.py` 파일의 트레이닝 루프를 사용했습니다. 파이프라인을 이해하기 위해 앞뒤로 많이 살펴봐야 했습니다.
- 현재: `LightningModule`은 모델이 `training_step`, `validation_step` 등과 함께 정의되는 시스템 역할을 합니다. 이제 모듈화되고 공유 가능합니다.
- 이전: TensorFlow/Keras의 가장 좋은 부분은 입력 데이터 파이프라인입니다. 해당 데이터셋 카탈로그는 풍부하고 성장하고 있습니다. PyTorch의 데이터 파이프라인은 가장 큰 문제점이었습니다. 일반적인 PyTorch 코드에서 데이터 다운로드/정리/준비는 일반적으로 여러 파일에 흩어져 있습니다.
- 현재: DataModule은 데이터 파이프라인을 하나의 공유 및 재사용 가능한 클래스로 구성합니다. 이는 필요한 변환 및 데이터 처리/다운로드 단계와 함께 `train_dataloader`, `val_dataloader`(s), `test_dataloader`(s)의 모음일 뿐입니다.
- 이전: Keras를 사용하면 `model.fit`을 호출하여 모델을 트레이닝하고 `model.predict`를 호출하여 추론을 실행할 수 있습니다. `model.evaluate`는 테스트 데이터에 대한 간단한 평가를 제공했습니다. PyTorch에서는 그렇지 않습니다. 일반적으로 별도의 `train.py` 및 `test.py` 파일을 찾을 수 있습니다.
- 현재: `LightningModule`을 사용하면 `Trainer`가 모든 것을 자동화합니다. 모델을 트레이닝하고 평가하려면 `trainer.fit` 및 `trainer.test`를 호출하기만 하면 됩니다.
- 이전: TensorFlow는 TPU를 좋아하고 PyTorch는...
- 현재: PyTorch Lightning를 사용하면 여러 GPU와 심지어 TPU에서도 동일한 모델을 쉽게 트레이닝할 수 있습니다.
- 이전: 저는 콜백의 큰 팬이며 사용자 지정 콜백을 작성하는 것을 선호합니다. Early Stopping과 같이 사소한 것도 기존 PyTorch와의 논의 대상이었습니다.
- 현재: PyTorch Lightning를 사용하면 Early Stopping 및 Model Checkpointing이 매우 쉽습니다. 사용자 지정 콜백을 작성할 수도 있습니다.

## 🎨 결론 및 리소스

이 리포트가 도움이 되었기를 바랍니다. 코드를 가지고 놀고 원하는 데이터셋으로 이미지 분류기를 트레이닝하는 것이 좋습니다.

PyTorch Lightning에 대해 자세히 알아볼 수 있는 몇 가지 리소스는 다음과 같습니다.
- [단계별 연습](https://lightning.ai/docs/pytorch/latest/starter/introduction.html) - 이것은 공식 튜토리얼 중 하나입니다. 해당 문서는 정말 잘 작성되어 있으며 훌륭한 학습 리소스로 적극 권장합니다.
- [Weights & Biases와 함께 Pytorch Lightning 사용](https://wandb.me/lightning) - W&B와 함께 PyTorch Lightning를 사용하는 방법에 대해 자세히 알아보기 위해 실행할 수 있는 빠른 colab입니다.
