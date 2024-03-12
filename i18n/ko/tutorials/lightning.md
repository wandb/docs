
# PyTorch Lightning

[**Colab 노트북에서 시도해 보세요 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Image_Classification_using_PyTorch_Lightning.ipynb)

PyTorch Lightning을 사용하여 이미지 분류 파이프라인을 구축할 것입니다. 우리는 코드의 가독성과 재현성을 높이기 위해 이 [스타일 가이드](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html)를 따를 것입니다. 이에 대한 멋진 설명은 [여기](https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY)에서 확인할 수 있습니다.

## PyTorch Lightning과 W&B 설정하기

이 튜토리얼을 위해 PyTorch Lightning(뻔하지 않나요!)과 Weights & Biases가 필요합니다.


```
!pip install lightning -q
# weights and biases 설치
!pip install wandb -qU
```

이제 이런 임포트가 필요합니다.



```
import lightning.pytorch as pl
# 당신이 가장 좋아하는 기계학습 추적 툴
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

## 🔧 DataModule - 우리가 바라는 데이터 파이프라인

DataModules는 LightningModule에서 데이터 관련 훅을 분리하여 데이터셋에 구애받지 않는 모델을 개발할 수 있는 방법입니다.

이는 데이터 파이프라인을 하나의 공유 가능하고 재사용 가능한 클래스로 조직화합니다. DataModule은 PyTorch에서 데이터 처리에 관련된 5단계를 캡슐화합니다:
- 다운로드 / 토큰화 / 처리.
- 정리하고 (아마도) 디스크에 저장.
- Dataset 안에 로드하기.
- 변형 적용하기 (회전, 토큰화 등…).
- DataLoader 안에 랩핑하기.

datamodules에 대해 더 알아보려면 [여기](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)를 참조하세요. Cifar-10 데이터셋을 위한 DataModule을 구축해봅시다.


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
        # 데이터로더에서 사용할 train/val 데이터셋 할당
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # 데이터로더에서 사용할 테스트 데이터셋 할당
        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)
```

## 📱 콜백

콜백은 프로젝트 간에 재사용할 수 있는 독립적인 프로그램입니다. PyTorch Lightning은 정기적으로 사용되는 몇 가지 [내장 콜백](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html#built-in-callbacks)을 제공합니다.
PyTorch Lightning에서 콜백에 대해 더 알아보려면 [여기](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html)를 참조하세요.

### 내장 콜백

이 튜토리얼에서는 `Trainer`에 전달할 수 있는 [Early Stopping](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.callbacks.EarlyStopping)과 [Model Checkpoint](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) 내장 콜백을 사용할 것입니다.

### 맞춤 콜백
Custom Keras 콜백에 익숙하다면, PyTorch 파이프라인에서 동일한 작업을 수행할 수 있는 기능이 있습니다.

이미지 분류를 수행하고 있기 때문에, 이미지 몇 가지 샘플에 대한 모델의 예측값을 시각화하는 기능이 유용할 수 있습니다. 이를 콜백 형태로 구현하면 초기 단계에서 모델을 디버깅하는 데 도움이 될 수 있습니다.


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
        # 모델 예측 얻기
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # wandb Image로 이미지 로그하기
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
            })
        
```

## 🎺 LightningModule - 시스템 정의하기

LightningModule은 모델이 아닌 시스템을 정의합니다. 여기서 시스템은 모든 연구 코드를 하나의 클래스로 그룹화하여 자체적으로 포함되게 합니다. `LightningModule`은 PyTorch 코드를 5개 섹션으로 구성합니다:
- 계산 (`__init__`).
- 트레이닝 루프 (`training_step`)
- 검증 루프 (`validation_step`)
- 테스트 루프 (`test_step`)
- 옵티마이저 (`configure_optimizers`)

이를 통해 쉽게 공유할 수 있는 데이터셋에 구애받지 않는 모델을 구축할 수 있습니다. Cifar-10 분류를 위한 시스템을 구축해봅시다.


```
class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4):
        super().__init__()
        
        # 하이퍼파라미터 로그
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

    # 컨브 블록에서 Linear 레이어로 들어가는 출력 텐서의 크기를 반환합니다.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # 컨브 블록에서 피처 텐서를 반환합니다.
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

## 🚋 트레이닝 및 평가

이제 `DataModule`을 사용하여 데이터 파이프라인을 조직하고 `LightningModule`을 사용하여 모델 아키텍처+트레이닝 루프를 조직했으므로, PyTorch Lightning `Trainer`는 나머지 모든 것을 자동화합니다.

Trainer는 다음을 자동화합니다:
- 에포크 및 배치 반복
- `optimizer.step()`, `backward`, `zero_grad()` 호출
- `.eval()` 호출, grads 활성화/비활성화
- 가중치 저장 및 로딩
- Weights & Biases 로깅
- 멀티-GPU 트레이닝 지원
- TPU 지원
- 16비트 트레이닝 지원


```
dm = CIFAR10DataModule(batch_size=32)
# x_dataloader에 엑세스하려면 prepare_data와 setup을 호출해야 합니다.
dm.prepare_data()
dm.setup()

# 맞춤 ImagePredictionLogger 콜백에 필요한 샘플.
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

# 모델 트레이닝 ⚡🚅⚡
trainer.fit(model, dm)

# 보류 중인 테스트 세트에서 모델 평가 ⚡⚡
trainer.test(dataloaders=dm.test_dataloader())

# wandb 실행 종료
wandb.finish()
```

## 최종 생각들
저는 TensorFlow/Keras 에코시스템에서 왔고, PyTorch가 우아한 프레임워크임에도 불구하고 조금 압도적으로 느껴집니다. 하지만 그것은 개인적인 경험일 뿐입니다. PyTorch Lightning을 탐색하면서, PyTorch로부터 저를 멀어지게 했던 거의 모든 이유가 해결되었다는 것을 깨달았습니다. 여기 제가 흥분하는 몇 가지 요약이 있습니다:
- 그때: 전통적인 PyTorch 모델 정의는 여기저기 흩어져 있었습니다. 모델은 어떤 `model.py` 스크립트에, 트레이닝 루프는 `train.py` 파일에 있었습니다. 파이프라인을 이해하려면 많은 왔다갔다를 해야 했습니다.
- 지금: `LightningModule`은 모델이 `training_step`, `validation_step` 등과 함께 정의되는 시스템으로 작용합니다. 이제 모듈화되어 공유하기 쉽습니다.
- 그때: TensorFlow/Keras의 가장 좋은 점은 입력 데이터 파이프라인입니다. 그들의 데이터셋 카탈로그는 풍부하고 성장하고 있습니다. PyTorch의 데이터 파이프라인은 가장 큰 고통의 지점이었습니다. 일반적인 PyTorch 코드에서 데이터 다운로드/정리/준비는 보통 여러 파일에 흩어져 있습니다.
- 지금: DataModule은 데이터 파이프라인을 하나의 공유 가능하고 재사용 가능한 클래스로 조직화합니다. 그것은 단순히 `train_dataloader`, `val_dataloader`(들), `test_dataloader`(들)의 컬렉션과 일치하는 변형 및 데이터 처리/다운로드 단계가 필요합니다.
- 그때: Keras에서는 `model.fit`을 호출하여 모델을 훈련시키고 `model.predict`를 호출하여 추론을 실행할 수 있습니다. `model.evaluate`는 테스트 데이터에 대한 좋은 오래된 간단한 평가를 제공했습니다. PyTorch의 경우는 그렇지 않습니다. 보통 `train.py`와 `test.py` 파일이 따로 있습니다.
- 지금: `LightningModule`이 있으면 `Trainer`가 모든 것을 자동화합니다. 모델을 트레이닝하고 평가하기 위해 단순히 `trainer.fit`과 `trainer.test`를 호출하기만 하면 됩니다.
- 그때: TensorFlow는 TPU를 좋아하고, PyTorch는...음!
- 지금: PyTorch Lightning으로, 동일한 모델을 여러 GPU와 심지어 TPU에서 트레이닝하는 것이 매우 쉽습니다. 와우!
- 그때: 저는