---
title: PyTorch Lightning
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Image_Classification_using_PyTorch_Lightning.ipynb'/>

우리는 PyTorch Lightning을 사용하여 이미지 분류 파이프라인을 구축할 것입니다. 코드의 가독성과 재현성을 높이기 위해 우리는 이 [스타일 가이드](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html)를 따를 것입니다. 이에 대한 멋진 설명은 [여기](https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY)에서 확인할 수 있습니다.

## PyTorch Lightning 및 W&B 설정하기

이 튜토리얼에서는 PyTorch Lightning과 Weights and Biases가 필요합니다.

```shell
pip install lightning -q
pip install wandb -qU
```

```python
import lightning.pytorch as pl
# 가장 좋아하는 기계학습 추적 툴
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

## 🔧 DataModule - 우리가 자격 있는 데이터 파이프라인

DataModules는 LightningModule에서 데이터 관련 훅을 분리하여 데이터셋에 구애받지 않는 모델을 개발할 수 있는 방법입니다.

이것은 데이터 파이프라인을 하나의 공유 가능하고 재사용 가능한 클래스로 구성합니다. datamodule은 PyTorch에서 데이터 처리에 관련된 다섯 가지 단계를 캡슐화합니다:
- 다운로드 / 토큰화 / 처리. 
- 정리하고 디스크에 (필요하면) 저장.
- 데이터셋 내부에 로드.
- 변환 적용 (회전, 토큰화 등).
- DataLoader로 래핑.

datamodule에 대해 더 알아보세요 [여기](https://lightning.ai/docs/pytorch/stable/data/datamodule.html). Cifar-10 데이터셋을 위한 datamodule을 만들어봅시다.

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
        # dataloaders에서 사용하기 위한 train/val 데이터셋 할당
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # dataloader(s)에서 사용하기 위한 테스트 데이터셋 할당
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

콜백은 여러 프로젝트에서 재사용할 수 있는 독립형 프로그램입니다. PyTorch Lightning은 정기적으로 사용되는 몇 가지 [내장 콜백](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html#built-in-callbacks)을 제공합니다.
PyTorch Lightning의 콜백에 대해 더 알아보세요 [여기](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html).

### 내장 콜백

이 튜토리얼에서는 [Early Stopping](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.callbacks.EarlyStopping)과 [Model Checkpoint](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) 내장 콜백을 사용할 것입니다. 이들은 `Trainer`에 전달될 수 있습니다.

### 커스텀 콜백
Custom Keras 콜백에 익숙하다면, PyTorch 파이프라인에서도 동일한 기능을 사용할 수 있다는 것은 단순한 부가적인 이점일 뿐입니다.

이미지 분류를 수행하고 있기 때문에 모델의 예측값을 이미지 샘플에서 시각화하는 기능이 도움이 될 수 있습니다. 이것은 콜백 형태로 모델을 초기 단계에서 디버그하는 데 도움이 될 수 있습니다.

```
class ImagePredictionLogger(pl.callbacks.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # 텐서를 CPU로 이동
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # 모델 예측값 가져오기
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # 이미지 wandb 이미지로 로그
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
            })
        
```

## 🎺 LightningModule - 시스템 정의하기

LightningModule은 시스템을 정의하며 모델이 아닙니다. 여기서 시스템은 모든 연구 코드를 하나의 클래스로 그룹화하여 독립형으로 만들었습니다. `LightningModule`은 PyTorch 코드를 다섯 가지 섹션으로 구성합니다:
- 연산 (`__init__`).
- 트레이닝 루프 (`training_step`)
- 검증 루프 (`validation_step`)
- 테스트 루프 (`test_step`)
- 옵티마이저 (`configure_optimizers`)

이로 인해 쉽게 공유할 수 있는 데이터셋에 종속되지 않는 모델을 구축할 수 있습니다. Cifar-10 분류를 위한 시스템을 만들어봅시다.

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

    # conv 블록에서 Linear 레이어로 들어가는 출력 텐서의 크기를 반환합니다.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # conv 블록에서 피처 텐서를 반환합니다.
    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x
    
    # 추론 시 사용됩니다.
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

`DataModule`을 사용하여 데이터 파이프라인을, `LightningModule`을 사용하여 모델 아키텍처와 트레이닝 루프를 구성했으므로, PyTorch Lightning `Trainer`는 나머지 모든 것을 자동화해줍니다. 

Trainer가 자동화하는 것은:
- 에폭 및 배치 반복
- `optimizer.step()`, `backward`, `zero_grad()` 호출
- `.eval()` 호출, 그래드 활성화/비활성화
- 가중치 저장 및 로드
- Weights and Biases 로깅
- 다중-GPU 트레이닝 지원
- TPU 지원
- 16-bit 트레이닝 지원

```
dm = CIFAR10DataModule(batch_size=32)
# x_dataloader에 엑세스하려면 prepare_data와 setup를 호출해야 합니다.
dm.prepare_data()
dm.setup()

# 이미지 예측을 로깅하는 커스텀 ImagePredictionLogger 콜백에 필요한 샘플.
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

# 홀드아웃 테스트 세트에서 모델 평가 ⚡⚡
trainer.test(dataloaders=dm.test_dataloader())

# wandb run 종료
wandb.finish()
```

## 마지막 생각
저는 TensorFlow/Keras 에코시스템에서 왔으며, PyTorch는 우아한 프레임워크임에도 불구하고 다소 압도적이라고 느꼈습니다. 하지만 이는 개인적인 경험에 불과합니다. PyTorch Lightning을 탐색하면서, PyTorch에서 멀리하게 만들었던 거의 모든 이유가 해결되었음을 깨달았습니다. 여기 제 흥미로운 요약을 빠르게 소개합니다:
- 그때: 기존 PyTorch 모델 정의는 여기저기 흩어져 있었습니다. 모델은 `model.py` 스크립트에 있고, 트레이닝 루프는 `train.py` 파일에 있습니다. 파이프라인을 이해하기 위해 왔다 갔다 보는 일이 많았습니다. 
- 지금: `LightningModule`은 시스템 역할을 하며, 모델과 `training_step`, `validation_step` 등을 함께 정의합니다. 이제는 모듈화되어 있고 공유 가능합니다.
- 그때: TensorFlow/Keras의 가장 좋은 점은 입력 데이터 파이프라인입니다. 그들의 데이터셋 카탈로그는 풍부하고 계속 성장하고 있습니다. PyTorch의 데이터 파이프라인은 가장 큰 고통점이었습니다. 일반적인 PyTorch 코드에서는 데이터 다운로드/정리/준비가 여러 파일에 흩어져 있는 경우가 많습니다. 
- 지금: DataModule은 데이터 파이프라인을 하나의 공유 가능하고 재사용 가능한 클래스로 구성합니다. 이는 `train_dataloader`, `val_dataloader`(들), `test_dataloader`(들) 및 필요한 매칭 변환과 데이터 처리/다운로드 단계의 단순한 모음입니다.
- 그때: Keras에서는 `model.fit`을 호출하여 모델을 트레이닝하고, `model.predict`를 사용하여 추론을 실행할 수 있습니다. `model.evaluate`는 테스트 데이터에 대한 익숙하고 간단한 평가를 제공했습니다. PyTorch에서는 그렇지 않았습니다. 보통은 별도의 `train.py`과 `test.py` 파일을 찾게 됩니다. 
- 지금: `LightningModule`이 적용되면서, `Trainer`가 모든 것을 자동화합니다. `trainer.fit`과 `trainer.test`만 호출하여 모델을 트레이닝하고 평가하면 됩니다.
- 그때: TensorFlow는 TPU를 좋아하지만, PyTorch는... 음! 
- 지금: PyTorch Lightning을 사용하면 단일 모델을 여러 GPU에서 훈련하고 TPU에서도 매우 쉽게 훈련할 수 있습니다. 와우!
- 그때: 저는 콜백의 열성 팬이며, 커스텀 콜백 작성을 선호합니다. Early Stopping 같은 사소한 것도 기존 PyTorch에서는 논의점이 되곤 했습니다. 
- 지금: PyTorch Lightning을 사용하면 Early Stopping 및 Model Checkpointing이 수월합니다. 저는 심지어 커스텀 콜백을 작성할 수도 있습니다.

## 🎨 결론 및 리소스

이 리포트가 도움이 되길 바랍니다. 코드를 실험해보고 원하는 데이터셋으로 이미지 분류기를 트레이닝 할 것을 권장합니다.

PyTorch Lightning에 대해 더 알아볼 수 있는 몇 가지 리소스를 소개합니다:
- [단계별 안내](https://lightning.ai/docs/pytorch/latest/starter/introduction.html) - 이는 공식 튜토리얼 중 하나입니다. 그들의 문서는 정말 잘 작성되어 있으며, 훌륭한 학습 리소스로 적극 추천합니다.
- [Weights & Biases와 함께 Pytorch Lightning 사용하기](https://wandb.me/lightning) - W&B와 PyTorch Lightning을 사용하는 방법을 더 알아보기 위해 실행할 수 있는 빠른 colab입니다.