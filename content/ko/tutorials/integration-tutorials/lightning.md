---
title: PyTorch Lightning
menu:
  tutorials:
    identifier: ko-tutorials-integration-tutorials-lightning
    parent: integration-tutorials
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Image_Classification_using_PyTorch_Lightning.ipynb" >}}
PyTorch Lightning을 활용하여 이미지 분류 파이프라인을 구축해봅시다. 코드의 가독성과 재현성을 높이기 위해 [이 스타일 가이드](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html)를 따라 진행합니다. 관련 설명을 [여기](https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY)에서 확인할 수도 있습니다.

## PyTorch Lightning 및 W&B 환경 설정

이 튜토리얼에서는 PyTorch Lightning과 W&B가 필요합니다.

```shell
pip install lightning -q
pip install wandb -qU
```

```python
import lightning.pytorch as pl

# 즐겨 사용하는 기계학습 추적 툴
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

## DataModule - 우리가 원하던 데이터 파이프라인

DataModules는 LightningModule로부터 데이터 관련 훅을 분리해서 데이터셋에 독립적인 모델을 개발할 수 있게 해줍니다.

데이터 파이프라인을 하나의 공유 가능한, 재사용 가능한 클래스로 구성합니다. datamodule은 PyTorch에서의 데이터 처리에 필요한 5가지 단계를 모두 캡슐화합니다:
- 다운로드 / 토큰화 / 처리.
- 데이터 정리 및 (필요시) 디스크 저장.
- Dataset 내부로 로드.
- 변환(회전, 토큰화 등) 적용.
- DataLoader로 래핑.

datamodule에 대해 더 알고 싶다면 [여기](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)를 참고하세요. 이제 Cifar-10 데이터셋용 datamodule을 만들어봅시다.

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

        # dataloader에서 사용할 test 데이터셋 할당
        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)
```

## 콜백(Callback)

콜백은 프로젝트 간에 재사용할 수 있는 독립적인 프로그램입니다. PyTorch Lightning에는 자주 사용되는 [빌트인 콜백](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html#built-in-callbacks)들이 있습니다. PyTorch Lightning에서의 콜백에 대한 자세한 설명은 [여기](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html)를 참고하세요.

### 빌트인 콜백

이 튜토리얼에서는 [Early Stopping](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.callbacks.EarlyStopping)과 [Model Checkpoint](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) 빌트인 콜백을 사용할 예정입니다. 이 콜백들은 `Trainer`에 전달할 수 있습니다.

### 커스텀 콜백
Keras에서 커스텀 콜백을 사용해보셨다면, PyTorch 파이프라인에서 똑같이 쓸 수 있다는 점이 정말 반가울 거예요.

이번 튜토리얼의 이미지 분류 작업에서는, 모델이 예측한 이미지를 샘플로 직접 시각화하며 확인할 수 있습니다. 이를 콜백 형태로 활용하면 모델을 초기에 디버깅하는데 도움이 됩니다.

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
        # 모델 예측값 얻기
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # 이미지를 wandb Image로 로그 기록
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
            })
        
```

## LightningModule - 시스템 정의하기

LightningModule은 단순히 모델이 아닌 하나의 시스템을 정의합니다. 즉, 관련 코드를 하나의 클래스로 묶어 자기 완결적으로 만들어줍니다. `LightningModule`은 PyTorch 코드를 다섯 섹션으로 나눠 정리해줍니다:
- 계산(`__init__`)
- 트레이닝 루프(`training_step`)
- 검증 루프(`validation_step`)
- 테스트 루프(`test_step`)
- 옵티마이저 구성(`configure_optimizers`)

이렇게 데이터셋에 독립적인 모델을 쉽게 공유할 수 있게 됩니다. Cifar-10 분류용 시스템을 만들어봅시다.

```
class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4):
        super().__init__()
        
        # 하이퍼파라미터 로그 기록
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

    # conv 블록에서 Linear로 넘어가는 출력 텐서 크기 반환
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # conv 블록의 특징 텐서 반환
    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x
    
    # 추론 시 사용
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

## 학습 및 평가

이제 데이터 파이프라인은 `DataModule`, 모델 아키텍처와 트레이닝 루프는 `LightningModule`로 구성했으니, PyTorch Lightning의 `Trainer`가 나머지를 자동화해줍니다.

Trainer의 자동화 항목:
- 에포크 및 배치 반복
- `optimizer.step()`, `backward`, `zero_grad()` 호출
- `.eval()`호출 및 grad 동작 ON/OFF 설정
- 가중치 저장 및 불러오기
- W&B 로그 기록
- 멀티 GPU 트레이닝 지원
- TPU 지원
- 16비트 트레이닝 지원

```
dm = CIFAR10DataModule(batch_size=32)
# x_dataloader를 엑세스하려면 prepare_data와 setup을 호출해야 함
dm.prepare_data()
dm.setup()

# 커스텀 ImagePredictionLogger 콜백이 이미지 예측을 기록하려면 샘플이 필요합니다.
val_samples = next(iter(dm.val_dataloader()))
val_imgs, val_labels = val_samples[0], val_samples[1]
val_imgs.shape, val_labels.shape
```

```
model = LitModel((3, 32, 32), dm.num_classes)

# wandb logger 초기화
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

# 모델 학습
trainer.fit(model, dm)

# 테스트 세트에서 모델 평가 ⚡⚡
trainer.test(dataloaders=dm.test_dataloader())

# wandb run 종료
run.finish()
```

## 마무리 생각
저는 TensorFlow/Keras 에코시스템에서 시작해서, PyTorch가 우아한 프레임워크임에도 다소 어렵게 느껴졌습니다. 하지만 PyTorch Lightning을 살펴보면서, PyTorch를 꺼리게 했던 거의 모든 이유가 해결됐다는 걸 알게 됐어요. 간단히 정리하자면 이렇습니다:
- 예전: 기존의 PyTorch 모델 정의는 여기저기에 흩어져 있었습니다. 모델은 어떤 `model.py` 스크립트에, 트레이닝 루프는 `train.py` 파일에 있고… 파이프라인을 이해하려면 왔다갔다 해야 했죠. 
- 지금: `LightningModule` 안에 모델, `training_step`, `validation_step` 등이 모두 정의되어, 훨씬 모듈화되고 공유하기 좋아졌습니다.
- 예전: TensorFlow/Keras의 데이터 파이프라인이 정말 편했고, 데이터셋 카탈로그도 다양했습니다. PyTorch의 데이터 파이프라인은 늘 고민거리였어요. 일반적으로 데이터 다운로드/전처리가 여러 파일에 흩어져 있었죠. 
- 지금: DataModule 덕분에 모든 데이터 파이프라인 과정을 공유 및 재사용 가능한 하나의 클래스로 정리할 수 있습니다. train/val/test dataloader와 매칭되는 transforms, 데이터 처리/다운로드 코드를 손쉽게 구성할 수 있죠.
- 예전: Keras에서는 `model.fit`으로 간단히 학습을 시작하고, `model.predict`로 추론할 수 있었습니다. `model.evaluate`도 손쉽게 쓸 수 있었구요. PyTorch에서는 대부분 별도의 `train.py`, `test.py` 파일에서 각자 수행하곤 했습니다. 
- 지금: LightningModule 체계에서는 Trainer가 모든 것을 자동화해줍니다. 그냥 `trainer.fit`, `trainer.test`만 호출하면 끝!
- 예전: TensorFlow는 TPU를 아주 잘 지원했는데, PyTorch는...
- 지금: PyTorch Lightning 덕분에 동일 모델을 여러 GPU, 심지어 TPU에서도 너무 쉽게 돌릴 수 있습니다.
- 예전: Callbacks를 무척 좋아하고 직접 작성하는 걸 선호하는데, Early Stopping 같은 것조차 예전 PyTorch에서는 고민거리였습니다.
- 지금: PyTorch Lightning은 Early Stopping, Model Checkpoint 등 콜백 활용이 정말 쉬워졌고, 커스텀 콜백도 얼마든지 쓸 수 있습니다.

## 🎨 정리 및 참고 자료

이 리포트가 도움이 되었길 바랍니다. 코드로 직접 실험해보고, 원하는 데이터셋으로 이미지 분류기를 트레이닝해보시길 추천합니다.

PyTorch Lightning에 대해 더 배우고 싶다면 다음 자료를 참고해 주세요:
- [단계별 가이드](https://lightning.ai/docs/pytorch/latest/starter/introduction.html): 공식 튜토리얼 중 하나로, 문서가 매우 잘 정리되어 있어 적극 추천합니다.
- [W&B와 Pytorch Lightning 함께 사용하기](https://wandb.me/lightning): W&B와 PyTorch Lightning을 함께 사용하는 방법을 배우고 싶은 분께 좋은 colab 예제입니다.