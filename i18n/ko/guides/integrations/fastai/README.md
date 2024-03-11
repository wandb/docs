---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Fastai

**fastai**를 사용하여 모델을 트레이닝하는 경우, `WandbCallback`을 사용한 쉬운 인테그레이션을 W&B에서 제공합니다. [인터랙티브 문서에서 예제와 함께 자세히 알아보세요 →](https://app.wandb.ai/borisd13/demo\_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA)

## W&B로 로그하기

**a)** [https://wandb.ai/site](https://wandb.ai/site)에서 무료 계정을 [가입](https://wandb.ai/site)한 후 wandb 계정에 로그인하세요.

**b)** `pip`을 사용하여 Python 3 환경의 컴퓨터에 wandb 라이브러리를 설치하세요.

**c)** 컴퓨터에서 wandb 라이브러리에 로그인하세요. 여기서 API 키를 찾을 수 있습니다: [https://wandb.ai/authorize](https://wandb.ai/authorize).

<Tabs
  defaultValue="script"
  values={[
    {label: '커맨드라인', value: 'script'},
    {label: '노트북', value: 'notebook'},
  ]}>
  <TabItem value="script">

```shell
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

그런 다음 `learner` 또는 `fit` 메소드에 `WandbCallback`을 추가하세요:

```python
import wandb
from fastai.callback.wandb import *

# wandb run을 시작합니다
wandb.init(project="my_project")

# 한 트레이닝 단계에만 로그를 하려면
learn.fit(..., cbs=WandbCallback())

# 모든 트레이닝 단계에 계속해서 로그를 하려면
learn = learner(..., cbs=WandbCallback())
```

:::info
Fastai의 버전 1을 사용하는 경우, [Fastai v1 문서](v1.md)를 참조하세요.
:::

## WandbCallback 인수

`WandbCallback`은 다음 인수를 받습니다:

| 인수                     | 설명                                                                                                                                                                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| log                      | 모델의 "`그레이디언트`", "`파라미터`", "`all`" 또는 `None`(기본값)을 로그할지 여부. 손실 & 메트릭은 항상 로그됩니다.                                                                                                                                 |
| log\_preds               | 예측 샘플을 로그할지 여부 (기본값은 `True`).                                                                                                                                                                                               |
| log\_preds\_every\_epoch | 매 에포크마다 예측을 로그할지 아니면 끝에 로그할지 여부 (기본값은 `False`)                                                                                                                                                                                    |
| log\_model               | 모델을 로그할지 여부 (기본값은 False). 이는 `SaveModelCallback`도 필요로 합니다.                                                                                                                                                                  |
| model\_name              | 저장할 `파일`의 이름, `SaveModelCallback`을 오버라이드합니다.                                                                                                                                                                                                |
| log\_dataset             | <ul><li><code>False</code> (기본값)</li><li><code>True</code>는 learn.dls.path를 참조하는 폴더를 로그합니다.</li><li>명시적으로 경로를 정의하여 어떤 폴더를 로그할지 참조할 수 있습니다.</li></ul><p><em>참고: "models" 하위 폴더는 항상 무시됩니다.</em></p> |
| dataset\_name            | 로그된 데이터셋의 이름 (기본값은 `폴더 이름`).                                                                                                                                                                                                           |
| valid\_dl                | 예측 샘플에 사용되는 아이템이 포함된 `DataLoaders` (기본값은 `learn.dls.valid`에서 무작위 아이템).                                                                                                                                                  |
| n\_preds                 | 로그된 예측의 수 (기본값은 36).                                                                                                                                                                                                                |
| seed                     | 무작위 샘플을 정의하는 데 사용됩니다.                                                                                                                                                                                                                            |

커스텀 워크플로우의 경우, 데이터셋과 모델을 수동으로 로그할 수 있습니다:

* `log_dataset(path, name=None, metadata={})`
* `log_model(path, name=None, metadata={})`

_참고: "models"라는 모든 하위 폴더는 무시됩니다._

## 분산 트레이닝

`fastai`는 `distrib_ctx` 컨텍스트 관리자를 사용하여 분산 트레이닝을 지원합니다. W&B는 이를 자동으로 지원하며, 여러분이 Multi-GPU 실험을 즉시 추적할 수 있게 해줍니다.

아래에 최소한의 예제를 보여줍니다:

<Tabs
  defaultValue="script"
  values={[
    {label: '스크립트', value: 'script'},
    {label: '노트북', value: 'notebook'},
  ]}>
  <TabItem value="script">


```python
import wandb
from fastai.vision.all import *
from fastai.distributed import *
from fastai.callback.wandb import WandbCallback

wandb.require(experiment="service")
path = rank0_first(lambda: untar_data(URLs.PETS) / "images")


def train():
    dls = ImageDataLoaders.from_name_func(
        path,
        get_image_files(path),
        valid_pct=0.2,
        label_func=lambda x: x[0].isupper(),
        item_tfms=Resize(224),
    )
    wandb.init("fastai_ddp", entity="capecape")
    cb = WandbCallback()
    learn = vision_learner(dls, resnet34, metrics=error_rate, cbs=cb).to_fp16()
    with learn.distrib_ctx(sync_bn=False):
        learn.fit(1)


if __name__ == "__main__":
    train()
```

터미널에서 다음을 실행하세요:

```
$ torchrun --nproc_per_node 2 train.py
```

이 경우, 기계에는 2개의 GPU가 있습니다.

  </TabItem>
  <TabItem value="notebook">

노트북 내에서 바로 분산 트레이닝을 실행할 수 있습니다!

```python
import wandb
from fastai.vision.all import *

from accelerate import notebook_launcher
from fastai.distributed import *
from fastai.callback.wandb import WandbCallback

wandb.require(experiment="service")
path = untar_data(URLs.PETS) / "images"


def train():
    dls = ImageDataLoaders.from_name_func(
        path,
        get_image_files(path),
        valid_pct=0.2,
        label_func=lambda x: x[0].isupper(),
        item_tfms=Resize(224),
    )
    wandb.init("fastai_ddp", entity="capecape")
    cb = WandbCallback()
    learn = vision_learner(dls, resnet34, metrics=error_rate, cbs=cb).to_fp16()
    with learn.distrib_ctx(in_notebook=True, sync_bn=False):
        learn.fit(1)


notebook_launcher(train, num_processes=2)
```

  </TabItem>
</Tabs>

### 메인 프로세스에서만 로그하기

위의 예제에서, `wandb`는 프로세스마다 하나의 run을 시작합니다. 트레이닝이 끝났을 때, 두 개의 run이 생길 것입니다. 이는 가끔 혼란을 줄 수 있으며, 메인 프로세스에서만 로그하고 싶을 수 있습니다. 이를 위해서는 수동으로 어떤 프로세스에 있는지 감지하고, 다른 모든 프로세스에서 실행을 생성하는 것(즉, `wandb.init`을 호출하는 것)을 피해야 합니다.

<Tabs
  defaultValue="script"
  values={[
    {label: '스크립트', value: 'script'},
    {label: '노트북', value: 'notebook'},
  ]}>
  <TabItem value="script">

```python
import wandb
from fastai.vision.all import *
from fastai.distributed import *
from fastai.callback.wandb import WandbCallback

wandb.require(experiment="service")
path = rank0_first(lambda: untar_data(URLs.PETS) / "images")


def train():
    cb = []
    dls = ImageDataLoaders.from_name_func(
        path,
        get_image_files(path),
        valid_pct=0.2,
        label_func=lambda x: x[0].isupper(),
        item_tfms=Resize(224),
    )
    if rank_distrib() == 0:
        run = wandb.init("fastai_ddp", entity="capecape")
        cb = WandbCallback()
    learn = vision_learner(dls, resnet34, metrics=error_rate, cbs=cb).to_fp16()
    with learn.distrib_ctx(sync_bn=False):
        learn.fit(1)


if __name__ == "__main__":
    train()
```
터미널에서 다음을 호출하세요:

```
$ torchrun --nproc_per_node 2 train.py
```

  </TabItem>
  <TabItem value="notebook">

```python
import wandb
from fastai.vision.all import *

from accelerate는 notebook_launcher
from fastai.distributed import *
from fastai.callback.wandb는 WandbCallback

wandb.require(experiment="service")
path = untar_data(URLs.PETS) / "images"


def train():
    cb = []
    dls = ImageDataLoaders.from_name_func(
        path,
        get_image_files(path),
        valid_pct=0.2,
        label_func=lambda x: x[0].isupper(),
        item_tfms=Resize(224),
    )
    if rank_distrib() == 0:
        run = wandb.init("fastai_ddp", entity="capecape")
        cb = WandbCallback()
    learn = vision_learner(dls, resnet34, metrics=error_rate, cbs=cb).to_fp16()
    with learn.distrib_ctx(in_notebook=True, sync_bn=False):
        learn.fit(1)


notebook_launcher(train, num_processes=2)
```

  </TabItem>
</Tabs>

## 예제들

* [Fastai 모델 시각화, 추적 및 비교](https://app.wandb.ai/borisd13/demo\_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA): 철저하게 문서화된 안내서
* [CamVid에서의 이미지 세그멘테이션](http://bit.ly/fastai-wandb): 인테그레이션의 샘플 유스 케이스