---
title: fastai
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

만약 **fastai**를 사용하여 모델을 훈련하고 있다면, W&B는 `WandbCallback`을 사용한 쉬운 인테그레이션을 제공합니다. 예제를 포함한 상호작용 문서를 탐색하세요[ interactive docs with examples →](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA)

## Log with W&B

**a)** [가입하기](https://wandb.ai/site)에서 무료 계정을 만드시고, wandb 계정에 로그인하세요.

**b)** Python 3 환경에서 `pip`을 사용하여 wandb 라이브러리를 설치하세요.

**c)** 기기에 wandb 라이브러리에 로그인하세요. API 키는 여기에서 찾을 수 있습니다: [https://wandb.ai/authorize](https://wandb.ai/authorize).

<Tabs
  defaultValue="script"
  values={[
    {label: 'Command Line', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
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

그런 다음 `WandbCallback`을 `learner` 또는 `fit` 메소드에 추가하세요:

```python
import wandb
from fastai.callback.wandb import *

# wandb run을 로그하는 시작
wandb.init(project="my_project")

# 특정 트레이닝 단계 동안만 로그하려면
learn.fit(..., cbs=WandbCallback())

# 모든 트레이닝 단계를 계속 로그하려면
learn = learner(..., cbs=WandbCallback())
```

:::info
Fastai의 버전 1을 사용 중이라면, [Fastai v1 docs](v1.md)를 참조하세요.
:::

## WandbCallback 인수

`WandbCallback`은 다음 인수를 허용합니다:

| Args                     | Description                                                                                                                                                                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| log                      | 모델의: "`gradients`" , "`parameters`", "`all`" 또는 `None` (기본값)를 로그할지 여부. 손실 및 메트릭은 항상 로그됩니다.                                                                                                                                   |
| log_preds               | 예측 샘플을 로그할지 여부 (기본값은 `True`).                                                                                                                                                                            |
| log_preds_every_epoch | 매 에포크마다 예측을 로그할지 아니면 마지막에만 로그할지 여부 (기본값은 `False`)                                                                                                                                      |
| log_model               | 우리의 모델을 로그할지 여부 (기본값의 경우 False). 이것은 또한 `SaveModelCallback`이 필요합니다.                                                                                                                                                                  |
| model_name              | 저장할 `file`의 이름, `SaveModelCallback`을 덮어씀                                                                                                                                                                                   |
| log_dataset             | <ul><li><code>False</code> (기본값)</li><li><code>True</code>는 learn.dls.path로 참조된 폴더를 로그합니다.</li><li>로그할 폴더를 참조하기 위해 경로를 명시적으로 정의할 수 있습니다.</li></ul><p><em>참고: 서브폴더 "models"는 항상 무시됩니다.</em></p>  |
| dataset_name            | 로깅된 데이터셋의 이름 (기본값은 `폴더 이름`).                                                                                                                                                                            |
| valid_dl                | 예측 샘플에 사용된 항목이 포함된 `DataLoaders` (기본값은 `learn.dls.valid`의 임의 항목).                                                                                                                                     |
| n_preds                 | 기록된 예측값의 수 (기본값은 36).                                                                                                                                                                                            |
| seed                     | 랜덤 샘플을 정의하는 데 사용됨.                                                                                                                                                                                    |

사용자 지정 워크플로우의 경우, 데이터셋과 모델을 수동으로 로그할 수 있습니다:

* `log_dataset(path, name=None, metadata={})`
* `log_model(path, name=None, metadata={})`

_참고: 어떤 서브폴더 "모델"도 무시됩니다._

## Distributed Training

`fastai`는 `distrib_ctx` 컨텍스트 매니저를 사용하여 분산 트레이닝을 지원합니다. W&B는 이를 자동으로 지원하며, Multi-GPU 실험을 바로 추적할 수 있도록 합니다.

최소 예제는 아래와 같습니다:

<Tabs
  defaultValue="script"
  values={[
    {label: 'Script', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
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

그런 다음 터미널에서 실행하세요:

```
$ torchrun --nproc_per_node 2 train.py
```

이 경우, 기계에 2개의 GPU가 있습니다.

  </TabItem>
  <TabItem value="notebook">

이제 노트북 안에서 직접 분산 트레이닝을 실행할 수 있습니다!

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

### 주 프로세스에서만 로그하기

위의 예제에서는, `wandb`가 프로세스마다 하나의 run을 생성합니다. 트레이닝이 끝나면 두 개의 run이 생성됩니다. 이는 때때로 혼란스러울 수 있으며, 주 프로세스에서만 로그하고 싶을 수 있습니다. 이를 위해, 수동으로 어떤 프로세스에 있는지 감지하고 다른 모든 프로세스에서는 run(모든 다른 프로세스에서 `wandb.init` 호출)을 생성하지 않으면 됩니다.

<Tabs
  defaultValue="script"
  values={[
    {label: 'Script', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
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
터미널에서 호출하세요:

```
$ torchrun --nproc_per_node 2 train.py
```

  </TabItem>
  <TabItem value="notebook">

```python
import wandb
from fastai.vision.all import *

from accelerate import notebook_launcher
from fastai.distributed import *
from fastai.callback.wandb import WandbCallback

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

## Examples

* [Visualize, track, and compare Fastai models](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA): 철저히 문서화된 안내
* [Image Segmentation on CamVid](http://bit.ly/fastai-wandb): 인테그레이션의 샘플 유스 케이스