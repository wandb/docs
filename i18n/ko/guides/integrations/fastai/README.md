---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Fastai

**fastai**를 사용하여 모델을 학습하는 경우, `WandbCallback`을 사용하여 W&B와 쉽게 통합할 수 있습니다. [인터랙티브 문서에서 예제와 함께 자세히 알아보기 →](https://app.wandb.ai/borisd13/demo\_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA)

## W&B로 로깅하기

**a)** [https://wandb.ai/site](https://wandb.ai/site)에서 무료 계정에 [가입](https://wandb.ai/site)한 후 wandb 계정에 로그인합니다.

**b)** `pip`을 사용하여 Python 3 환경의 컴퓨터에 wandb 라이브러리를 설치합니다.

**c)** 컴퓨터에서 wandb 라이브러리에 로그인합니다. 여기에서 API 키를 찾을 수 있습니다: [https://wandb.ai/authorize](https://wandb.ai/authorize).

<Tabs
  defaultValue="script"
  values={[
    {label: '명령 줄', value: 'script'},
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

그런 다음 `WandbCallback`을 `learner` 또는 `fit` 메서드에 추가합니다:

```python
import wandb
from fastai.callback.wandb import *

# wandb 실행 로깅 시작
wandb.init(project="my_project")

# 한 학습 단계 동안에만 로깅하려면
learn.fit(..., cbs=WandbCallback())

# 모든 학습 단계에 대해 지속적으로 로깅하려면
learn = learner(..., cbs=WandbCallback())
```

:::info
Fastai의 1 버전을 사용하는 경우 [Fastai v1 문서](v1.md)를 참조하세요.
:::

## WandbCallback 인수

`WandbCallback`은 다음 인수를 받습니다:

| 인수                     | 설명                                                                                                                                                                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| log                      | 모델의 "`그레이디언트`", "`파라미터`", "`all`" 또는 `None` (기본값)을 로깅할지 여부. 손실과 메트릭은 항상 로깅됩니다.                                                                                                                                 |
| log\_preds               | 예측 샘플을 로깅할지 여부 (기본값은 `True`).                                                                                                                                                                                               |
| log\_preds\_every\_epoch | 매 에포크마다 예측을 로깅할지 또는 끝에서 로깅할지의 여부 (기본값은 `False`)                                                                                                                                                                                    |
| log\_model               | 모델을 로깅할지 여부 (기본값은 False). 이는 `SaveModelCallback`도 필요합니다.                                                                                                                                                                  |
| model\_name              | 저장할 `파일`의 이름으로, `SaveModelCallback`을 덮어씁니다.                                                                                                                                                                                                |
| log\_dataset             | <ul><li><code>False</code> (기본값)</li><li><code>True</code>는 learn.dls.path를 참조하는 폴더를 로깅합니다.</li><li>특정 폴더를 참조하기 위해 경로를 명시적으로 정의할 수 있습니다.</li></ul><p><em>참고: "models" 서브폴더는 항상 무시됩니다.</em></p> |
| dataset\_name            | 로깅된 데이터세트의 이름 (기본값은 `폴더 이름`).                                                                                                                                                                                                           |
| valid\_dl                | 예측 샘플에 사용되는 아이템을 포함하는 `DataLoaders` (기본값은 `learn.dls.valid`에서 무작위 아이템).                                                                                                                                                  |
| n\_preds                 | 로깅된 예측의 수 (기본값은 36).                                                                                                                                                                                                                |
| seed                     | 무작위 샘플을 정의하는 데 사용됩니다.                                                                                                                                                                                                                            |

맞춤 워크플로우의 경우 데이터세트와 모델을 수동으로 로깅할 수 있습니다:

* `log_dataset(path, name=None, metadata={})`
* `log_model(path, name=None, metadata={})`

_참고: "models"라는 모든 서브폴더는 무시됩니다._

## 분산 학습

`fastai`는 `distrib_ctx` 컨텍스트 관리자를 사용하여 분산 학습을 지원합니다. W&B는 이를 자동으로 지원하여 여러 GPU 실험을 즉시 추적할 수 있습니다.

아래에 최소 예제가 나와 있습니다:

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

그런 다음 터미널에서 다음을 실행합니다:

```
$ torchrun --nproc_per_node 2 train.py
```

이 경우, 컴퓨터에는 2개의 GPU가 있습니다.

  </TabItem>
  <TabItem value="notebook">

노트북 내에서 직접 분산 학습을 실행할 수 있습니다!

```python
import wandb
from fastai.vision.all import *

from accelerate에 notebook_launcher
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

### 주 프로세스에서만 로깅하기

위의 예제에서는 `wandb`가 프로세스 당 하나의 실행을 시작합니다. 학습이 끝나면 두 개의 실행이 생성됩니다. 이는 때때로 혼란스러울 수 있으며, 주 프로세스에서만 로깅하고 싶을 수 있습니다. 이를 위해서는 수동으로 어떤 프로세스에 있는지 감지하고 다른 모든 프로세스에서 실행을 생성하지 않도록 (즉, `wandb.init`을 호출하지 않도록) 해야 합니다.

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

from accelerate에 notebook_launcher
from fastai.distributed는 *
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

## 예시

* [Fastai 모델 시각화, 추적 및 비교](https://app.wandb.ai/borisd13/demo\_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA): 철저하게 문서화된 연습
* [CamVid에서의 이미지 세그멘테이션](http://bit.ly/fastai-wandb): 통합의 샘플 사용 사례