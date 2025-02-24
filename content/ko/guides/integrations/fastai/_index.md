---
title: fastai
cascade:
- url: guides/integrations/fastai/:filename
menu:
  default:
    identifier: ko-guides-integrations-fastai-_index
    parent: integrations
weight: 100
---

**fastai**를 사용하여 모델을 트레이닝하는 경우, W&B는 `WandbCallback`을 통해 쉽게 통합할 수 있습니다. 자세한 내용은 [예제가 포함된 대화형 문서 →](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA)에서 확인하세요.

## 가입 및 API 키 생성

API 키는 사용자의 머신을 W&B에 인증합니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
보다 간소화된 접근 방식을 위해 [https://wandb.ai/authorize](https://wandb.ai/authorize)로 직접 이동하여 API 키를 생성할 수 있습니다. 표시된 API 키를 복사하여 비밀번호 관리자와 같은 안전한 위치에 저장하십시오.
{{% /alert %}}

1. 오른쪽 상단 모서리에 있는 사용자 프로필 아이콘을 클릭합니다.
2. **User Settings**를 선택한 다음 **API Keys** 섹션으로 스크롤합니다.
3. **Reveal**을 클릭합니다. 표시된 API 키를 복사합니다. API 키를 숨기려면 페이지를 새로 고칩니다.

## `wandb` 라이브러리 설치 및 로그인

`wandb` 라이브러리를 로컬에 설치하고 로그인하려면 다음을 수행합니다.

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 API 키로 설정합니다.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` 라이브러리를 설치하고 로그인합니다.

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

{{% tab header="Python notebook" value="python-notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## `learner` 또는 `fit` 메소드에 `WandbCallback` 추가

```python
import wandb
from fastai.callback.wandb import *

# start logging a wandb run
wandb.init(project="my_project")

# To log only during one training phase
learn.fit(..., cbs=WandbCallback())

# To log continuously for all training phases
learn = learner(..., cbs=WandbCallback())
```

{{% alert %}}
Fastai 버전 1을 사용하는 경우 [Fastai v1 문서]({{< relref path="v1.md" lang="ko" >}})를 참조하십시오.
{{% /alert %}}

## WandbCallback 인수

`WandbCallback`은 다음 인수를 허용합니다.

| Args                     | Description                                                                                                                                                                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| log                      | 모델의 다음 항목을 기록할지 여부: `gradients`, `parameters`, `all` 또는 `None`(기본값). 손실 및 메트릭은 항상 기록됩니다.                                                                                                                                |
| log_preds               | 예측 샘플을 기록할지 여부(기본값은 `True`).                                                                                                                                                                                                                   |
| log_preds_every_epoch | 에포크마다 예측을 기록할지 또는 마지막에 기록할지 여부(기본값은 `False`).                                                                                                                                                                                    |
| log_model               | 모델을 기록할지 여부(기본값은 False). `SaveModelCallback`도 필요합니다.                                                                                                                                                                                 |
| model_name              | 저장할 `file`의 이름으로 `SaveModelCallback`을 재정의합니다.                                                                                                                                                                                               |
| log_dataset             | <ul><li><code>False</code> (기본값)</li><li><code>True</code>는 learn.dls.path에서 참조하는 폴더를 기록합니다.</li><li>경로를 명시적으로 정의하여 기록할 폴더를 참조할 수 있습니다.</li></ul><p><em>참고: 하위 폴더 "models"는 항상 무시됩니다.</em></p> |
| dataset_name            | 기록된 데이터셋의 이름(기본값은 `folder name`).                                                                                                                                                                                                           |
| valid_dl                | 예측 샘플에 사용되는 항목을 포함하는 `DataLoaders`(기본값은 `learn.dls.valid`의 임의 항목).                                                                                                                                                                  |
| n_preds                 | 기록된 예측 수(기본값은 36).                                                                                                                                                                                                                            |
| seed                     | 임의 샘플을 정의하는 데 사용됩니다.                                                                                                                                                                                                                            |

커스텀 워크플로우의 경우 데이터셋과 모델을 수동으로 기록할 수 있습니다.

* `log_dataset(path, name=None, metadata={})`
* `log_model(path, name=None, metadata={})`

_참고: 하위 폴더 "models"는 무시됩니다._

## 분산 트레이닝

`fastai`는 컨텍스트 관리자 `distrib_ctx`를 사용하여 분산 트레이닝을 지원합니다. W&B는 이를 자동으로 지원하며 Multi-GPU Experiments를 즉시 추적할 수 있도록 합니다.

다음 최소 예제를 검토하십시오.

{{< tabpane text=true >}}
{{% tab header="Script" value="script" %}}

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

터미널에서 다음을 실행합니다.

```shell
$ torchrun --nproc_per_node 2 train.py
```

이 경우 머신에 2개의 GPU가 있습니다.

{{% /tab %}}
{{% tab header="Python notebook" value="notebook" %}}

이제 노트북 내에서 직접 분산 트레이닝을 실행할 수 있습니다.

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

{{% /tab %}}
{{< /tabpane >}}

### 메인 프로세스에서만 기록

위의 예에서 `wandb`는 프로세스당 하나의 run을 시작합니다. 트레이닝이 끝나면 두 개의 run이 생성됩니다. 이는 혼란스러울 수 있으므로 메인 프로세스에서만 기록할 수 있습니다. 이렇게 하려면 수동으로 어떤 프로세스에 있는지 감지하고 다른 모든 프로세스에서 run을 생성하지 않도록 해야 합니다(`wandb.init` 호출).

{{< tabpane text=true >}}
{{% tab header="Script" value="script" %}}

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
터미널에서 다음을 호출합니다.

```
$ torchrun --nproc_per_node 2 train.py
```

{{% /tab %}}
{{% tab header="Python notebook" value="notebook" %}}

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

{{% /tab %}}
{{< /tabpane >}}

## 예제

* [Visualize, track, and compare Fastai models](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA): 자세한 설명이 포함된 연습
* [Image Segmentation on CamVid](http://bit.ly/fastai-wandb): 통합의 샘플 유스 케이스
