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

**fastai**로 모델을 트레이닝 할 때, W&B는 `WandbCallback`을 이용해 쉽게 인테그레이션 할 수 있습니다. 자세한 내용과 예시는 [인터랙티브 문서에서 확인하세요 →](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA)

## 회원가입 및 API 키 생성

API 키는 여러분의 머신을 W&B에 인증하는 역할을 합니다. API 키는 사용자 프로필에서 생성할 수 있습니다.

{{% alert %}}
좀 더 간단하게 진행하려면, [W&B 인증 페이지](https://wandb.ai/authorize)에서 바로 API 키를 생성할 수 있습니다. 화면에 표시되는 API 키를 복사해 암호화된 저장소(예: 패스워드 매니저)에 안전하게 보관하세요.
{{% /alert %}}

1. 우측 상단의 사용자 프로필 아이콘을 클릭하세요.
1. **User Settings**를 선택한 후, **API Keys** 섹션까지 스크롤하세요.
1. **Reveal**을 클릭하면 API 키가 표시됩니다. 해당 키를 복사하세요. 키를 다시 숨기려면 페이지를 새로고침하세요.

## `wandb` 라이브러리 설치 및 로그인

로컬 환경에 `wandb` 라이브러리를 설치하고 로그인 하려면 다음을 따르세요.

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 여러분의 API 키로 설정합니다.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` 라이브러리를 설치하고 로그인하세요.

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

## `WandbCallback`을 `learner` 또는 `fit` 메소드에 추가하기

```python
import wandb
from fastai.callback.wandb import *

# wandb run에 대한 로그 시작
wandb.init(project="my_project")

# 한 번의 트레이닝 구간만 로그할 경우
learn.fit(..., cbs=WandbCallback())

# 모든 트레이닝 구간 동안 계속 로그할 경우
learn = learner(..., cbs=WandbCallback())
```

{{% alert %}}
Fastai 버전 1을 사용 중이라면 [Fastai v1 문서]({{< relref path="v1.md" lang="ko" >}})를 참고하세요.
{{% /alert %}}

## WandbCallback 인수

`WandbCallback`에서 사용할 수 있는 인수는 다음과 같습니다.

| 인수                     | 설명                                                                                                                                                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| log                      | 모델의 `gradients`, `parameters`, `all`, 또는 `None`(기본값)을 로그할지 여부입니다. 손실 및 메트릭은 항상 로그됩니다.                                                                                                               |
| log_preds                | 예측 샘플을 로그할지 여부 (기본값: `True`).                                                                                                                                                                                      |
| log_preds_every_epoch    | 에포크마다 예측값을 로그할지 혹은 마지막에만 할지 여부 (기본값: `False`)                                                                                                                                                                    |
| log_model                | 모델을 로그할지 여부 (기본값: False). 이 옵션을 사용하려면 `SaveModelCallback`도 필요합니다.                                                                                                                                                 |
| model_name               | 저장할 `file`의 이름 지정, `SaveModelCallback`을 덮어씁니다.                                                                                                                                                      |
| log_dataset              | <ul><li><code>False</code> (기본값)</li><li><code>True</code>일 경우, learn.dls.path에서 참조하는 폴더를 로그합니다.</li><li>로그할 폴더의 경로를 명시적으로 정의할 수도 있습니다.</li></ul><p><em>참고: "models"라는 하위 폴더는 항상 무시됩니다.</em></p> |
| dataset_name             | 로그할 데이터셋의 이름 (기본값은 폴더명).                                                                                                                                                                                 |
| valid_dl                 | 예측 샘플로 사용할 아이템을 담고 있는 `DataLoaders` (기본값은 `learn.dls.valid`에서 무작위 아이템 사용).                                                                                                                                            |
| n_preds                  | 로그할 예측 샘플 개수 (기본값: 36).                                                                                                                                                                                           |
| seed                     | 무작위 샘플 지정을 위한 시드 값.                                                                                                                                                                                             |

커스텀 워크플로우가 필요하다면 아래처럼 직접 데이터셋과 모델을 로그할 수도 있습니다:

* `log_dataset(path, name=None, metadata={})`
* `log_model(path, name=None, metadata={})`

_참고: "models"라는 하위 폴더는 무시됩니다._

## 분산 트레이닝

`fastai`는 `distrib_ctx` 컨텍스트 매니저를 이용한 분산 트레이닝을 지원합니다. W&B에서는 이를 자동으로 감지하여 Multi-GPU 실험을 바로 추적할 수 있습니다.

최소 예제는 다음과 같습니다:

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

이후 터미널에서 아래와 같이 실행합니다:

```shell
$ torchrun --nproc_per_node 2 train.py
```

이 예시에서는 머신에 GPU 2개가 있다고 가정합니다.

{{% /tab %}}
{{% tab header="Python notebook" value="notebook" %}}

노트북 안에서도 곧바로 분산 트레이닝을 실행할 수 있습니다.

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

### 메인 프로세스에서만 로그하기

위의 예시들에서 `wandb`는 프로세스마다 하나의 run을 생성합니다. 트레이닝 종료 시 2개의 run이 생성되는데, 이는 혼란을 줄 수 있으므로 메인 프로세스에서만 로그하고 싶을 수 있습니다. 이 경우, 코드 내에서 현재 프로세스를 직접 확인 후 메인 프로세스를 제외한 다른 곳에서는 run을 만들지 않으면 됩니다 (`wandb.init`를 메인 프로세스 이외에서 호출하지 않음).

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
터미널에서 다음과 같이 실행합니다:

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

* [Visualize, track, and compare Fastai models](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA): 친절하게 설명된 워크스루 예제입니다.
* [Image Segmentation on CamVid](https://bit.ly/fastai-wandb): 이 인테그레이션의 실제 유스 케이스 예제입니다.