---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Fastai

**fastai** を使ってモデルをトレーニングする場合、W&B の `WandbCallback` を使用した簡単なインテグレーションがあります。[インタラクティブなドキュメントとサンプルはこちら →](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA) 

## W&B でログを取る

**a)** [Sign up](https://wandb.ai/site) して、無料アカウントに登録し、次に wandb アカウントにログインします。

**b)** Python 3 環境で `pip` を使用して wandb ライブラリをマシンにインストールします。

**c)** マシンで wandb ライブラリにログインします。APIキーはここで見つかります: [https://wandb.ai/authorize](https://wandb.ai/authorize).

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

次に `WandbCallback` を `learner` または `fit` メソッドに追加します:

```python
import wandb
from fastai.callback.wandb import *

# wandb run を開始してログを取る
wandb.init(project="my_project")

# トレーニングの1フェーズのみをログする場合
learn.fit(..., cbs=WandbCallback())

# すべてのトレーニングフェーズで継続的にログする場合
learn = learner(..., cbs=WandbCallback())
```

:::info
Fastai バージョン1を使用している場合は、[Fastai v1 docs](v1.md) を参照してください。
:::

## WandbCallback 引数

`WandbCallback` は以下の引数を受け取ります:

| 引数                      | 説明                                                                                                                                                                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| log                      | モデルのログを取るかどうか: "`gradients`" , "`parameters`", "`all`" あるいは `None` (デフォルト)。ロスとメトリクスは常にログされます。                                                                                                                                 |
| log\_preds               | 予測サンプルをログするかどうか (デフォルトは `True`) 。                                                                                                                                                                                               |
| log\_preds\_every\_epoch | 毎エポックごとの予測をログするか終了時にログするか (デフォルトは `False`) 。                                                                                                                                                                                    |
| log\_model               | モデルをログするかどうか (デフォルトは False)。これには `SaveModelCallback` も必要です。                                                                                                                                                                  |
| model\_name              | 保存する `file` の名前、 `SaveModelCallback` を上書きします。                                                                                                                                                                                                |
| log\_dataset             | <ul><li><code>False</code> (デフォルト)</li><li><code>True</code> は learn.dls.path によって参照されるフォルダーをログします。</li><li>ログするフォルダーを明示的に参照するためのパスを定義することもできます。</li></ul><p><em>注: サブフォルダー「models」は常に無視されます。</em></p> |
| dataset\_name            | ログされたデータセットの名前 (デフォルトは `フォルダー名`)。                                                                                                                                                                                                           |
| valid\_dl                | 予測サンプルに使用されるアイテムを含む `DataLoaders` (デフォルトは `learn.dls.valid` のランダムアイテム) 。                                                                                                                                                  |
| n\_preds                 | ログされる予測の数 (デフォルトは 36)。                                                                                                                                                                                                                |
| seed                     | ランダムサンプルを定義するために使用される。                                                                                                                                                                                                                            |

カスタムワークフロー用には、データセットとモデルを手動でログすることができます:

* `log_dataset(path, name=None, metadata={})`
* `log_model(path, name=None, metadata={})`

_注: いかなるサブフォルダー「models」も無視されます。_

## 分散トレーニング

`fastai` はコンテキストマネージャ `distrib_ctx` を使用して分散トレーニングをサポートしています。W&B はこれを自動的にサポートし、マルチGPU実験を簡単にトラッキングできます。

以下は最小限の例です:

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

次に、ターミナルで次のコマンドを実行します:

```
$ torchrun --nproc_per_node 2 train.py
```

この場合、マシンには2つのGPUがあります。

  </TabItem>
  <TabItem value="notebook">

ノートブック内で直接分散トレーニングを実行できます！

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

### メインプロセスのみでのログ

上記の例では、`wandb` はプロセスごとに1つの run を起動します。トレーニング終了時に2つの run が作成されるため、混乱を招くことがあります。メインプロセスのみでログする場合は、手動でどのプロセスにいるかを検出し、他のすべてのプロセスで `wandb.init` を呼び出さないようにする必要があります。

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
次に、ターミナルに以下を実行します:

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

* [Visualize, track, and compare Fastai models](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA): 詳細なドキュメント付きのウォークスルー
* [Image Segmentation on CamVid](http://bit.ly/fastai-wandb): インテグレーションのサンプルユースケース