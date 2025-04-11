---
title: fastai
cascade:
- url: guides/integrations/fastai/:filename
menu:
  default:
    identifier: ja-guides-integrations-fastai-_index
    parent: integrations
weight: 100
---

もしあなたが **fastai** を使ってモデルを訓練しているなら、W&B には `WandbCallback` を使用した簡単なインテグレーションがあります。[インタラクティブなドキュメントと例についてはこちらをご覧ください →](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA)

## 登録と APIキー の作成

APIキー は、あなたのマシンを W&B に認証します。APIキー は、ユーザープロフィールから生成できます。

{{% alert %}}
よりスムーズな方法として、直接 [https://wandb.ai/authorize](https://wandb.ai/authorize) にアクセスして APIキー を生成することができます。表示された APIキー をコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示された APIキー をコピーします。APIキー を非表示にするには、ページを再読み込みしてください。

## `wandb` ライブラリのインストールとログイン

`wandb` ライブラリをローカルにインストールしログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をあなたの APIキー に設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールしログインします。

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

## `learner` または `fit` メソッドに `WandbCallback` を追加する

```python
import wandb
from fastai.callback.wandb import *

# wandb run を開始してログをとる
wandb.init(project="my_project")

# トレーニングフェーズの一部のみログする場合
learn.fit(..., cbs=WandbCallback())

# すべてのトレーニングフェーズで継続的にログをとる場合
learn = learner(..., cbs=WandbCallback())
```

{{% alert %}}
Fastai のバージョン1を使用している場合は、[Fastai v1 ドキュメント]({{< relref path="v1.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

## WandbCallback 引数

`WandbCallback` は以下の引数を受け入れます:

| Args                     | 説明                                                                                                                                                                                                                                                        |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| log                      | モデルをログするかどうか: `gradients` 、`parameters`, `all` 、または `None` (デフォルト)。損失とメトリクスは常にログされます。                                                                                                                                  |
| log_preds               | 予測サンプルをログしたいかどうか (デフォルトは `True`)。                                                                                                                                                                                                     |
| log_preds_every_epoch | 予測をエポックごとにログするか、最後にログするか (デフォルトは `False`)                                                                                                                                                                                      |
| log_model               | モデルをログしたいかどうか (デフォルトは False)。これには `SaveModelCallback` も必要です。                                                                                                                                                                |
| model_name              | 保存する `file` の名前、`SaveModelCallback` をオーバーライドします。                                                                                                                                                                                         |
| log_dataset             | <ul><li><code>False</code> (デフォルト)</li><li><code>True</code> は learn.dls.path が参照するフォルダをログします。</li><li>ログするフォルダを参照するパスを明示的に定義できます。</li></ul><p><em>注: サブフォルダ "models" は常に無視されます。</em></p> |
| dataset_name            | ログされたデータセットの名前 (デフォルトは `フォルダ名`)。                                                                                                                                                                                                   |
| valid_dl                | 予測サンプルに使用する `DataLoaders` (デフォルトは `learn.dls.valid` からランダムなアイテム)                                                                                                                                                             |
| n_preds                 | ログする予測の数 (デフォルトは 36)。                                                                                                                                                                                                                        |
| seed                     | ランダムサンプルを定義するために使用します。                                                                                                                                                                                                               |

カスタムワークフローのために、データセットとモデルを手動でログすることができます:

* `log_dataset(path, name=None, metadata={})`
* `log_model(path, name=None, metadata={})`

_注: サブフォルダ "models" は無視されます。_

## 分散トレーニング

`fastai` はコンテキストマネージャー `distrib_ctx` を使用して分散トレーニングをサポートしています。W&B はこれを自動的にサポートし、マルチGPU実験をすぐにトラッキングできるようにします。

この簡単な例を確認してください:

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

そして、ターミナルで以下を実行します:

```shell
$ torchrun --nproc_per_node 2 train.py
```

この場合、マシンには 2 つの GPU があります。

{{% /tab %}}
{{% tab header="Python notebook" value="notebook" %}}

ノートブック内で直接分散トレーニングを実行することができます。

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

### メインプロセスのみでログを取る

上記の例では、`wandb` はプロセスごとに1 つの run を起動します。トレーニングの終了時には、2 つの run ができます。これが混乱を招くこともあり、メインプロセスだけでログを取りたい場合があります。そのためには、手動でどのプロセスにいるかを検出し、他のプロセスでは run (すなわち `wandb.init` の呼び出し) を作成しないようにする必要があります。

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
ターミナルで以下を実行します:

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

## 例

* [Visualize, track, and compare Fastai models](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA): 十分に文書化された手順
* [Image Segmentation on CamVid](http://bit.ly/fastai-wandb): インテグレーションのサンプルユースケース