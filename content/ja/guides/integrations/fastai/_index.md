---
title: fastai
menu:
  default:
    identifier: README
    parent: integrations
cascade:
- url: guides/integrations/fastai/:filename
weight: 100
---

もし **fastai** を使ってモデルのトレーニングを行っている場合、W&B には `WandbCallback` を利用した簡単なインテグレーションがあります。詳しくは[インタラクティブなドキュメントとサンプルはこちら →](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA)

## サインアップと API キーの作成

APIキーは、あなたのマシンを W&B に認証するためのものです。APIキーはユーザープロファイルから発行できます。

{{% alert %}}
もっと手軽な方法としては、[W&B 認証ページ](https://wandb.ai/authorize) から直接 APIキーを発行できます。表示された API キーをコピーして、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 画面右上のユーザープロファイルアイコンをクリックします。
1. **User Settings** を選び、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックし、表示された API キーをコピーします。APIキーを再度非表示にするにはページをリロードしてください。

## `wandb` ライブラリのインストールとログイン

ローカルで `wandb` ライブラリをインストールし、ログインします：

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref "/guides/models/track/environment-variables.md" >}}) に、あなたの API キーを設定してください。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールし、ログインします。



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

## `WandbCallback` を `learner` もしくは `fit` メソッドに追加

```python
import wandb
from fastai.callback.wandb import *

# wandb run の ログを開始
wandb.init(project="my_project")

# 1つのトレーニングフェーズのみログを記録する場合
learn.fit(..., cbs=WandbCallback())

# すべてのトレーニングフェーズで継続的にログを記録する場合
learn = learner(..., cbs=WandbCallback())
```

{{% alert %}}
Fastai のバージョン 1 を利用している場合は、[Fastai v1 のドキュメント]({{< relref "v1.md" >}}) を参照してください。
{{% /alert %}}

## WandbCallback の引数

`WandbCallback` は次の引数を受け取ります：

| 引数                     | 説明                                                                                                                                                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| log                      | モデルの `gradients`、`parameters`、`all`、または `None`（デフォルト）をログに記録するかどうか。損失値とメトリクスは常にログに記録されます。                                                                                                   |
| log_preds               | 予測サンプルをログに記録するかどうか（デフォルトは `True`）。                                                                                                                                                                                             |
| log_preds_every_epoch | 毎エポックごと、または最後だけ 予測結果をログに記録するか（デフォルトは `False`）                                                                                                                                                                          |
| log_model               | モデルをログに記録するかどうか（デフォルトはFalse）。`SaveModelCallback` も必要です。                                                                                                                                                                  |
| model_name              | 保存するファイル名。`SaveModelCallback` の設定を上書きします。                                                                                                                                                                                             |
| log_dataset             | <ul><li><code>False</code>（デフォルト）</li><li><code>True</code> で learn.dls.path が指すフォルダをログします。</li><li>または明示的なパスが指定できます。</li></ul><p><em>※「models」サブフォルダは必ず無視されます。</em></p>               |
| dataset_name            | ログに記録するデータセットの名前（デフォルトはフォルダ名）                                                                                                                                                                                                 |
| valid_dl                | 予測サンプルで使用する `DataLoaders`（デフォルトは `learn.dls.valid` からランダムに選ばれたアイテム）                                                                                                                                                    |
| n_preds                 | ログに記録する予測の数（デフォルトは36）。                                                                                                                                                                                                              |
| seed                    | ランダムサンプルを定義する際に利用されます。                                                                                                                                                                                                            |

カスタムワークフローの場合は、次のようにデータセットやモデルを手動でログできます：

* `log_dataset(path, name=None, metadata={})`
* `log_model(path, name=None, metadata={})`

_※「models」サブフォルダは必ず無視されます。_

## 分散トレーニング

`fastai` は `distrib_ctx` コンテキストマネージャ経由で分散トレーニングをサポートします。W&B 側も自動でサポートしており、複数 GPU での実験もそのまま追跡できます。

以下のミニマルなサンプルもご参照ください：

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

ターミナルで以下のように実行します：

```shell
$ torchrun --nproc_per_node 2 train.py
```

この例では、マシンに GPU が2枚ある場合となっています。

{{% /tab %}}
{{% tab header="Python notebook" value="notebook" %}}

ノートブック環境でも分散トレーニングが可能です。

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

### メインプロセスのみログに記録したい場合

上記のサンプルでは `wandb` がプロセスごとに1つずつ run を作成します。そのため、トレーニング終了時には 2つの run ができます。これが紛らわしい場合、メインプロセスのみログしたい場合は、どのプロセスで実行しているかを手動で判定し、他のプロセスでは run（`wandb.init`）を作成しないようにしてください。

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
ターミナルで次のように実行します：

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

## 事例紹介

* [Visualize, track, and compare Fastai models](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA): とても詳しく説明されたチュートリアル
* [Image Segmentation on CamVid](https://bit.ly/fastai-wandb): インテグレーションのユースケース例