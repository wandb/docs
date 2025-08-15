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

もし **fastai** を使ってモデルのトレーニングを行う場合、W&B は `WandbCallback` を利用した簡単なインテグレーションを提供しています。[ 例付きのインタラクティブなドキュメントで詳細をチェック →](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA)

## サインアップして API キーを作成する

API キーは、あなたのマシンと W&B の認証を行うためのものです。ユーザープロフィールから API キーを生成できます。

{{% alert %}}
より簡単な方法として、[W&B 認証ページ](https://wandb.ai/authorize) に直接アクセスして API キーを生成できます。表示された API キーをコピーして、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 画面右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリック。表示された API キーをコピーします。API キーを非表示にするには、ページをリロードしてください。

## `wandb` ライブラリのインストールとログイン

`wandb` ライブラリをローカルにインストールしてログインするには：

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) にあなたの API キーをセットします。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールしてログインします。



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

## `WandbCallback` を `learner` または `fit` メソッドへ追加

```python
import wandb
from fastai.callback.wandb import *

# wandb run のログを開始
wandb.init(project="my_project")

# ひとつのトレーニングフェーズのみログする場合
learn.fit(..., cbs=WandbCallback())

# 全てのトレーニングフェーズで継続的にログする場合
learn = learner(..., cbs=WandbCallback())
```

{{% alert %}}
Fastai バージョン 1 を使用している場合は [Fastai v1 docs]({{< relref path="v1.md" lang="ja" >}}) をご参照ください。
{{% /alert %}}

## WandbCallback の引数

`WandbCallback` で利用できる主な引数は以下の通りです：

| Args                     | 説明                                                                                                                                                                                                                                                           |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| log                      | モデルの`gradients`、`parameters`、`all`、または`None`（デフォルト）のいずれをログするか。ロスとメトリクスは常にログされます。                                                                                                                                   |
| log_preds               | 予測サンプルをログしたいかどうか（デフォルトは `True`）。                                                                                                                                                                                                       |
| log_preds_every_epoch | 予測のログを毎エポック行うか、最後にのみ行うか（デフォルトは `False`）                                                                                                                                                                                          |
| log_model               | モデル自体をログするかどうか（デフォルトは False）。これには `SaveModelCallback` も必要です。                                                                                                                                                                   |
| model_name              | 保存する `file` の名前。`SaveModelCallback` を上書き。                                                                                                                                                                                                         |
| log_dataset             | <ul><li><code>False</code>（デフォルト）</li><li><code>True</code> にすると learn.dls.path で指定したフォルダをログします。</li><li>明示的にパスを指定してログするフォルダを変更できます。</li></ul><p><em>※"models" サブフォルダは常に除外されます。</em></p>                |
| dataset_name            | ログするデータセットの名前（デフォルトはフォルダ名）。                                                                                                                                                                                                         |
| valid_dl                | 予測サンプルに使われる `DataLoaders`（デフォルトは `learn.dls.valid` からランダムに選択）。                                                                                                                                                                      |
| n_preds                 | ログする予測数（デフォルトは 36）。                                                                                                                                                                                                                           |
| seed                    | ランダムサンプル選択用のシード。                                                                                                                                                                                                                                |

カスタムワークフロー向けには、データセットやモデルの手動ログも可能です：

* `log_dataset(path, name=None, metadata={})`
* `log_model(path, name=None, metadata={})`

_※ "models" サブフォルダは無視されます。_

## 分散トレーニング

`fastai` では、コンテキストマネージャー `distrib_ctx` により分散トレーニングが可能です。W&B はこれを自動でサポートし、Multi-GPU 実験も簡単にトラッキングできます。

以下はミニマルな例です：

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

その後、ターミナルで以下コマンドを実行してください：

```shell
$ torchrun --nproc_per_node 2 train.py
```

ここではマシンに GPU が 2 台ある場合の例です。

{{% /tab %}}
{{% tab header="Python notebook" value="notebook" %}}

ノートブック内でも分散トレーニングが直接実行できます。

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

### メインプロセスのみでログする

上記の例では、`wandb` がプロセスごとに 1 run を作成します。そのためトレーニング終了後に 2 つの run ができます。混乱を避けたい場合や、メインプロセスのみでログしたい場合には、どのプロセスで実行中かを自分で判定し、他のプロセスでは run（`wandb.init` の呼び出し）を作成しないようにしてください。

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
ターミナルで以下を実行：

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

## 事例集

* [Visualize, track, and compare Fastai models](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA): 詳細なウォークスルー付き。
* [Image Segmentation on CamVid](https://bit.ly/fastai-wandb): インテグレーションのユースケース例。