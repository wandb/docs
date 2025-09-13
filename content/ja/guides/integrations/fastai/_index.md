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

fastai で モデル をトレーニングしている場合、W&B には `WandbCallback` を使った簡単なインテグレーションがあります。詳しくは [ 例付きのインタラクティブなドキュメント →](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA) を参照してください。

## サインアップして APIキー を作成

APIキー は、あなたのマシンを W&B に認証するためのものです。APIキー は ユーザー プロフィールから生成できます。

{{% alert %}}
よりスムーズに行うには、[W&B authorization page](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピーして、パスワード マネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上の ユーザー プロフィール アイコンをクリックします。
1. **User Settings** を選び、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックして表示された APIキー をコピーします。APIキー を隠すには、ページを再読み込みします。

## `wandb` ライブラリをインストールしてログイン

ローカルに `wandb` ライブラリをインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [environment variable]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) にあなたの APIキー を設定します。

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

## `WandbCallback` を `learner` または `fit` メソッドに追加

```python
import wandb
from fastai.callback.wandb import *

# wandb の run の ログを開始
wandb.init(project="my_project")

# 1 回のトレーニング フェーズの間だけ ログ する場合
learn.fit(..., cbs=WandbCallback())

# すべてのトレーニング フェーズで継続的に ログ する場合
learn = learner(..., cbs=WandbCallback())
```

{{% alert %}}
Fastai の バージョン 1 を使う場合は、[Fastai v1 docs]({{< relref path="v1.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

## WandbCallback の 引数

`WandbCallback` は次の引数を受け付けます。

| 引数                     | 説明                                                                                                                                                                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| log                      | モデル の次のいずれかを ログ するか: `gradients`、`parameters`、`all` または `None`（デフォルト）。損失と メトリクス は常に ログ されます。                                                                                                                                 |
| log_preds               | 予測 サンプルを ログ するかどうか（デフォルトは `True`）。                                                                                                                                                                                               |
| log_preds_every_epoch | 毎 エポック ごとに 予測 を ログ するか、それとも最後に ログ するか（デフォルトは `False`）。                                                                                                                                                                                    |
| log_model               | モデル を ログ するかどうか（デフォルトは False）。これには `SaveModelCallback` も必要です。                                                                                                                                                                  |
| model_name              | 保存する `file` の 名前。`SaveModelCallback` を上書きします。                                                                                                                                                                                                |
| log_dataset             | <ul><li><code>False</code>（デフォルト）</li><li><code>True</code> は learn.dls.path で参照されるフォルダを ログ します。</li><li>どのフォルダを ログ するかを指す パス を明示的に指定できます。</li></ul><p><em>注意: サブフォルダ "models" は常に無視されます。</em></p> |
| dataset_name            | ログ された データセット の 名前（デフォルトは `folder name`）。                                                                                                                                                                                                           |
| valid_dl                | 予測 サンプルに使用するアイテムを含む `DataLoaders`（デフォルトは `learn.dls.valid` からのランダムなアイテム）。                                                                                                                                                  |
| n_preds                 | ログ する 予測 の 数（デフォルトは 36）。                                                                                                                                                                                                                |
| seed                     | ランダム サンプルを決めるために使用。                                                                                                                                                                                                                            |

カスタム ワークフロー向けに、データセット と モデル を手動で ログ することもできます:

* `log_dataset(path, name=None, metadata={})`
* `log_model(path, name=None, metadata={})`

_ 注意: どのサブフォルダ "models" も無視されます。_

## 分散 トレーニング

`fastai` は、コンテキスト マネージャー `distrib_ctx` を使って分散 トレーニング をサポートしています。W&B はこれを自動でサポートし、マルチ GPU の 実験 をそのまま トラッキング できるようにします。

最小の例を確認してください:

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

次に、ターミナル で 以下を実行します:

```shell
$ torchrun --nproc_per_node 2 train.py
```

この場合、マシンには 2 つの GPU があります。

{{% /tab %}}
{{% tab header="Python notebook" value="notebook" %}}

ノートブック 内で直接 分散 トレーニング を実行できます。

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

### メイン プロセス でのみ ログ

上の例では、`wandb` は プロセス ごとに 1 つの run を起動します。トレーニングの最後には 2 つの run ができます。これは混乱を招くことがあるため、メイン プロセス のみで ログ したい場合があります。その場合は、自分がどのプロセスにいるかを手動で判定し、他のプロセスでは run を作成しない（`wandb.init` を呼ばない）ようにします。

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
ターミナル で次を実行します:

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

* [Fastai モデル を可視化・トラッキング・比較](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA): 丁寧に解説したチュートリアル。
* [CamVid での 画像セグメンテーション](https://bit.ly/fastai-wandb): この インテグレーション の 代表的な ユースケース。