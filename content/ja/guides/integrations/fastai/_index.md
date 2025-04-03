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

**fastai** を使用してモデルをトレーニングする場合、W&B には `WandbCallback` を使用した簡単なインテグレーションがあります。[インタラクティブなドキュメントと例はこちら →](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA)

## サインアップして API キーを作成する

API キーは、W&B に対してお客様のマシンを認証します。API キーは、ユーザー プロフィールから生成できます。

{{% alert %}}
より効率的なアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして API キーを生成できます。表示された API キーをコピーして、パスワード マネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上隅にあるユーザー プロフィール アイコンをクリックします。
2. **ユーザー設定** を選択し、**API キー** セクションまでスクロールします。
3. **表示** をクリックします。表示された API キーをコピーします。API キーを非表示にするには、ページをリロードします。

## `wandb` ライブラリをインストールしてログインする

`wandb` ライブラリをローカルにインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を API キーに設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` ライブラリをインストールしてログインします。

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

# start logging a wandb run
wandb.init(project="my_project")

# To log only during one training phase
learn.fit(..., cbs=WandbCallback())

# To log continuously for all training phases
learn = learner(..., cbs=WandbCallback())
```

{{% alert %}}
Fastai のバージョン 1 を使用する場合は、[Fastai v1 ドキュメント]({{< relref path="v1.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

## WandbCallback 引数

`WandbCallback` は、次の引数を受け入れます。

| Args                     | Description                                                                                                                                                                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| log                      | モデルのログを記録するかどうか: `gradients` 、`parameters`、`all` または `None` (デフォルト)。損失とメトリクスは常にログに記録されます。                                                                                                                                 |
| log_preds               | 予測サンプルをログに記録するかどうか (デフォルトは `True`)。                                                                                                                                                                                               |
| log_preds_every_epoch | エポックごとに予測をログに記録するか、最後にログに記録するか (デフォルトは `False`)                                                                                                                                                                                    |
| log_model               | モデルをログに記録するかどうか (デフォルトは False)。これには `SaveModelCallback` も必要です                                                                                                                                                                  |
| model_name              | 保存する `file` の名前。`SaveModelCallback` をオーバーライドします                                                                                                                                                                                                |
| log_dataset             | <ul><li><code>False</code> (デフォルト)</li><li><code>True</code> は、learn.dls.path で参照されるフォルダーをログに記録します。</li><li>パスを明示的に定義して、ログに記録するフォルダーを参照できます。</li></ul><p><em>注: サブフォルダー "models" は常に無視されます。</em></p> |
| dataset_name            | ログに記録されたデータセットの名前 (デフォルトは `folder name`)。                                                                                                                                                                                                           |
| valid_dl                | 予測サンプルに使用されるアイテムを含む `DataLoaders` (デフォルトは `learn.dls.valid` からのランダムなアイテム。                                                                                                                                                  |
| n_preds                 | ログに記録された予測の数 (デフォルトは 36)。                                                                                                                                                                                                                |
| seed                     | ランダム サンプルを定義するために使用されます。                                                                                                                                                                                                                            |

カスタム ワークフローでは、データセットとモデルを手動でログに記録できます。

* `log_dataset(path, name=None, metadata={})`
* `log_model(path, name=None, metadata={})`

_注: サブフォルダー "models" はすべて無視されます。_

## 分散トレーニング

`fastai` は、コンテキスト マネージャー `distrib_ctx` を使用して分散トレーニングをサポートします。W&B はこれを自動的にサポートし、すぐに使える Multi-GPU の Experiments を追跡できるようにします。

この最小限の例を確認してください。

{{< tabpane text=true >}}
{{% tab header="スクリプト" value="script" %}}

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

次に、ターミナルで次を実行します。

```shell
$ torchrun --nproc_per_node 2 train.py
```

この場合、マシンには 2 つの GPU があります。

{{% /tab %}}
{{% tab header="Python ノートブック" value="notebook" %}}

ノートブック内で分散トレーニングを直接実行できるようになりました。

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

### メイン プロセスでのみログを記録する

上記の例では、`wandb` はプロセスごとに 1 つの run を起動します。トレーニングの最後に、2 つの run が作成されます。これは混乱を招く可能性があるため、メイン プロセスでのみログに記録したい場合があります。そのためには、どのプロセスに手動でいるかを検出し、他のすべてのプロセスで run を作成 ( `wandb.init` を呼び出す) しないようにする必要があります。

{{< tabpane text=true >}}
{{% tab header="スクリプト" value="script" %}}

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
ターミナルで次を呼び出します。

```
$ torchrun --nproc_per_node 2 train.py
```

{{% /tab %}}
{{% tab header="Python ノートブック" value="notebook" %}}

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

* [Fastai モデルの可視化、追跡、比較](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA): 完全に文書化されたチュートリアル
* [CamVid での画像セグメンテーション](http://bit.ly/fastai-wandb): インテグレーションのサンプル ユースケース
