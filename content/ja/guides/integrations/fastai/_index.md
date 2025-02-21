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

**fastai** を使用してモデルをトレーニングする場合、W&B には `WandbCallback` を使用した簡単な インテグレーション があります。[インタラクティブなドキュメントと例はこちら →](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA)

## サインアップして API キーを作成する

API キー は、お使いのマシンを W&B に対して認証します。API キー は、 ユーザー プロフィールから生成できます。

{{% alert %}}
より効率的なアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして API キー を生成できます。表示された API キー をコピーして、パスワード マネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上隅にある ユーザー プロフィール アイコンをクリックします。
2. [**User Settings**]を選択し、[**API Keys**]セクションまでスクロールします。
3. [**Reveal**]をクリックします。表示された API キー をコピーします。API キー を非表示にするには、ページをリロードします。

## `wandb` ライブラリをインストールしてログインする

ローカルに `wandb` ライブラリをインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を API キー に設定します。

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

## `WandbCallback` を `learner` または `fit` メソッドに追加する

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
Fastai の バージョン 1 を使用している場合は、[Fastai v1 のドキュメント]({{< relref path="v1.md" lang="ja" >}})を参照してください。
{{% /alert %}}

## WandbCallback の引数

`WandbCallback` は、次の 引数 を受け入れます。

| 引数                     | 説明                                                                                                                                                                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| log                      | モデルの `gradients` 、 `parameters` 、 `all` または `None` (デフォルト) を ログ に記録するかどうか。損失と メトリクス は常に ログ に記録されます。                                                                                                                                 |
| log_preds               | 予測 サンプル を ログ に記録するかどうか (デフォルトは `True`)。                                                                                                                                                                                               |
| log_preds_every_epoch | エポック ごとに予測を ログ に記録するか、最後に ログ に記録するか (デフォルトは `False`)                                                                                                                                                                                    |
| log_model               | モデルを ログ に記録するかどうか (デフォルトは False)。これには `SaveModelCallback` も必要です。                                                                                                                                                                  |
| model_name              | 保存する `file` の名前で、 `SaveModelCallback` をオーバーライドします。                                                                                                                                                                                                |
| log_dataset             | <ul><li><code>False</code> (デフォルト)</li><li><code>True</code> は、learn.dls.path によって参照されるフォルダーを ログ に記録します。</li><li>ログ に記録するフォルダーを参照するために、パスを明示的に定義できます。</li></ul><p><em>注: サブフォルダー "models" は常に無視されます。</em></p> |
| dataset_name            | ログ に記録された データセット の名前 (デフォルトは `folder name`)。                                                                                                                                                                                                           |
| valid_dl                | 予測 サンプル に使用される項目を含む `DataLoaders` (デフォルトは `learn.dls.valid` からのランダムな項目。                                                                                                                                                  |
| n_preds                 | ログ に記録された予測の数 (デフォルトは 36)。                                                                                                                                                                                                                |
| seed                     | ランダム サンプル を定義するために使用されます。                                                                                                                                                                                                                            |

カスタム ワークフロー の場合は、 データセット と モデル を手動で ログ に記録できます。

* `log_dataset(path, name=None, metadata={})`
* `log_model(path, name=None, metadata={})`

_注: サブフォルダー "models" は無視されます。_

## 分散 トレーニング

`fastai` は、コンテキスト マネージャー `distrib_ctx` を使用して分散 トレーニング をサポートしています。W&B はこれを自動的にサポートし、すぐに使える Multi-GPU 実験 を追跡できます。

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

次に、 ターミナル で次を実行します。

```shell
$ torchrun --nproc_per_node 2 train.py
```

この場合、マシンには 2 つの GPU があります。

{{% /tab %}}
{{% tab header="Python notebook" value="notebook" %}}

これで、 ノートブック 内で分散 トレーニング を直接実行できるようになりました。

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

### メイン プロセス でのみ ログ に記録する

上記の例では、 `wandb` は プロセス ごとに 1 つの run を起動します。トレーニング の最後に、2 つの run が作成されます。これは混乱を招く場合があるため、メイン プロセス でのみ ログ に記録したい場合があります。これを行うには、どの プロセス にいるかを 手動で検出し、他のすべての プロセス で run を作成しないようにする必要があります ( `wandb.init` を呼び出さない)。

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
ターミナル で次を呼び出します。

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

* [Visualize, track, and compare Fastai models](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA): 完全に文書化されたチュートリアル
* [Image Segmentation on CamVid](http://bit.ly/fastai-wandb): インテグレーション の サンプル ユースケース
