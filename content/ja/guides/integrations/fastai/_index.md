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

**fastai** を使用してモデルをトレーニングする場合、W&Bは `WandbCallback`を使用して簡単なインテグレーションを提供します。[インタラクティブなドキュメントと例 →](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA)で詳細を探ってください。

## サインアップしてAPIキーを作成

APIキーは、W&Bに対してあなたのマシンを認証します。ユーザープロフィールからAPIキーを生成できます。

{{% alert %}}
よりスムーズなアプローチとして、直接 [https://wandb.ai/authorize](https://wandb.ai/authorize) にアクセスしてAPIキーを生成できます。表示されたAPIキーをコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上隅のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックして表示されたAPIキーをコピーします。APIキーを非表示にするには、ページをリロードします。

## `wandb` ライブラリをインストールしてログイン

`wandb` ライブラリをローカルにインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})をあなたのAPIキーに設定します。

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

{{% tab header="Pythonノートブック" value="python-notebook" %}}

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

# wandb runを開始してログを記録
wandb.init(project="my_project")

# あるトレーニングフェーズ中のみログを記録する場合
learn.fit(..., cbs=WandbCallback())

# すべてのトレーニングフェーズで継続的にログを記録する場合
learn = learner(..., cbs=WandbCallback())
```

{{% alert %}}
Fastaiのバージョン1を使用している場合は、[Fastai v1のドキュメント]({{< relref path="v1.md" lang="ja" >}})を参照してください。
{{% /alert %}}

## WandbCallbackの引数

`WandbCallback`は次の引数を受け入れます：

| 引数                      | 説明                                                                                                                                                                                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| log                      | モデルの: `gradients` , `parameters`, `all` または `None` (デフォルト) をログに記録するかどうか。損失やメトリクスは常にログに記録されます。                                                                                                                                 |
| log_preds               | 予測サンプルをログに記録するかどうか（デフォルトは `True`）。                                                                                                                                                                                               |
| log_preds_every_epoch | 予測をエポックごとにログに記録するか、終了時に記録するか（デフォルトは `False`）。                                                                                                                                                                            |
| log_model               | モデルをログに記録するかどうか（デフォルトは `False`）。これには `SaveModelCallback` も必要です。                                                                                                                                                              |
| model_name              | 保存する `file` の名前、`SaveModelCallback` をオーバーライドします。                                                                                                                                                                                         |
| log_dataset             | <ul><li><code>False</code> (デフォルト)</li><li><code>True</code> は learn.dls.path に参照されるフォルダをログに記録します。</li><li>明示的にフォルダをログに記録するパスを定義できます。</li></ul><p><em>注: サブフォルダ "models" は常に無視されます。</em></p> |
| dataset_name            | ログされたデータセットの名前（デフォルトは `フォルダ名`）。                                                                                                                                                                                               |
| valid_dl                | 予測サンプルに使用される `DataLoaders`（デフォルトは `learn.dls.valid` からのランダムアイテム）。                                                                                                                                                          |
| n_preds                 | ログに記録される予測の数（デフォルトは36）。                                                                                                                                                                                                               |
| seed                     | ランダムサンプルを定義するために使用されます。                                                                                                                                                                                                            |

カスタムワークフローの場合、データセットやモデルを手動でログに記録できます：

* `log_dataset(path, name=None, metadata={})`
* `log_model(path, name=None, metadata={})`

_注: サブフォルダ "models" は無視されます。_

## 分散トレーニング

`fastai`は、コンテキストマネージャー `distrib_ctx` を使用して分散トレーニングをサポートしています。W&Bはこれを自動的にサポートし、マルチGPU実験をそのまま追跡することを可能にします。

この最小の例を確認してください：

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

次に、ターミナルで次のコマンドを実行します：

```shell
$ torchrun --nproc_per_node 2 train.py
```

この場合、マシンには2つのGPUがあります。

{{% /tab %}}
{{% tab header="Pythonノートブック" value="notebook" %}}

ノートブック内から直接分散トレーニングを実行できます。

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

### メインプロセスのみにログを記録

上記の例では、`wandb`はプロセスごとに1つのrunを起動します。トレーニング終了時には2つのrunsが得られます。これに混乱することがあるため、メインプロセスのみにログを記録したい場合があります。そのためには、手動でどのプロセスであるかを検出し、他のすべてのプロセスでrun（`wandb.init`の呼び出し）を作成しないようにします。

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
ターミナルで次のコマンドを呼び出します：

```
$ torchrun --nproc_per_node 2 train.py
```

{{% /tab %}}
{{% tab header="Pythonノートブック" value="notebook" %}}

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

* [Visualize, track, and compare Fastai models](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA): 丁寧にドキュメント化されたウォークスルー
* [Image Segmentation on CamVid](http://bit.ly/fastai-wandb): インテグレーションのサンプルユースケース