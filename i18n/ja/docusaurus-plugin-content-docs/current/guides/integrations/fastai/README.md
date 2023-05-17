import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Fastai

**fastai** を使ってモデルをトレーニングしている場合、`WandbCallback` を利用して W&B と簡単に連携できます。詳細は[インタラクティブな解説と例 →](https://app.wandb.ai/borisd13/demo\_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA) で確認できます。

## W&B でのログの開始

**a)** [https://wandb.ai/site](https://wandb.ai/site) で無料アカウントに[登録](https://wandb.ai/site)し、wandb アカウントにログインします。

**b)** Python 3 の環境で `pip` を使ってマシンに wandb ライブラリをインストールします。

**c)** マシン上の wandb ライブラリにログインします。APIキーはこちらで見つけられます：[https://wandb.ai/authorize](https://wandb.ai/authorize)。

<Tabs
  defaultValue="script"
  values={[
    {label: 'コマンドライン', value: 'script'},
    {label: 'ノートブック', value: 'notebook'},
  ]}>
  <TabItem value="script">

```python
pip install wandb
wandb login
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install wandb

import wandb
wandb.login()
```

  </TabItem>
</Tabs>


次に、`WandbCallback` を `learner` または `fit` メソッドに追加します:

```python
import wandb
from fastai.callback.wandb import *

# wandb run のログ記録を開始
wandb.init(project='my_project')

# トレーニングの一部でのみログを記録する場合
learn.fit(..., cbs=WandbCallback())

# すべてのトレーニングフェーズで連続してログを記録する場合
learn = learner(..., cbs=WandbCallback())
```

:::info
Fastaiのバージョン1を使用している場合は、 [Fastai v1ドキュメント](v1.md) を参照してください。
:::
## WandbCallback 引数

`WandbCallback`は以下の引数を受け付けます:

| Args                     | 説明                                                                                                                                                                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| log                      | モデルの "`gradients`" , "`parameters`", "`all`" または `None` (デフォルト) をログするかどうか。損失とメトリクスは常にログされます。                                                                                                                                 |
| log\_preds               | 予測サンプルをログするかどうか（デフォルトは `True`）。                                                                                                                                                                                                 |
| log\_preds\_every\_epoch | 予測を毎エポックログするか、最後にログするか（デフォルトは `False`）。                                                                                                                                                                                    |
| log\_model               | モデルをログするかどうか（デフォルトは False）。これには `SaveModelCallback` も必要です。                                                                                                                                                                  |
| model\_name              | 保存する `file` の名前。`SaveModelCallback` を上書きします。                                                                                                                                                                                                |
| log\_dataset             | <ul><li><code>False</code>（デフォルト）</li><li><code>True</code> は、learn.dls.path で参照されるフォルダをログします。</li><li>ログするフォルダを明示的に参照するパスを定義できます。</li></ul><p><em>注意：サブフォルダ「models」は常に無視されます。</em></p> |
| dataset\_name            | ログされたデータセットの名前（デフォルトは `folder name`）。                                                                                                                                                                                                           |
| valid\_dl                | 予測サンプルに使用されるアイテムが含まれる `DataLoaders`（デフォルトは `learn.dls.valid` のランダムアイテム）。                                                                                                                                                  |
| n\_preds                 | ログされる予測の数（デフォルトは 36）。                                                                                                                                                                                                                |
| seed                     | ランダムサンプルを定義するために使用されるシード。                                                                                                                                                                                                                            |

カスタムワークフローの場合、データセットとモデルを手動でログすることができます。

* `log_dataset(path, name=None, metadata={})`
* `log_model(path, name=None, metadata={})`

_注意: 任意のサブフォルダ "models"は無視されます。_

## 分散トレーニング

`fastai`は、コンテキストマネージャ`distrib_ctx`を使用することで、分散トレーニングをサポートしています。W&Bはこれを自動的にサポートし、マルチGPU実験をすぐにトラッキングすることができます。

以下に簡単な例を示します。

<Tabs
  defaultValue="script"
  values={[
    {label: 'スクリプト', value: 'script'},
    {label: 'ノートブック', value: 'notebook'},
  ]}>
  <TabItem value="script">

<!-- {% code title="train.py" %} -->
```python
import wandb
from fastai.vision.all import *
from fastai.distributed import *
from fastai.callback.wandb import WandbCallback

wandb.require(experiment="service")
path = rank0_first(lambda: untar_data(URLs.PETS)/'images')

def train():
    dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2,
        label_func=lambda x: x[0].isupper(), item_tfms=Resize(224))
    wandb.init('fastai_ddp', entity='capecape')
    cb = WandbCallback()
    learn = vision_learner(dls, resnet34, metrics=error_rate, cbs=cb).to_fp16()
    with learn.distrib_ctx(sync_bn=False):
        learn.fit(1)
        
if __name__ == "__main__":
    train()
```
その後、ターミナルで以下のコマンドを実行します:

```
$ torchrun --nproc_per_node 2 train.py
```

この例では、マシンには2つのGPUがあります。

  </TabItem>
  <TabItem value="notebook">

これで、ノートブック内で分散トレーニングを直接実行できるようになりました！

```python
import wandb
from fastai.vision.all import *

from accelerate import notebook_launcher
from fastai.distributed import *
from fastai.callback.wandb import WandbCallback

wandb.require(experiment="service")
path = untar_data(URLs.PETS)/'images'

def train():
    dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2,
        label_func=lambda x: x[0].isupper(), item_tfms=Resize(224))
    wandb.init('fastai_ddp', entity='capecape')
    cb = WandbCallback()
    learn = vision_learner(dls, resnet34, metrics=error_rate, cbs=cb).to_fp16()
    with learn.distrib_ctx(in_notebook=True, sync_bn=False):
        learn.fit(1)

```
notebook_launcher(train, num_processes=2)
```

  </TabItem>
</Tabs>

### メインプロセスでのみログを記録する

上記の例では、`wandb`はプロセスごとに1つのrunを起動します。トレーニングが終了すると、2つのrunができます。これは時々混乱を招くことがあり、メインプロセスでのみログを記録したい場合があります。これを行うには、手動でどのプロセスにいるかを検出し、runを作成しないようにする必要があります（すべての他のプロセスで`wandb.init`を呼び出さないようにする）。

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
path = rank0_first(lambda: untar_data(URLs.PETS)/'images')

def train():
    cb = []
    dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2,
        label_func=lambda x: x[0].isupper(), item_tfms=Resize(224))
    if rank_distrib() == 0:
        run = wandb.init('fastai_ddp', entity='capecape')
        cb = WandbCallback()
    learn = vision_learner(dls, resnet34, metrics=error_rate, cbs=cb).to_fp16()
    with learn.distrib_ctx(sync_bn=False):
        learn.fit(1)

if __name__ == "__main__":
    train()
```
ターミナルで呼び出すには：

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
path = untar_data(URLs.PETS)/'images'

def train():
    cb = []
    dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2,
        label_func=lambda x: x[0].isupper(), item_tfms=Resize(224))
    if rank_distrib() == 0:
        run = wandb.init('fastai_ddp', entity='capecape')
        cb = WandbCallback()
    learn = vision_learner(dls, resnet34, metrics=error_rate, cbs=cb).to_fp16()
    with learn.distrib_ctx(in_notebook=True, sync_bn=False):
        learn.fit(1)
notebook_launcher(train, num_processes=2)

```

  </TabItem>

</Tabs>

## 例

* [Fastaiモデルの可視化・トラッキング・比較](https://app.wandb.ai/borisd13/demo_config/reports/Visualize-track-compare-Fastai-models--Vmlldzo4MzAyNA): 詳細にドキュメント化されたウォークスルー

* [CamVidにおける画像セグメンテーション](http://bit.ly/fastai-wandb): この統合のサンプルユースケース