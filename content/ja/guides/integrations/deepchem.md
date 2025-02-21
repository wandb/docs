---
title: DeepChem
description: DeepChem ライブラリ と W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-deepchem
    parent: integrations
weight: 70
---

[DeepChem ライブラリ](https://github.com/deepchem/deepchem) は、創薬、材料科学、化学、および生物学における深層学習の利用を демократизирует するオープンソースのツールを提供します。この W&B インテグレーションにより、DeepChem を使用してモデルをトレーニングする際に、シンプルで使いやすい 実験管理 とモデルの チェックポイント が追加されます。

## 3 行のコードで DeepChem のログを記録

```python
logger = WandbLogger(…)
model = TorchModel(…, wandb_logger=logger)
model.fit(…)
```

{{< img src="/images/integrations/cd.png" alt="" >}}

## Report と Google Colab

[W&B with DeepChem: Molecular Graph Convolutional Networks](https://wandb.ai/kshen/deepchem_graphconv/reports/Using-W-B-with-DeepChem-Molecular-Graph-Convolutional-Networks--Vmlldzo4MzU5MDc?galleryTag=) の記事で、W&B DeepChem インテグレーションを使用して生成されたグラフの例をご覧ください。

すぐに動作する コードに取り組むには、こちらの [**Google Colab**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/deepchem/W%26B_x_DeepChem.ipynb) をご覧ください。

## 実験 を追跡

[KerasModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#keras-models) または [TorchModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#pytorch-models) タイプの DeepChem モデル用に W&B をセットアップします。

### サインアップして API キーを作成する

API キー は、W&B に対してマシンを認証します。API キー は、 ユーザー プロファイルから生成できます。

{{% alert %}}
より効率的なアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして API キー を生成できます。表示された API キー をコピーして、パスワード マネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上隅にある ユーザー プロファイル アイコンをクリックします。
2. [**User Settings**] を選択し、[**API Keys**] セクションまでスクロールします。
3. [**Reveal**] をクリックします。表示された API キー をコピーします。API キー を非表示にするには、ページをリロードします。

### `wandb` ライブラリをインストールしてログインする

`wandb` ライブラリをローカルにインストールしてログインするには:

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

### トレーニング データと 評価 データ を W&B に ログ 記録する

トレーニング 損失と 評価メトリクス は、W&B に自動的に ログ 記録できます。オプションの 評価 は、DeepChem [ValidationCallback](https://github.com/deepchem/deepchem/blob/master/deepchem/models/callbacks.py) を使用して有効にできます。`WandbLogger` は ValidationCallback コールバックを検出し、生成された メトリクス を ログ 記録します。

{{< tabpane text=true >}}

{{% tab header="TorchModel" value="torch" %}}

```python
from deepchem.models import TorchModel, ValidationCallback

vc = ValidationCallback(…)  # optional
model = TorchModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```

{{% /tab %}}

{{% tab header="KerasModel" value="keras" %}}

```python
from deepchem.models import KerasModel, ValidationCallback

vc = ValidationCallback(…)  # optional
model = KerasModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```

{{% /tab %}}

{{< /tabpane >}}
