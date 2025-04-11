---
title: DeepChem
description: DeepChem ライブラリ と W&B の統合方法について
menu:
  default:
    identifier: ja-guides-integrations-deepchem
    parent: integrations
weight: 70
---

The [DeepChem library](https://github.com/deepchem/deepchem) は、創薬、材料科学、化学、生物学におけるディープラーニングの利用を民主化するオープンソースツールを提供します。この W&B インテグレーションは、DeepChem を使用してモデルをトレーニングする際に、シンプルで使いやすい実験管理とモデルチェックポイントを追加します。

## DeepChem のロギングを 3 行のコードで

```python
logger = WandbLogger(…)
model = TorchModel(…, wandb_logger=logger)
model.fit(…)
```

{{< img src="/images/integrations/cd.png" alt="" >}}

## Report と Google Colab

W&B DeepChem インテグレーションを使用して生成されたチャートの例として、[Using W&B with DeepChem: Molecular Graph Convolutional Networks](https://wandb.ai/kshen/deepchem_graphconv/reports/Using-W-B-with-DeepChem-Molecular-Graph-Convolutional-Networks--Vmlldzo4MzU5MDc?galleryTag=) の記事を参照してください。

すぐに動作するコードを見たい場合は、この [**Google Colab**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/deepchem/W%26B_x_DeepChem.ipynb) をチェックしてください。

## 実験管理のトラッキング

[KerasModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#keras-models)または[TorchModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#pytorch-models) タイプの DeepChem モデルに W&B を設定します。

### サインアップと API キーの作成

APIキーは、あなたのマシンを W&B に認証します。APIキーはユーザープロフィールから生成できます。

{{% alert %}}
よりスムーズなアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして、APIキーを生成できます。表示されるAPIキーをコピーし、パスワードマネージャーなど安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示されたAPIキーをコピーします。APIキーを隠すには、ページを再読み込みします。

### `wandb` ライブラリのインストールとログイン

`wandb` ライブラリをローカルにインストールしてログインするには：

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をあなたのAPIキーに設定します。

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

### トレーニングと評価データを W&B にログする

トレーニング損失と評価メトリクスは、W&Bに自動的に記録されます。オプションの評価は、DeepChem の [ValidationCallback](https://github.com/deepchem/deepchem/blob/master/deepchem/models/callbacks.py) を使用して有効化できます。`WandbLogger` は ValidationCallback コールバックを検出し、生成されたメトリクスをログします。

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