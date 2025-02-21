---
title: DeepChem
description: W&B を DeepChem ライブラリ と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-deepchem
    parent: integrations
weight: 70
---

The [DeepChem library](https://github.com/deepchem/deepchem) は、創薬、材料科学、化学、生物学におけるディープラーニングの利用を一般化するためのオープンソースツールを提供します。この W&B インテグレーションは、DeepChem を用いてモデルをトレーニングする際に、シンプルで使いやすい実験管理とモデルチェックポイントを追加します。

## コード3行でのDeepChemロギング

```python
logger = WandbLogger(…)
model = TorchModel(…, wandb_logger=logger)
model.fit(…)
```

{{< img src="/images/integrations/cd.png" alt="" >}}

## レポートと Google Colab

W&B DeepChem インテグレーションを使用して生成された例のチャートについては、[Using W&B with DeepChem: Molecular Graph Convolutional Networks](https://wandb.ai/kshen/deepchem_graphconv/reports/Using-W-B-with-DeepChem-Molecular-Graph-Convolutional-Networks--Vmlldzo4MzU5MDc?galleryTag=)の記事を参照してください。

動作するコードをすぐに確認したい場合は、この [**Google Colab**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/deepchem/W%26B_x_DeepChem.ipynb) をチェックしてください。

## 実験追跡

[KerasModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#keras-models) または [TorchModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#pytorch-models) 型の DeepChem モデル用に W&B をセットアップします。

### サインアップと API キーの作成

API キーは、あなたのマシンを W&B に認証します。API キーはユーザープロファイルから生成できます。

{{% alert %}}
よりスムーズな方法として、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして API キーを生成できます。表示された API キーをコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロファイルアイコンをクリックします。
1. **ユーザー設定**を選択し、**API キー**セクションまでスクロールします。
1. **表示**をクリックします。表示された API キーをコピーします。API キーを非表示にするには、ページを再読み込みしてください。

### `wandb` ライブラリのインストールとログイン

ローカルに `wandb` ライブラリをインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` 環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をあなたの API キーに設定します。

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

{{% tab header="Python ノートブック" value="python-notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}

{{< /tabpane >}}

### トレーニングと評価データを W&B にログ

トレーニングの損失と評価メトリクスは、W&B に自動的にログを取ることができます。オプションの評価は DeepChem の [ValidationCallback](https://github.com/deepchem/deepchem/blob/master/deepchem/models/callbacks.py) を使用して有効にでき、`WandbLogger` は ValidationCallback コールバックを検出し、生成されたメトリクスをログに記録します。

{{< tabpane text=true >}}

{{% tab header="TorchModel" value="torch" %}}

```python
from deepchem.models import TorchModel, ValidationCallback

vc = ValidationCallback(…)  # オプション
model = TorchModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```

{{% /tab %}}

{{% tab header="KerasModel" value="keras" %}}

```python
from deepchem.models import KerasModel, ValidationCallback

vc = ValidationCallback(…)  # オプション
model = KerasModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```

{{% /tab %}}

{{< /tabpane >}}