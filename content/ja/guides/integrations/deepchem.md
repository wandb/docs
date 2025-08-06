---
title: DeepChem
description: W&B を DeepChem ライブラリと統合する方法
menu:
  default:
    identifier: ja-guides-integrations-deepchem
    parent: integrations
weight: 70
---

[DeepChem library](https://github.com/deepchem/deepchem) は、創薬、材料科学、化学、生物学におけるディープラーニング活用の民主化を目指して、オープンソースのツールを提供しています。この W&B とのインテグレーションによって、DeepChem でモデルをトレーニングする際に、シンプルかつ簡単に実験管理やモデルのチェックポイント保存ができるようになります。

## DeepChem のロギングを3行で

```python
logger = WandbLogger(…)
model = TorchModel(…, wandb_logger=logger)
model.fit(…)
```

{{< img src="/images/integrations/cd.png" alt="DeepChem 分子解析" >}}

## Report と Google Colab

W&B DeepChem インテグレーションを使って生成されたチャートの例については、[W&B with DeepChem: Molecular Graph Convolutional Networks](https://wandb.ai/kshen/deepchem_graphconv/reports/Using-W-B-with-DeepChem-Molecular-Graph-Convolutional-Networks--Vmlldzo4MzU5MDc?galleryTag=) の記事をご覧ください。

すぐにコードを試してみたい場合は、こちらの [Google Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/deepchem/W%26B_x_DeepChem.ipynb) をチェックしてください。

## 実験をトラッキングする

DeepChem の [KerasModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#keras-models) または [TorchModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#pytorch-models) タイプのモデルで W&B を設定しましょう。

### サインアップと API キーの作成

APIキーは、マシンを W&B に認証するためのものです。APIキーはユーザープロフィールから発行できます。

{{% alert %}}
より簡単に API キーを取得したい場合は、[W&B 認証ページ](https://wandb.ai/authorize)へ直接アクセスしてください。表示された API キーをコピーし、パスワードマネージャーなどの安全な場所に保存しましょう。
{{% /alert %}}

1. 画面右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックして API キーを表示し、コピーします。API キーを隠したい場合はページを再読み込みしてください。

### `wandb` ライブラリのインストールとログイン

ローカル環境に `wandb` ライブラリをインストールしてログインする手順です。

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) に API キーを設定します。

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

### トレーニングおよび評価データを W&B へログ

トレーニング損失や評価メトリクスは、自動的に W&B へログできます。DeepChem の [ValidationCallback](https://github.com/deepchem/deepchem/blob/master/deepchem/models/callbacks.py) を使うことで、オプションの評価も有効にできます。`WandbLogger` は ValidationCallback コールバックを検知し、生成されたメトリクスを自動でログします。

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