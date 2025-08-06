---
title: DeepChem
description: W&B を DeepChem ライブラリと統合する方法
menu:
  default:
    identifier: deepchem
    parent: integrations
weight: 70
---

[DeepChem ライブラリ](https://github.com/deepchem/deepchem) は、創薬、材料科学、化学、生物学におけるディープラーニングの活用を広めるためのオープンソース ツールを提供しています。この W&B インテグレーションを利用することで、DeepChem を使ったモデルのトレーニング時に簡単かつ手軽に実験管理やモデルのチェックポイント保存を行うことができます。

## DeepChem のログ記録を 3 行で

```python
logger = WandbLogger(…)
model = TorchModel(…, wandb_logger=logger)
model.fit(…)
```

{{< img src="/images/integrations/cd.png" alt="DeepChem 分子分析" >}}

## レポートと Google Colab

W&B DeepChem インテグレーションで作成されたチャートの例については、[W&B with DeepChem: Molecular Graph Convolutional Networks](https://wandb.ai/kshen/deepchem_graphconv/reports/Using-W-B-with-DeepChem-Molecular-Graph-Convolutional-Networks--Vmlldzo4MzU5MDc?galleryTag=) の記事をご参照ください。

すぐにコードを動かしたい場合は、[Google Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/deepchem/W%26B_x_DeepChem.ipynb) をご覧ください。

## 実験管理をはじめよう

DeepChem の [KerasModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#keras-models) や [TorchModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#pytorch-models) タイプのモデルで W&B をセットアップしましょう。

### サインアップと API キーの作成

APIキーは、お使いのマシンを W&B と認証するために使用します。APIキーはユーザープロフィールから発行できます。

{{% alert %}}
より簡単な方法として、[W&B 認証ページ](https://wandb.ai/authorize)に直接アクセスしてAPIキーを発行することもできます。表示されたAPIキーをコピーし、パスワードマネージャーなど安全な場所に保管してください。
{{% /alert %}}

1. 画面右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選び、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックし、表示されたAPIキーをコピーします。APIキーを非表示にするにはページをリロードしてください。

### `wandb` ライブラリのインストールとログイン

ローカル環境で `wandb` ライブラリをインストールし、ログインする方法です。

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref "/guides/models/track/environment-variables.md" >}}) にAPIキーをセットします。

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

### トレーニングと評価データを W&B にログ

トレーニングの損失や評価メトリクスは自動的に W&B に記録されます。DeepChem の [ValidationCallback](https://github.com/deepchem/deepchem/blob/master/deepchem/models/callbacks.py) を使えば評価もオプションで有効にできます。`WandbLogger` は ValidationCallback コールバックを検出して生成されたメトリクスをログします。

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