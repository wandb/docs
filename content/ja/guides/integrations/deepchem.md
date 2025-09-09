---
title: DeepChem
description: W&B を DeepChem ライブラリと統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-deepchem
    parent: integrations
weight: 70
---

[DeepChem library](https://github.com/deepchem/deepchem) は、創薬・材料科学・化学・生物学におけるディープラーニング活用を民主化するためのオープンソースツールを提供します。W&B とのこのインテグレーションにより、DeepChem でモデルをトレーニングする際に、シンプルで使いやすい 実験管理 と モデルのチェックポイント作成 を追加できます。

## 3 行で DeepChem のログ記録

```python
logger = WandbLogger(…)
model = TorchModel(…, wandb_logger=logger)
model.fit(…)
```

{{< img src="/images/integrations/cd.png" alt="DeepChem による分子解析" >}}

## Report と Google Colab

W&B の DeepChem インテグレーションで生成されるチャート例は、[W&B と DeepChem を使う: 分子グラフ畳み込みネットワーク](https://wandb.ai/kshen/deepchem_graphconv/reports/Using-W-B-with-DeepChem-Molecular-Graph-Convolutional-Networks--Vmlldzo4MzU5MDc?galleryTag=) の記事をご覧ください。

すぐに動くコードを試すなら、こちらの [Google Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/deepchem/W%26B_x_DeepChem.ipynb) をチェックしてください。

## 実験をトラッキング

DeepChem の [KerasModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#keras-models) または [TorchModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#pytorch-models) タイプのモデルで W&B を設定します。

### サインアップして API キーを作成

API キーは、あなたのマシンを W&B に認証するためのものです。API キーはあなたのユーザープロフィールから生成できます。

{{% alert %}}
より手早く行うには、[W&B authorization page](https://wandb.ai/authorize) に直接アクセスして API キーを生成できます。表示された API キーをコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 画面右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示された API キーをコピーします。API キーを隠すにはページを再読み込みしてください。

### `wandb` ライブラリをインストールしてログイン

ローカルに `wandb` ライブラリをインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` の[環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})にあなたの API キーを設定します。

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

### トレーニングと評価のデータを W&B にログする

トレーニング損失や評価メトリクスは W&B に自動でログできます。任意の評価は DeepChem の [ValidationCallback](https://github.com/deepchem/deepchem/blob/master/deepchem/models/callbacks.py) を使って有効にできます。`WandbLogger` は ValidationCallback を検知し、生成されたメトリクスをログします。

{{< tabpane text=true >}}

{{% tab header="TorchModel" value="torch" %}}

```python
from deepchem.models import TorchModel, ValidationCallback

vc = ValidationCallback(…)  # 任意
model = TorchModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```

{{% /tab %}}

{{% tab header="KerasModel" value="keras" %}}

```python
from deepchem.models import KerasModel, ValidationCallback

vc = ValidationCallback(…)  # 任意
model = KerasModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```

{{% /tab %}}

{{< /tabpane >}}