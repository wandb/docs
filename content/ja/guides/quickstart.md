---
title: W&B クイックスタート
description: W&B クイックスタート
menu:
  default:
    identifier: ja-guides-quickstart
    parent: guides
url: /ja/quickstart
weight: 2
---

W&B をインストールして、お好きな規模の機械学習実験をトラッキング、可視化、管理しましょう。

## サインアップしてAPIキーを作成する

W&Bとマシンを認証するには、ユーザープロファイルまたは[wandb.ai/authorize](https://wandb.ai/authorize)でAPIキーを生成します。APIキーをコピーして安全に保管してください。

## `wandb` ライブラリをインストールしてログインする

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を設定します。

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

{{% tab header="Python ノートブック" value="notebook" %}}

```notebook
!pip install wandb
import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## ランを開始してハイパーパラメーターをトラックする

Python スクリプトやノートブックで、[`wandb.init()`]({{< relref path="/ref/python/run.md" lang="ja" >}})を使用して W&B のランオブジェクトを初期化します。`config` パラメータには辞書を使用してハイパーパラメーターの名前と値を指定します。

```python
run = wandb.init(
    project="my-awesome-project",  # プロジェクトを指定する
    config={                        # ハイパーパラメーターとメタデータをトラックする
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```

W&B のコア要素として [ラン]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) は使用され、[メトリクスをトラックする]({{< relref path="/guides/models/track/" lang="ja" >}})、[ログを作成する]({{< relref path="/guides/models/track/log/" lang="ja" >}}) など様々なことができます。

## コンポーネントを組み立てる

この模擬トレーニングスクリプトは、W&Bにシミュレートされた精度と損失のメトリクスをログします:

```python
# train.py
import wandb
import random

wandb.login()

epochs = 10
lr = 0.01

run = wandb.init(
    project="my-awesome-project",    # プロジェクトを指定する
    config={                         # ハイパーパラメーターとメタデータをトラックする
        "learning_rate": lr,
        "epochs": epochs,
    },
)

offset = random.random() / 5
print(f"lr: {lr}")

# トレーニングランをシミュレーション
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()
```

[wandb.ai/home](https://wandb.ai/home) にアクセスして、記録された精度や損失メトリクス、および各トレーニングステップでの変化を確認してください。次のイメージは、各ランからトラックされた損失と精度を示しています。各ランオブジェクトは、**Runs** 列に生成された名前と共に表示されます。

{{< img src="/images/quickstart/quickstart_image.png" alt="各ランからトラックされた損失と精度を表示しています。" >}}

## 次のステップ

W&B エコシステムのさらなる機能を探求しましょう:

1. PyTorch や Hugging Face のライブラリ、および SageMaker のようなサービスと W&B を組み合わせた [W&B インテグレーションチュートリアル]({{< relref path="guides/integrations/" lang="ja" >}}) を読んでみてください。
2. [W&B Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) を使用して、ランを整理し、自動可視化し、学びを要約し、共同作業者と更新を共有します。
3. [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を作成して、データセット、モデル、依存関係、および機械学習パイプライン全体の結果をトラックします。
4. [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) を使用してハイパーパラメーター検索を自動化し、モデルを最適化します。
5. [中央ダッシュボード]({{< relref path="/guides/models/tables/" lang="ja" >}}) でランを分析し、モデルの予測を可視化し、洞察を共有します。
6. [W&B AI Academy](https://wandb.ai/site/courses/) を訪れて、ハンズオンのコースを通じて LLMs、MLOps、W&B Models について学びましょう。