---
title: W&B Quickstart
description: W&B クイックスタート
menu:
  default:
    identifier: ja-guides-quickstart
    parent: guides
url: quickstart
weight: 2
---

あらゆる規模の機械学習実験を追跡、可視化、管理するために W&B をインストールします。

## サインアップして APIキーを作成する

W&B で機械を認証するには、ユーザープロフィールまたは [wandb.ai/authorize](https://wandb.ai/authorize) から APIキーを生成します。APIキーをコピーして安全に保管してください。

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

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install wandb
import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## run を開始してハイパーパラメータを追跡する

Python スクリプトまたは notebook で、[`wandb.init()`]({{< relref path="/ref/python/run.md" lang="ja" >}}) で W&B の run オブジェクトを初期化します。`config` パラメータに辞書を使用して、ハイパーパラメータの名前と値を指定します。

```python
run = wandb.init(
    project="my-awesome-project",  # プロジェクトを指定
    config={                        # ハイパーパラメータとメタデータを追跡
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```

[run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) は、W&B のコア要素として機能し、[メトリクスを追跡]({{< relref path="/guides/models/track/" lang="ja" >}})、[ログを作成]({{< relref path="/guides/models/track/log/" lang="ja" >}}) などに使用されます。

## コンポーネントを組み立てる

このモックトレーニングスクリプトは、シミュレートされた精度と損失のメトリクスを W&B に記録します。

```python
# train.py
import wandb
import random

wandb.login()

epochs = 10
lr = 0.01

run = wandb.init(
    project="my-awesome-project",    # プロジェクトを指定
    config={                         # ハイパーパラメータとメタデータを追跡
        "learning_rate": lr,
        "epochs": epochs,
    },
)

offset = random.random() / 5
print(f"lr: {lr}")

# トレーニング run をシミュレートする
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()
```

[wandb.ai/home](https://wandb.ai/home) にアクセスして、精度や損失などの記録されたメトリクスと、各トレーニングステップ中にそれらがどのように変化したかを確認します。次の図は、各 run から追跡された損失と精度を示しています。各 run オブジェクトは、生成された名前とともに [**Runs**] 欄に表示されます。

{{< img src="/images/quickstart/quickstart_image.png" alt="各 run から追跡された損失と精度を示します。" >}}

## 次のステップ

W&B エコシステムのその他の機能を探索します。

1. W&B と PyTorch のようなフレームワーク、Hugging Face のようなライブラリ、SageMaker のようなサービスを組み合わせた [W&B Integration チュートリアル]({{< relref path="guides/integrations/" lang="ja" >}}) を読みます。
2. [W&B Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) を使用して、run を整理し、可視化を自動化し、学びを要約し、コラボレーターと更新を共有します。
3. [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を作成して、機械学習パイプライン全体のデータセット、モデル、依存関係、および結果を追跡します。
4. [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) でハイパーパラメータの検索を自動化し、モデルを最適化します。
5. [中央ダッシュボード]({{< relref path="/guides/models/tables/" lang="ja" >}}) で run を分析し、モデルの予測を可視化し、インサイトを共有します。
6. [W&B AI Academy](https://wandb.ai/site/courses/) にアクセスして、実践的なコースを通じて LLM、MLOps、および W&B Models について学びます。
