---
title: W&B クイックスタート
description: W&B クイックスタート
menu:
  default:
    identifier: ja-guides-quickstart
    parent: guides
url: quickstart
weight: 2
---

あらゆる規模の 機械学習 実験 を追跡、可視化、管理するために W&B をインストールします。
{{% alert %}}
W&B Weave の情報をお探しですか？[Weave Python SDK クイックスタート](https://weave-docs.wandb.ai/quickstart) または [Weave TypeScript SDK クイックスタート](https://weave-docs.wandb.ai/reference/generated_typescript_docs/intro-notebook) をご覧ください。
{{% /alert %}}

## サインアップして API キーを作成

W&B でマシンを認証するには、ユーザー プロフィールまたは [wandb.ai/authorize](https://wandb.ai/authorize) で API キーを生成します。API キーをコピーして安全に保管してください。

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

## run を開始し、ハイパーパラメーターを追跡する

Python スクリプトまたは ノートブック で、[`wandb.init()`]({{< relref path="/ref/python/sdk/classes/run.md" lang="ja" >}}) を使って W&B の run オブジェクトを初期化します。`config` パラメータには 辞書 を使って、ハイパーパラメーター名と 値 を指定します。

```python
run = wandb.init(
    project="my-awesome-project",  # プロジェクトを指定
    config={                        # ハイパーパラメーターとメタデータを追跡
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```

[run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) は W&B の中核要素であり、[メトリクスを追跡]({{< relref path="/guides/models/track/" lang="ja" >}})、[ログを作成]({{< relref path="/guides/models/track/log/" lang="ja" >}}) するなどに使われます。

## コンポーネントを組み立てる

次のモック トレーニングスクリプトは、精度と損失のメトリクスを W&B にログします。

```python
import wandb
import random

wandb.login()

# run が記録されるプロジェクト
project = "my-awesome-project"

# ハイパーパラメーターの辞書
config = {
    'epochs' : 10,
    'lr' : 0.01
}

with wandb.init(project=project, config=config) as run:
    offset = random.random() / 5
    print(f"lr: {config['lr']}")
    
    # トレーニングの run をシミュレート
    for epoch in range(2, config['epochs']):
        acc = 1 - 2**-config['epochs'] - random.random() / config['epochs'] - offset
        loss = 2**-config['epochs'] + random.random() / config['epochs'] + offset
        print(f"epoch={config['epochs']}, accuracy={acc}, loss={loss}")
        run.log({"accuracy": acc, "loss": loss})
```

[wandb.ai/home](https://wandb.ai/home) にアクセスすると、精度や損失などの記録されたメトリクスと、各トレーニング ステップでの変化を確認できます。次の画像は各 run から追跡された損失と精度を示しています。各 run オブジェクトは自動生成された名前とともに **Runs** 列に表示されます。

{{< img src="/images/quickstart/quickstart_image.png" alt="各 run から追跡された損失と精度を示します。" >}}

## 次のステップ

W&B エコシステム のさらなる機能を探索しましょう:

1. PyTorch などの フレームワーク、Hugging Face などの ライブラリ、SageMaker などの サービス と W&B を組み合わせる方法を解説した [W&B インテグレーションのチュートリアル]({{< relref path="guides/integrations/" lang="ja" >}}) を読んでみましょう。
2. [W&B Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) を使って run を整理し、可視化を自動化し、学びを要約し、共同作業者と更新を共有しましょう。
3. [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を作成して、 機械学習 パイプライン 全体で データセット、モデル、依存関係、結果 を追跡しましょう。
4. [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) でハイパーパラメーター探索を自動化し、モデルを最適化しましょう。
5. run を分析し、モデルの 予測 を可視化し、[中央の ダッシュボード]({{< relref path="/guides/models/tables/" lang="ja" >}}) で洞察を共有しましょう。
6. [W&B AI Academy](https://wandb.ai/site/courses/) を訪れて、ハンズオンの コース を通じて LLM、MLOps、そして W&B Models について学びましょう。
7. [weave-docs.wandb.ai](https://weave-docs.wandb.ai/) を訪れて、Weave を使って LLM ベースの アプリケーション を追跡、実験、評価、デプロイ、改善する方法を学びましょう。