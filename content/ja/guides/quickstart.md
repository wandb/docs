---
title: W&B クイックスタート
description: W&B クイックスタート
menu:
  default:
    identifier: quickstart_models
    parent: guides
url: quickstart
weight: 2
---

W&B をインストールして、あらゆる規模の機械学習実験をトラッキング、可視化、管理しましょう。

{{% alert %}}
W&B Weave についてお探しですか？詳しくは [Weave Python SDK クイックスタート](https://weave-docs.wandb.ai/quickstart) または [Weave TypeScript SDK クイックスタート](https://weave-docs.wandb.ai/reference/generated_typescript_docs/intro-notebook) をご覧ください。
{{% /alert %}}

## サインアップして API キーを作成

マシンを W&B で認証するには、ユーザープロフィール もしくは [wandb.ai/authorize](https://wandb.ai/authorize) から APIキー を生成してください。APIキー をコピーし、安全に保管してください。

## `wandb` ライブラリのインストールとログイン

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref "/guides/models/track/environment-variables.md" >}}) を設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` ライブラリをインストールし、ログインします。

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

## run を開始してハイパーパラメータをトラッキング

Python のスクリプトやノートブック内で、[`wandb.init()`]({{< relref "/ref/python/sdk/classes/run.md" >}}) を使って W&B の run オブジェクトを初期化します。ハイパーパラメーター名や値は `config` パラメータに辞書形式で渡せます。

```python
run = wandb.init(
    project="my-awesome-project",  # プロジェクト名を指定
    config={                        # ハイパーパラメーターやメタデータをトラッキング
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```

[run]({{< relref "/guides/models/track/runs/" >}}) は W&B の中核となる要素であり、[メトリクスのトラッキング]({{< relref "/guides/models/track/" >}})、[ログの作成]({{< relref "/guides/models/track/log/" >}}) などに利用されます。

## コンポーネントを組み合わせる

このモックのトレーニングスクリプトでは、シミュレーションした accuracy と loss のメトリクスを W&B に記録します：

```python
import wandb
import random

wandb.login()

# run を記録するプロジェクト名
project = "my-awesome-project"

# ハイパーパラメータの辞書
config = {
    'epochs' : 10,
    'lr' : 0.01
}

with wandb.init(project=project, config=config) as run:
    offset = random.random() / 5
    print(f"lr: {config['lr']}")
    
    # トレーニング run をシミュレート
    for epoch in range(2, config['epochs']):
        acc = 1 - 2**-config['epochs'] - random.random() / config['epochs'] - offset
        loss = 2**-config['epochs'] + random.random() / config['epochs'] + offset
        print(f"epoch={config['epochs']}, accuracy={acc}, loss={loss}")
        run.log({"accuracy": acc, "loss": loss})
```

[wandb.ai/home](https://wandb.ai/home) にアクセスすると、accuracy や loss などの記録済みメトリクスや、それぞれのトレーニングステップごとの変化を確認できます。次の画像は、それぞれの run からトラッキングされた loss と accuracy を示しています。各 run オブジェクトは **Runs** カラムに自動生成された名前で表示されます。

{{< img src="/images/quickstart/quickstart_image.png" alt="Shows loss and accuracy tracked from each run." >}}

## 次のステップ

W&B エコシステムのさらなる機能も試してみましょう：

1. [W&B Integration チュートリアル]({{< relref "guides/integrations/" >}}) で、PyTorch のようなフレームワーク、Hugging Face のようなライブラリ、SageMaker のようなサービスと W&B の連携方法を学ぶことができます。
2. [W&B Reports]({{< relref "/guides/core/reports/" >}}) を使って run を整理し、可視化を自動化、学びを要約、コラボレーターと最新情報を共有しましょう。
3. [W&B Artifacts]({{< relref "/guides/core/artifacts/" >}}) を作成し、データセット、モデル、依存関係、機械学習パイプライン全体の結果をトラッキングします。
4. [W&B Sweeps]({{< relref "/guides/models/sweeps/" >}}) でハイパーパラメータ探索を自動化し、モデルを最適化しましょう。
5. [central dashboard]({{< relref "/guides/models/tables/" >}}) で run を分析し、モデルの予測を可視化、インサイトを共有しましょう。
6. [W&B AI Academy](https://wandb.ai/site/courses/) でLLMやMLOps、W&B Models をハンズオンコースで学べます。
7. [weave-docs.wandb.ai](https://weave-docs.wandb.ai/) で、Weave を使った LLM アプリケーションのトラッキング、実験、評価、デプロイ、改善方法を学べます。