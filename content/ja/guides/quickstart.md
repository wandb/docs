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

W&B をインストールして、あらゆる規模の機械学習実験の追跡、可視化、管理を行いましょう。

{{% alert %}}
W&B Weave の情報をお探しですか？[Weave Python SDK クイックスタート](https://weave-docs.wandb.ai/quickstart) または [Weave TypeScript SDK クイックスタート](https://weave-docs.wandb.ai/reference/generated_typescript_docs/intro-notebook) をご覧ください。
{{% /alert %}}

## サインアップと APIキー の作成

W&B でマシンを認証するには、ユーザープロフィールまたは [wandb.ai/authorize](https://wandb.ai/authorize) で APIキー を生成します。生成した APIキー をコピーして、安全な場所に保管してください。

## `wandb` ライブラリのインストールとログイン

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})を設定します。

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

## run を開始してハイパーパラメーターを記録

Python スクリプトやノートブック内で [`wandb.init()`]({{< relref path="/ref/python/sdk/classes/run.md" lang="ja" >}}) を使い、W&B の run オブジェクトを初期化します。`config` パラメータにはハイパーパラメーター名と値の辞書を指定できます。

```python
run = wandb.init(
    project="my-awesome-project",  # 使用するプロジェクトを指定
    config={                        # ハイパーパラメーターやメタデータを記録
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```

[run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) は W&B の中心的な要素で、[メトリクスの追跡]({{< relref path="/guides/models/track/" lang="ja" >}})や[ログの作成]({{< relref path="/guides/models/track/log/" lang="ja" >}}) などに使用されます。

## 基本的な構成

このサンプルのトレーニングスクリプトでは、W&B に精度（accuracy）と損失（loss）のメトリクスをシミュレーションして記録します。

```python
import wandb
import random

wandb.login()

# run を記録するプロジェクト名
project = "my-awesome-project"

# ハイパーパラメーターを記載した辞書
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

[wandb.ai/home](https://wandb.ai/home) にアクセスして、精度や損失など記録されたメトリクスや、トレーニングごとの変化をグラフで確認できます。下の画像は、各 run で記録された損失と精度の例です。各 run オブジェクトは **Runs** 列に自動生成された名前で表示されます。

{{< img src="/images/quickstart/quickstart_image.png" alt="Shows loss and accuracy tracked from each run." >}}

## 次のステップ

W&B エコシステムのさまざまな機能を試してみましょう。

1. PyTorch などのフレームワークや Hugging Face のようなライブラリ、SageMaker などのサービスと連携した [W&B Integration チュートリアル]({{< relref path="guides/integrations/" lang="ja" >}}) を読む。
2. run の整理、自動可視化、学びの要約、コラボレーターとの共有などに [W&B Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) を活用。
3. [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を作成し、データセット、モデル、依存関係、結果を機械学習パイプライン全体でトラッキング。
4. [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) を使い、ハイパーパラメーターサーチの自動化やモデルの最適化を実施。
5. run の分析やモデル予測の可視化、洞察の共有には [central dashboard]({{< relref path="/guides/models/tables/" lang="ja" >}}) を使用。
6. [W&B AI Academy](https://wandb.ai/site/courses/) で LLM、MLOps、W&B Models についての実践型コースを学びましょう。
7. [weave-docs.wandb.ai](https://weave-docs.wandb.ai/) では、Weave を使い LLM アプリケーションの実験、評価、デプロイ、改善方法を学べます。