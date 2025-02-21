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

W&B をインストールして、数分で 機械学習 の 実験管理 を開始しましょう。

## サインアップして APIキー を作成する

APIキー は、あなたのマシンを W&B に対して認証します。APIキー は、 ユーザー プロフィールから生成できます。

{{% alert %}}
より効率的な方法として、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピーして、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上隅にある ユーザー プロフィールアイコンをクリックします。
2. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
3. **Reveal** をクリックします。表示された APIキー をコピーします。APIキー を非表示にするには、ページをリロードしてください。

## `wandb` ライブラリをインストールしてログインする

`wandb` ライブラリをローカルにインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [ 環境 変数 ]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をあなたの APIキー に設定します。

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

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## run を開始して ハイパーパラメーター を追跡する

Python スクリプトまたは ノートブック で [`wandb.init()`]({{< relref path="/ref/python/run.md" lang="ja" >}}) を使用して W&B の Run オブジェクトを初期化し、 ハイパーパラメーター の名前と 値 のキーと 値 のペアを持つ 辞書 を `config` パラメータに渡します。

```python
run = wandb.init(
    # この run が記録される プロジェクト を設定します
    project="my-awesome-project",
    # ハイパーパラメーター と run の メタデータ を追跡します
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```

[run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) は、W&B の基本的な構成要素です。これらは、[ メトリクス の追跡 ]({{< relref path="/guides/models/track/" lang="ja" >}}), [ ログ の作成 ]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) などで頻繁に使用します。

## 完全にまとめる

すべてをまとめると、 トレーニング スクリプト は次の コード例 のようになります。強調表示された コード は、W&B 固有の コード を示しています。
機械学習 の トレーニング を模倣する コード を追加したことに注意してください。

```python
# train.py
import wandb
import random  # for demo script

# highlight-next-line
wandb.login()

epochs = 10
lr = 0.01

# highlight-start
run = wandb.init(
    # この run が記録される プロジェクト を設定します
    project="my-awesome-project",
    # ハイパーパラメーター と run の メタデータ を追跡します
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
)
# highlight-end

offset = random.random() / 5
print(f"lr: {lr}")

# simulating a training run
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    # highlight-next-line
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()
```

以上です。[https://wandb.ai/home](https://wandb.ai/home) にある W&B アプリケーション に移動して、W&B で ログ に記録した メトリクス ( 精度 と 損失 ) が各 トレーニング ステップ でどのように改善されたかを確認してください。

{{< img src="/images/quickstart/quickstart_image.png" alt="上記 の スクリプト を実行するたびに追跡された損失と 精度 を示しています。" >}}

上の画像 (クリックして拡大) は、 上記 の スクリプト を実行するたびに追跡された損失と 精度 を示しています。作成された各 run オブジェクトは、**Runs** 列に表示されます。各 run 名はランダムに生成されます。

## 次は何ですか？

W&B エコシステム の残りの部分を探索してください。

1. [W&B Integrations]({{< relref path="guides/integrations/" lang="ja" >}}) をチェックして、PyTorch などの ML フレームワーク 、Hugging Face などの ML ライブラリ 、または SageMaker などの ML サービス と W&B を 統合 する方法を学んでください。
2. [W&B Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) を使用して、run を整理し、 可視化 を埋め込んで自動化し、 学び を記述し、 コラボレーター と更新を共有します。
3. [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を作成して、機械学習 パイプライン の各ステップを通じて、 データセット 、 モデル 、依存関係、および 結果 を追跡します。
4. [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) で ハイパーパラメーター の検索を自動化し、可能な モデル の スペース を探索します。
5. [ 中央 ダッシュボード ]({{< relref path="/guides/core/tables/" lang="ja" >}}) で データセット を理解し、 モデル の 予測 を 可視化 し、 インサイト を共有します。
6. W&B AI Academy に移動し、実践的な [コース](https://wandb.me/courses) で LLM、MLOps、および W&B Models について学びましょう。
