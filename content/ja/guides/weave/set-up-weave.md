---
title: W&B runs で Weave を使用する
description: Weave を インストール して 設定 し、LLM ワークフロー に関する データ と メトリクス を 記録 する 方法。
menu:
  default:
    identifier: ja-guides-weave-set-up-weave
weight: 100
---

Weave を W&B の run と統合すると、LLM ワークフローの振る舞いを全体像として把握できます。W&B が experiments、メトリクス、artifacts を追跡する一方で、Weave はプロンプト、レスポンス、ツール呼び出し、レイテンシ、トークン使用量を自動的に取得して、モデルのステップごとの実行に可視性を加えます。`wandb.init()` と並べて `weave` をインポートするだけで、追加のセットアップなしにトレースの収集を開始できます。これにより、W&B のダッシュボード上で、時間の経過に伴うエージェントのパフォーマンスをデバッグおよび計測しやすくなります。

トレースの取得方法や LLM の応答評価の始め方については、[Weave のドキュメント](https://weave-docs.wandb.ai/)をご覧ください。

## Weave をインストール

Weave をインストールするには、次を実行します:

```bash
pip install wandb weave
```

## W&B と一緒に Weave を自動初期化する

Weave をインストールしたら、インポートして W&B の run を初期化します。Weave を初期化するために追加の設定は不要です。

```python
import wandb
import weave

wandb.init(project="weave-demo")

# Weave は自動初期化され、トレースを取得する準備ができました。
# いつもどおりにコードを使えば、トレースはこの W&B の run に関連付けられます。
```

## LLM ワークフローの追跡を開始する

Weave は、OpenAI、Anthropic、Gemini などの一般的な LLM ライブラリをパッチして、LLM 呼び出しを自動追跡します。つまり、普段どおりに LLM を呼び出すだけで、Weave がその呼び出しを自動で追跡します。

例えば、次のコードスニペットは OpenAI への基本的な呼び出しを行い、追加の設定なしで Weave がトレースを取得します:

```python
import wandb
import weave
from openai import OpenAI

wandb.init(project="weave-demo")
client = OpenAI()

# Weave はこの呼び出しを自動で追跡します
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)
```

また、次のように `@weave.op` でデコレートすることで、任意の Python 関数を Weave で追跡できます:

```python
import wandb
import weave

wandb.init(project="weave-demo")

@weave.op
def agent_step(**kwargs):
    ...

def internal_step(**kwargs):
    ...


# Weave はこの呼び出しを自動で追跡します
agent_step()

# Weave はこの呼び出しを追跡しません
internal_step()
```

これにより、検索、スコアリング、データ前処理などを担う関数のデータを取得でき、LLM 以外のステップがエージェントの全体的な振る舞いにどのように寄与しているかを確認できます。

## トレースを表示する

コードの実行後、`wandb.init()` は W&B のダッシュボードへの複数のリンクを返します。トレースへのリンクは次のような形式です:

```shell
weave: Logged in as Weights & Biases user: example-user.
weave: View Weave data at https://wandb.ai/wandb/your-project/weave
weave: 🍩 https://wandb.ai/wandb/your-project/r/call/0198f4f7-2869-7694-ab8d-3d602de64377
```

ブラウザでリンクを開くと、ダッシュボードでトレースを確認できます。ダッシュボードを探索してトレース中に収集された各種メトリクスやデータを確認し、チームと結果を共有できます。