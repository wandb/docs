---
title: OpenAI API
description: OpenAI API で W&B を使用する方法。
menu:
  default:
    identifier: ja-guides-integrations-openai-api
    parent: integrations
weight: 240
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/openai/OpenAI_API_Autologger_Quickstart.ipynb" >}}

W&B OpenAI API インテグレーションを使用すると、ファインチューン されたモデルを含むすべての OpenAI モデルのリクエスト、レスポンス、トークン数、モデルの メタデータ を ログ 記録できます。

{{% alert %}}
W&B を使用して ファインチューニング の 実験 、 モデル 、 データセット を追跡し、同僚と 結果 を共有する方法については、[OpenAI ファインチューニング インテグレーション]({{< relref path="./openai-fine-tuning.md" lang="ja" >}})を参照してください。
{{% /alert %}}

API の入力と出力を ログ 記録すると、さまざまなプロンプトのパフォーマンスをすばやく評価したり、さまざまなモデル 設定 (温度など) を比較したり、トークン使用量などの他の使用状況 メトリクス を追跡したりできます。

{{< img src="/images/integrations/open_ai_autolog.png" alt="" >}}

## OpenAI Python API ライブラリをインストールする

W&B autolog インテグレーションは、OpenAI バージョン 0.28.1 以前で動作します。

OpenAI Python API バージョン 0.28.1 をインストールするには、以下を実行します。
```python
pip install openai==0.28.1
```

## OpenAI Python API の使用

### 1. autolog をインポートして初期化する
まず、`wandb.integration.openai` から `autolog` をインポートし、初期化します。

```python
import os
import openai
from wandb.integration.openai import autolog

autolog({"project": "gpt5"})
```

オプションで、`wandb.init()` が受け入れる 引数 を持つ 辞書 を `autolog` に渡すことができます。これには、プロジェクト 名、チーム名、Entities などが含まれます。[`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) の詳細については、API リファレンス ガイドを参照してください。

### 2. OpenAI API を呼び出す
OpenAI API への各呼び出しは、W&B に自動的に ログ 記録されるようになりました。

```python
os.environ["OPENAI_API_KEY"] = "XXX"

chat_request_kwargs = dict(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers"},
        {"role": "user", "content": "Where was it played?"},
    ],
)
response = openai.ChatCompletion.create(**chat_request_kwargs)
```

### 3. OpenAI API の入力とレスポンスを表示する

**ステップ 1** で `autolog` によって生成された W&B [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) リンクをクリックします。これにより、W&B アプリの プロジェクト ワークスペース にリダイレクトされます。

作成した run を選択して、トレース テーブル、トレース タイムライン、および使用された OpenAI LLM のモデル アーキテクチャ を表示します。

## autolog をオフにする
OpenAI API の使用が完了したら、`disable()` を呼び出して、すべての W&B プロセス を閉じることをお勧めします。

```python
autolog.disable()
```

これで、入力と補完が W&B に ログ 記録され、 分析 したり、同僚と共有したりできるようになります。
