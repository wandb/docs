---
title: OpenAI API
description: W&B を OpenAI API と一緒に使う方法
menu:
  default:
    identifier: openai-api
    parent: integrations
weight: 240
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/openai/OpenAI_API_Autologger_Quickstart.ipynb" >}}

W&B の OpenAI API インテグレーションを使うことで、すべての OpenAI モデル（ファインチューン済みモデルも含む）のリクエスト、レスポンス、トークン数、モデルメタデータを自動でログできます。

{{% alert %}}
[OpenAI ファインチューニングインテグレーション]({{< relref "./openai-fine-tuning.md" >}}) では、W&B でファインチューニングの Experiments、Models、Datasets をトラッキングし、同僚と結果を共有する方法をご紹介しています。
{{% /alert %}}

API の入力と出力をログすることで、異なるプロンプトやモデル設定（たとえば temperature など）ごとのパフォーマンス評価がすばやくでき、トークン使用量などのメトリクスのトラッキングも簡単に行えます。

{{< img src="/images/integrations/open_ai_autolog.png" alt="OpenAI API 自動ログ機能" >}}

## OpenAI Python API ライブラリをインストール

W&B の autolog インテグレーションは OpenAI バージョン 0.28.1 までに対応しています。

OpenAI Python API バージョン 0.28.1 をインストールするには、以下を実行します:
```python
pip install openai==0.28.1
```

## OpenAI Python API を使う

### 1. autolog のインポートと初期化
まず、`wandb.integration.openai` から `autolog` をインポートし、初期化します。

```python
import os
import openai
from wandb.integration.openai import autolog

autolog({"project": "gpt5"})
```

`autolog` にはオプションで、`wandb.init()` で受け付ける引数（プロジェクト名、チーム名、entity など）を辞書形式で渡せます。 [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) の詳しい説明は API リファレンスガイドをご参照ください。

### 2. OpenAI API を呼び出す
OpenAI API へのリクエストは、すべて自動的に W&B へログされます。

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

### 3. OpenAI API の入力とレスポンスを確認する

**ステップ 1** で `autolog` によって生成された W&B の [run]({{< relref "/guides/models/track/runs/" >}}) リンクをクリックしてください。W&B アプリのプロジェクト Workspace にアクセスできます。

作成した run を選択すると、トレーステーブル、トレースタイムライン、利用した OpenAI LLM のモデルアーキテクチャーが表示されます。

## autolog を無効にする
OpenAI API の利用が終わったら、`disable()` を呼び出して W&B のすべてのプロセスを終了することをおすすめします。

```python
autolog.disable()
```

これで、入力や補完結果が W&B にログされ、分析や同僚との共有に活用できます。