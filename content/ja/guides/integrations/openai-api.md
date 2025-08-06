---
title: OpenAI API
description: W&B を OpenAI API と一緒に使う方法
menu:
  default:
    identifier: ja-guides-integrations-openai-api
    parent: integrations
weight: 240
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/openai/OpenAI_API_Autologger_Quickstart.ipynb" >}}

W&B の OpenAI API インテグレーションを使うことで、全ての OpenAI モデル（ファインチューン済みモデルも含む）のリクエスト、レスポンス、トークン数、モデルのメタデータをログできます。

{{% alert %}}
W&B を使ってファインチューン実験やモデル、データセットをトラッキングし、その結果を同僚と共有する方法については、[OpenAI ファインチューニング インテグレーション]({{< relref path="./openai-fine-tuning.md" lang="ja" >}})をご覧ください。
{{% /alert %}}

API 入力と出力をログすることで、プロンプトの違いによるパフォーマンスを素早く評価したり、温度などの異なるモデル設定を比較したり、トークン使用量などのその他のメトリクスを記録して追跡することができます。

{{< img src="/images/integrations/open_ai_autolog.png" alt="OpenAI API automatic logging" >}}

## OpenAI Python API ライブラリのインストール

W&B の autolog インテグレーションは OpenAI バージョン 0.28.1 以下で動作します。

OpenAI Python API バージョン 0.28.1 をインストールするには、以下を実行してください。
```python
pip install openai==0.28.1
```

## OpenAI Python API の使い方

### 1. autolog をインポートして初期化する

まず、`wandb.integration.openai` から `autolog` をインポートし、初期化します。

```python
import os
import openai
from wandb.integration.openai import autolog

autolog({"project": "gpt5"})
```

オプションで、`autolog` に対して `wandb.init()` で受け付ける引数を含む辞書を渡すことができます。これにはプロジェクト名、チーム名、entity などが含まれます。[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}})の詳細は API リファレンスガイドを参照してください。

### 2. OpenAI API を呼び出す

OpenAI API にアクセスするたびに、その情報が自動的に W&B に記録されます。

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

### 3. OpenAI API の入力およびレスポンスを見る

**ステップ 1** で `autolog` によって生成された W&B の [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) リンクをクリックしてください。これで W&B App のプロジェクト ワークスペースへ遷移します。

作成した run を選択すると、トレーステーブル、トレースタイムライン、使用された OpenAI LLM のモデルアーキテクチャーを確認できます。

## autolog をオフにする

OpenAI API の利用が終わったら、すべての W&B プロセスを終了するため、`disable()` の呼び出しを推奨します。

```python
autolog.disable()
```

このようにして、入力と生成結果が W&B にログされ、分析や同僚との共有準備が整います。