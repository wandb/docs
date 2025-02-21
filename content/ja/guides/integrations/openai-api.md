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

W&B OpenAI API インテグレーションを使用して、リクエスト、レスポンス、トークン数、およびすべての OpenAI モデル（ファインチューンしたモデルも含む）のモデルメタデータをログします。

{{% alert %}}
[OpenAI ファインチューニングインテグレーション]({{< relref path="./openai-fine-tuning.md" lang="ja" >}}) を参照して、W&B を使用してファインチューンの実験、モデル、およびデータセットを追跡し、結果を同僚と共有する方法を学んでください。
{{% /alert %}}

API 入力と出力をログすることで、さまざまなプロンプトのパフォーマンスを素早く評価し、異なるモデル設定（たとえば、温度）を比較し、トークン使用量などのその他の使用メトリクスを追跡できます。

{{< img src="/images/integrations/open_ai_autolog.png" alt="" >}}

## OpenAI Python API ライブラリをインストールする

W&B autolog インテグレーションは、OpenAI バージョン 0.28.1 以下で動作します。

OpenAI Python API バージョン 0.28.1 をインストールするには、次のコマンドを実行します:
```python
pip install openai==0.28.1
```

## OpenAI Python API を使用する

### 1. autolog をインポートし、初期化する
まず、`wandb.integration.openai` から `autolog` をインポートして初期化します。

```python
import os
import openai
from wandb.integration.openai import autolog

autolog({"project": "gpt5"})
```

`wandb.init()` が受け入れる引数を含む辞書を `autolog` に渡すことができます。これには、プロジェクト名、チーム名、エンティティなどが含まれます。[`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) についての詳細は、API リファレンスガイドを参照してください。

### 2. OpenAI API を呼び出す
OpenAI API への各呼び出しは、W&B に自動的にログされます。

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

### 3. OpenAI API 入力とレスポンスを表示する

**ステップ 1** で `autolog` によって生成された W&B [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) リンクをクリックします。これにより、W&B アプリのプロジェクトワークスペースにリダイレクトされます。

作成した run を選択して、トレーステーブル、トレースタイムライン、および使用した OpenAI LLM のモデルアーキテクチャを表示します。

## autolog をオフにする
W&B は、OpenAI API の使用を終了したときにすべての W&B プロセスを終了するために `disable()` を呼び出すことをお勧めします。

```python
autolog.disable()
```

これで、入力とコンプリートが W&B にログされ、分析や同僚との共有の準備が整いました。