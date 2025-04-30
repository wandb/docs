---
title: OpenAI API
description: OpenAI API で W&B を使用する方法
menu:
  default:
    identifier: ja-guides-integrations-openai-api
    parent: integrations
weight: 240
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/openai/OpenAI_API_Autologger_Quickstart.ipynb" >}}

W&B OpenAI API インテグレーションを使用して、リクエスト、レスポンス、トークンカウント、モデルメタデータをすべての OpenAI Models、ファインチューニングされた Models を含めてログします。

{{% alert %}}
[OpenAI ファインチューニング インテグレーション]({{< relref path="./openai-fine-tuning.md" lang="ja" >}}) を参照して、W&B を使用してファインチューニング実験、Models、および Datasets を追跡し、同僚と結果を共有する方法を学んでください。
{{% /alert %}}

API 入出力をログに記録することで、異なるプロンプトの性能を迅速に評価し、異なるモデル設定（例えば温度）を比較し、トークン使用量などの他の使用メトリクスを追跡することができます。

{{< img src="/images/integrations/open_ai_autolog.png" alt="" >}}

## OpenAI Python API ライブラリをインストール

W&B オートログ インテグレーションは OpenAI version 0.28.1 以下で動作します。

OpenAI Python API version 0.28.1 をインストールするには、次を実行します：
```python
pip install openai==0.28.1
```

## OpenAI Python API を使用

### 1. autolog をインポートし、初期化
最初に、`wandb.integration.openai` から `autolog` をインポートし、初期化します。

```python
import os
import openai
from wandb.integration.openai import autolog

autolog({"project": "gpt5"})
```

オプションで、`wandb.init()` が受け入れる引数の辞書を `autolog` に渡すことができます。これにはプロジェクト名、チーム名、エンティティなどが含まれます。[`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) についての詳細は、API リファレンスガイドを参照してください。

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

### 3. OpenAI API 入力とレスポンスを確認

**ステップ 1** で `autolog` により生成された W&B [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) リンクをクリックしてください。これにより、W&B App のプロジェクトワークスペースにリダイレクトされます。

作成した run を選択すると、トレーステーブル、トレースタイムライン、使用した OpenAI LLM のモデルアーキテクチャーを確認することができます。

## オートログをオフにする
W&B は、OpenAI API の使用を終了した際に、`disable()` を呼び出してすべての W&B プロセスを閉じることを推奨します。

```python
autolog.disable()
```

これで入力と補完が W&B にログされ、分析や同僚との共有の準備が整います。