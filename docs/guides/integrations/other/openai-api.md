---
description: OpenAI API で W&B を使用する方法
slug: /guides/integrations/openai-api
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# OpenAI API

Weights & BiasesのOpenAI APIインテグレーションを使用すると、すべてのOpenAIモデル（ファインチューンされたモデルを含む）に対して、リクエスト、レスポンス、トークン数、モデルメタデータを1行のコードでログすることができます。

:::info
W&Bの自動ログインテグレーションは、`openai <= 0.28.1`で動作します。適切なバージョンの`openai`をインストールするには、`pip install openai==0.28.1`を使用してください。
:::

[**Colabノートブックでお試しください →**](https://github.com/wandb/examples/blob/master/colabs/openai/OpenAI_API_Autologger_Quickstart.ipynb)

たった1行のコードで、OpenAI Python SDKからWeights & Biasesへの入力と出力を自動でログできます！

![](/images/integrations/open_ai_autolog.png)

APIの入力と出力をログし始めると、さまざまなプロンプトの性能を迅速に評価したり、異なるモデル設定（例えば温度）を比較したり、トークン使用量などのその他の使用メトリクスを追跡することができます。

始めるには、`wandb`ライブラリをpipでインストールし、以下の手順に従ってください:

### 1. autologをインポートして初期化する
まず、`wandb.integration.openai`から`autolog`をインポートし、初期化します。

```python
import os
import openai
from wandb.integration.openai import autolog

autolog({"project": "gpt5"})
```

`wandb.init()`が受け入れる引数を持つ辞書を`autolog`に渡すこともできます。これには、プロジェクト名、チーム名、エンティティなどが含まれます。[`wandb.init`](../../../ref/python/init.md)の詳細については、APIリファレンスガイドを参照してください。

### 2. OpenAI APIを呼び出す
OpenAI APIへのすべての呼び出しが、Weights & Biasesに自動的にログされるようになります。

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

### 3. OpenAI APIの入力とレスポンスを見る

**ステップ1**で`autolog`によって生成されたWeights & Biasesの[run](../../runs/intro.md)リンクをクリックします。これにより、W&Bアプリのプロジェクトワークスペースにリダイレクトされます。

作成したrunを選択して、トレーステーブル、トレースタイムライン、および使用されたOpenAI LLMのモデルアーキテクチャを表示します。

### 4. autologを無効にする
OpenAI APIの使用を終了したら、すべてのW&Bプロセスを閉じるために`disable()`を呼び出すことをお勧めします。

```python
autolog.disable()
```

これで、入力と補完がWeights & Biasesにログされ、分析や同僚と共有する準備が整います。