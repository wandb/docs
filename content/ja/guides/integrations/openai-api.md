---
title: OpenAI API
description: OpenAI API と W&B の使い方
menu:
  default:
    identifier: ja-guides-integrations-openai-api
    parent: integrations
weight: 240
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/openai/OpenAI_API_Autologger_Quickstart.ipynb" >}}

W&B の OpenAI API インテグレーションを使えば、リクエストやレスポンス、トークン数、モデルの メタデータ を、ファインチューニング 済みを含むすべての OpenAI モデルについて ログ できます。 


{{% alert %}}
W&B を使って ファインチューニング の Experiments、Models、Datasets をトラッキングし、結果 を同僚と共有する方法は [OpenAI fine-tuning integration]({{< relref path="./openai-fine-tuning.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

API の入力と出力をログしておけば、異なるプロンプトの性能をすばやく評価したり、温度などの異なる モデル 設定を比較したり、トークン使用量などの使用 メトリクス を追跡できます。




{{< img src="/images/integrations/open_ai_autolog.png" alt="OpenAI API の自動ログ" >}}


## OpenAI Python API ライブラリをインストール

W&B の autolog インテグレーションは OpenAI バージョン 0.28.1 以下で動作します。

OpenAI Python API の バージョン 0.28.1 をインストールするには、次を実行します:
```python
pip install openai==0.28.1
```

## OpenAI Python API を使う

### 1. autolog をインポートして初期化する
まず、`wandb.integration.openai` から `autolog` をインポートして初期化します。  

```python
import os
import openai
from wandb.integration.openai import autolog

autolog({"project": "gpt5"})
```

任意で、`wandb.init()` が受け付ける 引数 を含む 辞書 を `autolog` に渡せます。これには Project 名、Team 名、Entity などを指定できます。[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) についての詳細は API リファレンス ガイド を参照してください。

### 2. OpenAI API を呼び出す
OpenAI API への各呼び出しは、自動的に W&B に ログ されます。

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

ステップ 1 で `autolog` により生成された W&B の [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) リンクをクリックします。W&B App のあなたの Project Workspace にリダイレクトされます。

作成した run を選ぶと、トレース テーブル、トレース タイムライン、使用した OpenAI の LLM の モデル アーキテクチャー を確認できます。

## autolog をオフにする
OpenAI API の使用が完了したら、すべての W&B の プロセス を終了するために `disable()` を呼び出すことをおすすめします。

```python
autolog.disable()
```

これで、入力とコンプリーションが W&B に ログ され、分析 や同僚との共有の準備が整います。