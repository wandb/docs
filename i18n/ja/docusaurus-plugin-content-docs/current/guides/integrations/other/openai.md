---
slug: /guides/integrations/openai
description: How to integrate W&B with OpenAI.
displayed_sidebar: ja
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenAI API

:::info
**Beta Integration**: これは新しい機能であり、この機能を改善するために積極的に取り組んでいます。何かフィードバックがあれば、お気軽にお問い合わせください — contact@wandb.com
:::

OpenAIのAPIは、機械学習開発者がGPT-4にアクセスできるようにします。これは非常にパワフルな自然言語モデルであり、自然言語を理解したり生成したりするタスクにほぼ適用できます。

## OpenAI APIコールを1行のコードでログに記録する
たった1行のコードで、OpenAI Python SDKからWeights & Biasesに入力と出力を自動的にログに記録できるようになりました！

![](/images/integrations/open_ai_autolog.png)

始めるには、`wandb`ライブラリをpipでインストールし、以下の手順に従ってください。

### 1. autologをインポートして初期化する
まず、`wandb.integration.openai`から`autolog`をインポートし、初期化します。

```python
import os
import openai
from wandb.integration.openai import autolog

autolog({"project":"gpt5"})
```

必要に応じて、`autolog`に`wandb.init()`が受け付ける引数を含むディクショナリを渡すことができます。これには、プロジェクト名、チーム名、エンティティなどが含まれます。 [`wandb.init`](../../../ref/python/init.md)についての詳細は、APIリファレンスガイドを参照してください。

### 2. OpenAI APIを呼び出す
これで、OpenAI APIへの各呼び出しがWeights＆Biasesに自動的に記録されます。

```python
os.environ["OPENAI_API_KEY"] = "XXX"

chat_request_kwargs = dict(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "あなたは親切なアシスタントです。"},
        {"role": "user", "content": "2020年のワールドシリーズで誰が勝ちましたか？"},
        {"role": "assistant", "content": "ロサンゼルス・ドジャースです"},
        {"role": "user", "content": "それはどこで行われましたか？"},
    ],
)
response = openai.ChatCompletion.create(**chat_request_kwargs)
```

### 3. OpenAI APIの入力とレスポンスを表示する

**手順1**で`autolog`によって生成されたWeights＆Biases [run](../../runs/intro.md)リンクをクリックします。これにより、W&Bアプリのプロジェクトワークスペースにリダイレクトされます。

作成したrunを選択して、トレーステーブル、トレースタイムライン、および使用されたOpenAI LLMのアーキテクチャーを表示します。
### 4. オートログを無効にする
OpenAI APIを使用し終わったら、`disable()`を呼び出して、W&Bのすべてのプロセスを閉じることをお勧めします。

```python
autolog.disable()
```

これで、入力と完成がWeights & Biasesにログされ、分析や同僚との共有の準備が整います。

## W&BにOpenAIの微調整をログする

OpenAIのAPIを使って[GPT-3を微調整する](https://beta.openai.com/docs/guides/fine-tuning)場合、W&Bの統合を利用して、実験、モデル、データセットを一元管理できるダッシュボードでトラッキングできるようになります。

![](/images/integrations/open_ai_api.png)

必要なのは、`openai wandb sync`という一行だけです。

## :sparkles: インタラクティブな例をチェックしよう

* [デモColab](http://wandb.me/openai-colab)
* [レポート - GPT-3の探索と微調整のヒント](http://wandb.me/openai-report)

## :tada: 1行で微調整を同期しよう！

openaiとwandbの最新バージョンを使用していることを確認してください。

```shell-session
$ pip install --upgrade openai wandb
```

次に、コマンドラインまたはスクリプトから結果を同期します。

<Tabs
  defaultValue="cli"
  values={[
    {label: 'コマンドライン', value: 'cli'},
    {label: 'Python', value: 'python_sdk'},
  ]}>
  <TabItem value="cli">

```shell-session
$ # 1行のコマンド
$ openai wandb sync

$ # オプションパラメータを渡す
$ openai wandb sync --help
```
  </TabItem>
  <TabItem value="python_sdk">

```python
from openai.wandb_logger import WandbLogger

# 1行のコマンド
WandbLogger.sync()

# オプションパラメータを渡す
WandbLogger.sync(
    id=None,
    n_fine_tunes=None,
    project="GPT-3",
    entity=None,
    force=False,
    **kwargs_wandb_init
)
```
  </TabItem>
</Tabs>
新しい完了した微調整をスキャンし、自動的にダッシュボードに追加します。

![](/images/integrations/open_ai_auto_scan.png)

さらに、トレーニングと検証のファイルがログ化されバージョン管理され、微調整の結果の詳細も記録されます。これにより、トレーニングデータと検証データを対話式に調べることができます。

![](/images/integrations/open_ai_validation_files.png)

## :gear: 任意の引数

| 引数                     | 説明                                                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| -i ID, --id ID           | 微調整のID（任意）                                                                                                       |
| -n N, --n\_fine\_tunes N | IDが提供されていない場合にログする最も新しい微調整の数。デフォルトでは、すべての微調整が同期されます。                 |
| --project PROJECT        | プロジェクトの名前。デフォルトでは、「GPT-3」です。                                                                        |
| --entity ENTITY          | runsを送信するユーザー名またはチーム名。デフォルトでは、デフォルトのエンティティ（通常はユーザー名）が使用されます。   |
| --force                  | ログ記録を強制し、同じ微調整の既存のwandb runを上書きします。                                                          |
| \*\*kwargs\_wandb\_init  | Pythonでは、追加の引数は[`wandb.init()`](../../../ref/python/init.md)に直接渡されます。                          |

## 🔍 サンプル予測の検証

[Tables](../../tables/intro.md)を使用して、サンプル予測をより良く可視化し、モデルを比較します。

![](/images/integrations/open_ai_inspect_sample.png)

新しいrunを作成します：

```python
run = wandb.init(project="GPT-3", job_type="eval")
```
推論用のモデルIDを取得します。

自動的にログされたアーティファクトを使用して、最新のモデルを取得できます。

```python
artifact_job = run.use_artifact("ENTITY/PROJECT/fine_tune_details:latest")
fine_tuned_model = artifact_job.metadata["fine_tuned_model"]
```

検証ファイルも取得できます。

```python
artifact_valid = run.use_artifact("ENTITY/PROJECT/FILENAME:latest")
valid_file = artifact_valid.get_path("FILENAME").download()
```

OpenAI APIを使っていくつかの推論を行います。

```python
# 推論を行い結果を記録する
my_prompts = ["PROMPT_1", "PROMPT_2"]
results = []
for prompt in my_prompts:
    res = openai.Completion.create(model=fine_tuned_model,
                                   prompt=prompt,
                                   ...)
    results.append(res["choices"][0]["text"])
```

結果をテーブルでログします。
```python
table = wandb.Table(columns=['prompt', 'completion'],
                    data=list(zip(my_prompts, results)))
```

## :question:よくある質問

### どのようにしてチームとrunを共有できますか？

以下のようにして、すべてのrunをチームアカウントと同期させます。

```shell-session
$ openai wandb sync --entity MY_TEAM_ACCOUNT
```

### runをどのように整理できますか？

runは自動的に整理され、ジョブタイプ、ベースモデル、学習率、トレーニングファイル名、その他のハイパーパラメータなどの設定パラメータに基づいてフィルター/並び替えができます。

また、runの名前を変更したり、ノートを追加したり、タグを作成してグループ化することができます。

満足したら、ワークスペースを保存し、レポートを作成するために、runからデータや保存されたアーティファクト（トレーニング/検証ファイル）をインポートできます。

### 微調整の詳細にどのようにアクセスできますか？

微調整の詳細はW&Bにアーティファクトとしてログされており、以下のようにアクセスできます。

```python
import wandb
```
artifact_job = wandb.run.use_artifact('USERNAME/PROJECT/job_details:VERSION')

```

ここで`VERSION`は以下のいずれかです。

* バージョン番号（例：`v2`）

* 微調整ID（例：`ft-xxxxxxxxx`）

* 自動的に追加されたエイリアス（例：`latest`）または手動で追加されたエイリアス

その後、`artifact_job.metadata` を通して微調整の詳細にアクセスできます。例えば、微調整されたモデルは `artifact_job.metadata["fine_tuned_model"]`で取得できます。

### ファインチューンが正常に同期されなかった場合は？

いつでも `openai wandb sync` を再度呼び出すことで、正常に同期されなかったランを再同期できます。

必要に応じて、`openai wandb sync --id fine_tune_id --force` を呼び出して、特定のファインチューンを強制的に再同期できます。

### W&Bでデータセットをトラッキングできますか？

はい、Artifactsを通じて、データセットの作成、分割、モデルのトレーニングおよび評価を含む、W&Bの完全な開発フローを統合できます！

これにより、モデルの完全なトレーサビリティが実現されます。

![](/images/integrations/open_ai_faq_can_track.png)

## :books: リソース

* [OpenAI Fine-tuning Documentation](https://beta.openai.com/docs/guides/fine-tuning) は非常に詳細で、多くの有益なヒントが含まれています。

* [デモColab](http://wandb.me/openai-colab)

* [レポート - GPT-3 Exploration & Fine-tuning Tips](http://wandb.me/openai-report)