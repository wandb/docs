---
title: Metaflow
description: W&B と Metaflow を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-metaflow
    parent: integrations
weight: 200
---

## 概要

[Metaflow](https://docs.metaflow.org) は、Netflixが開発したMLワークフローを作成し実行するためのフレームワークです。

このインテグレーションにより、ユーザーはMetaflowのステップとフローにデコレータを適用して、W&Bにパラメータとアーティファクトを自動的にログすることができます。

* ステップをデコレートすると、そのステップ内の特定のタイプに対してログのオンまたはオフが適用されます。
* フローをデコレートすると、フロー内のすべてのステップに対してログのオンまたはオフが適用されます。

## クイックスタート

### サインアップしてAPIキーを作成する

APIキーはあなたのマシンをW&Bに認証します。ユーザープロフィールからAPIキーを生成することができます。

{{% alert %}}
よりスムーズな方法として、[https://wandb.ai/authorize](https://wandb.ai/authorize)に直接アクセスしてAPIキーを生成できます。表示されたAPIキーをコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
2. **User Settings**を選択し、**API Keys**セクションまでスクロールします。
3. **Reveal**をクリックし、表示されたAPIキーをコピーします。ページをリロードするとAPIキーを隠すことができます。

### `wandb`ライブラリをインストールしてログインする

ローカルに`wandb`ライブラリをインストールし、ログインするためには次の手順を行います。

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})をAPIキーに設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb`ライブラリをインストールしてログインします。

    ```shell
    pip install -Uqqq metaflow fastcore wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install -Uqqq metaflow fastcore wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install -Uqqq metaflow fastcore wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

### フローとステップをデコレートする

{{< tabpane text=true >}}
{{% tab header="Step" value="step" %}}

ステップをデコレートすることで、そのステップ内の特定のタイプに対してログのオンまたはオフが適用されます。

この例では、`start`における全てのデータセットとモデルがログされます。

```python
from wandb.integration.metaflow import wandb_log

class WandbExampleFlow(FlowSpec):
    @wandb_log(datasets=True, models=True, settings=wandb.Settings(...))
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> データセットとしてアップロード
        self.model_file = torch.load(...)  # nn.Module    -> モデルとしてアップロード
        self.next(self.transform)
```
{{% /tab %}}

{{% tab header="Flow" value="flow" %}}

フローをデコレートすることは、すべての構成ステップをデフォルトでデコレートすることに相当します。

この場合、`WandbExampleFlow`のすべてのステップは、各ステップを `@wandb_log(datasets=True, models=True)`でデコレートするのと同様に、デフォルトでデータセットとモデルをログします。

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # すべての@stepをデコレート
class WandbExampleFlow(FlowSpec):
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> データセットとしてアップロード
        self.model_file = torch.load(...)  # nn.Module    -> モデルとしてアップロード
        self.next(self.transform)
```
{{% /tab %}}

{{% tab header="Flow and Steps" value="flow_and_steps" %}}

フローをデコレートすることは、すべてのステップをデフォルトでデコレートすることを意味します。つまり、後でステップを別の`@wandb_log`でデコレートすると、フローレベルのデコレーションが上書きされます。

この例では:

* `start`と`mid`は両方、データセットとモデルをログします。
* `end`は、データセットもモデルもログしません。

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # startとmidをデコレートするのと同じ
class WandbExampleFlow(FlowSpec):
  # このステップはデータセットとモデルをログします
  @step
  def start(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> データセットとしてアップロード
    self.model_file = torch.load(...)  # nn.Module    -> モデルとしてアップロード
    self.next(self.mid)

  # このステップもデータセットとモデルをログします
  @step
  def mid(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> データセットとしてアップロード
    self.model_file = torch.load(...)  # nn.Module    -> モデルとしてアップロード
    self.next(self.end)

  # このステップは上書きされており、データセットもモデルもログしません
  @wandb_log(datasets=False, models=False)
  @step
  def end(self):
    self.raw_df = pd.read_csv(...).    
    self.model_file = torch.load(...)
```
{{% /tab %}}
{{< /tabpane >}}

## データへプログラムでアクセスする

キャプチャされた情報には3つの方法でアクセスできます: [`wandb`クライアントライブラリ]({{< relref path="/ref/python/" lang="ja" >}})を使用してオリジナルのPythonプロセス内でログされたもの、[ウェブアプリUI]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})、あるいは[パブリックAPI]({{< relref path="/ref/python/public-api/" lang="ja" >}})をプログラムで使用する方法です。パラメータはW&Bの[`config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}})に保存され、[Overviewタブ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}})で見つけることができます。`datasets`、`models`、およびその他は[W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}})に保存され、[Artifactsタブ]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ja" >}})で見つけることができます。基本的なPythonタイプはW&Bの[`summary`]({{< relref path="/guides/models/track/log/" lang="ja" >}})ディクショナリに保存され、Overviewタブで見ることができます。これらの情報を外部からプログラムで取得する方法の詳細については、[パブリックAPIのガイド]({{< relref path="/guides/models/track/public-api-guide.md" lang="ja" >}})をご覧ください。

### クイックリファレンス

| データ                                           | クライアントライブラリ                         | UI                     |
| ----------------------------------------------- | ----------------------------------------- | --------------------- |
| `Parameter(...)`                                | `wandb.config`                            | Overviewタブ, Config  |
| `datasets`, `models`, `others`                  | `wandb.use_artifact("{var_name}:latest")` | Artifactsタブ         |
| 基本的なPython型 (`dict`, `list`, `str`, etc.)  | `wandb.summary`                           | Overviewタブ, Summary |

### `wandb_log`引数

| kwarg      | オプション                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `datasets` | <ul><li><code>True</code>: インスタンス変数がデータセットの場合にログする</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                         |
| `models`   | <ul><li><code>True</code>: インスタンス変数がモデルの場合にログする</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                           |
| `others`   | <ul><li><code>True</code>: pickleとしてシリアライズ可能なその他のものをログする</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                |
| `settings` | <ul><li><code>wandb.Settings(...)</code>: このステップまたはフローのために独自の<code>wandb</code>設定を指定する</li><li><code>None</code>: <code>wandb.Settings()</code>を渡すのと同じ</li></ul><p>デフォルトでは、もし:</p><ul><li><code>settings.run_group</code>が<code>None</code>であれば、<code>\{flow_name\}/\{run_id\}</code>に設定されます</li><li><code>settings.run_job_type</code>が<code>None</code>であれば、<code>\{run_job_type\}/\{step_name\}</code>に設定されます</li></ul> |

## よくある質問

### 正確には何をログしますか？すべてのインスタンスとローカル変数をログしますか？

`wandb_log`はインスタンス変数のみをログします。ローカル変数は決してログされません。これは不要なデータをログしないために役立ちます。

### どのようなデータ型がログされますか？

現在、以下のタイプをサポートしています：

| ログ設定           | 型                                                                                                                        |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| デフォルト(常にオン) | <ul><li><code>dict, list, set, str, int, float, bool</code></li></ul>                                                       |
| `datasets`          | <ul><li><code>pd.DataFrame</code></li><li><code>pathlib.Path</code></li></ul>                                               |
| `models`            | <ul><li><code>nn.Module</code></li><li><code>sklearn.base.BaseEstimator</code></li></ul>                                    |
| `others`            | <ul><li><a href="https://wiki.python.org/moin/UsingPickle">pickle-able</a>でありJSONシリアライズ可能なもの</li></ul>       |

### どのようにログの振る舞いを設定できますか？

| 変数の種類       | 振る舞い                     | 例               | データ型         |
| ---------------- | ------------------------------ | --------------- | -------------- |
| インスタンス       | 自動ログされる                 | `self.accuracy` | `float`        |
| インスタンス       | `datasets=True`の場合にログ   | `self.df`       | `pd.DataFrame` |
| インスタンス       | `datasets=False`の場合はログされない | `self.df`       | `pd.DataFrame` |
| ローカル         | ログされない                   | `accuracy`      | `float`        |
| ローカル         | ログされない                   | `df`            | `pd.DataFrame` |

### アーティファクトのリネージは追跡されますか？

はい。ステップAの出力であり、ステップBの入力であるアーティファクトがあれば、リネージDAGを自動的に構築します。

この振る舞いの例については、この[ノートブック](https://colab.research.google.com/drive/1wZG-jYzPelk8Rs2gIM3a71uEoG46u_nG#scrollTo=DQQVaKS0TmDU)および対応する [W&B Artifactsページ](https://wandb.ai/megatruong/metaflow_integration/artifacts/dataset/raw_df/7d14e6578d3f1cfc72fe/graph)をご覧ください。