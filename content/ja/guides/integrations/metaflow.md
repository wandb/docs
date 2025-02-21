---
title: Metaflow
description: Metaflow と W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-metaflow
    parent: integrations
weight: 200
---

## 概要

[Metaflow](https://docs.metaflow.org) は、ML ワークフローを作成および実行するために [Netflix](https://netflixtechblog.com) によって作成されたフレームワークです。

このインテグレーションにより、ユーザーは Metaflow の [ステップとフロー](https://docs.metaflow.org/metaflow/basics) にデコレーターを適用して、パラメータ と Artifacts を自動的に W&B に記録できます。

* ステップをデコレートすると、そのステップ内の特定の種類に対するログ記録をオフまたはオンにします。
* フローをデコレートすると、フロー内のすべてのステップに対するログ記録をオフまたはオンにします。

## クイックスタート

### サインアップして APIキー を作成する

APIキー は、お使いのマシンを W&B に対して認証します。APIキー は、ユーザー プロファイルから生成できます。

{{% alert %}}
より効率的なアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピーして、パスワード マネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上隅にあるユーザー プロファイル アイコンをクリックします。
2. [**User Settings（ユーザー 設定）**]を選択し、[**API Keys（APIキー ）**]セクションまでスクロールします。
3. [**Reveal（表示）**]をクリックします。表示された APIキー をコピーします。APIキー を非表示にするには、ページをリロードします。

### `wandb` ライブラリをインストールしてログインする

`wandb` ライブラリをローカルにインストールしてログインするには：

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を APIキー に設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` ライブラリをインストールしてログインします。

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

ステップをデコレートすると、そのステップ内の特定の種類に対するログ記録をオフまたはオンにします。

この例では、`start` 内のすべてのデータセット と Models が記録されます。

```python
from wandb.integration.metaflow import wandb_log

class WandbExampleFlow(FlowSpec):
    @wandb_log(datasets=True, models=True, settings=wandb.Settings(...))
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> upload as dataset
        self.model_file = torch.load(...)  # nn.Module    -> upload as model
        self.next(self.transform)
```
{{% /tab %}}

{{% tab header="Flow" value="flow" %}}

フローをデコレートすることは、構成するすべてのステップをデフォルトでデコレートすることと同じです。

この場合、`WandbExampleFlow` のすべてのステップは、データセット と Models をデフォルトでログ記録します。これは、各ステップを `@wandb_log(datasets=True, models=True)` でデコレートするのと同じです。

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # decorate all @step 
class WandbExampleFlow(FlowSpec):
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> upload as dataset
        self.model_file = torch.load(...)  # nn.Module    -> upload as model
        self.next(self.transform)
```
{{% /tab %}}

{{% tab header="Flow and Steps" value="flow_and_steps" %}}

フローをデコレートすることは、すべてのステップをデフォルトでデコレートすることと同じです。つまり、後で別の `@wandb_log` でステップをデコレートすると、フローレベルのデコレートがオーバーライドされます。

この例では：

* `start` と `mid` は、データセット と Models の両方を記録します。
* `end` は、データセット も Models も記録しません。

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # same as decorating start and mid
class WandbExampleFlow(FlowSpec):
  # this step will log datasets and models
  @step
  def start(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> upload as dataset
    self.model_file = torch.load(...)  # nn.Module    -> upload as model
    self.next(self.mid)

  # this step will also log datasets and models
  @step
  def mid(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> upload as dataset
    self.model_file = torch.load(...)  # nn.Module    -> upload as model
    self.next(self.end)

  # this step is overwritten and will NOT log datasets OR models
  @wandb_log(datasets=False, models=False)
  @step
  def end(self):
    self.raw_df = pd.read_csv(...).    
    self.model_file = torch.load(...)
```
{{% /tab %}}
{{< /tabpane >}}

## プログラムでデータにアクセスする

キャプチャした情報には、[`wandb` クライアント ライブラリ]({{< relref path="/ref/python/" lang="ja" >}})を使用して、ログに記録されている元の Python プロセス内から、[Web アプリ UI]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})から、または [パブリック API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を使用してプログラムで、3 つの方法でアクセスできます。`Parameter` は W&B の [`config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) に保存され、[Overview タブ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}}) にあります。`datasets`、`models`、および `others` は [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) に保存され、[Artifacts タブ]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ja" >}}) にあります。基本 Python 型は W&B の [`summary`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) 辞書に保存され、Overview タブにあります。API を使用してこの情報を外部からプログラムで取得する方法の詳細については、[パブリック API のガイド]({{< relref path="/guides/models/track/public-api-guide.md" lang="ja" >}})を参照してください。

### クイック リファレンス

| データ                                            | クライアント ライブラリ                            | UI                    |
| ----------------------------------------------- | ----------------------------------------- | --------------------- |
| `Parameter(...)`                                | `wandb.config`                            | Overview タブ、Config  |
| `datasets`、`models`、`others`                  | `wandb.use_artifact("{var_name}:latest")` | Artifacts タブ         |
| 基本 Python 型 (`dict`、`list`、`str` など) | `wandb.summary`                           | Overview タブ、Summary |

### `wandb_log` の kwargs

| kwarg      | オプション                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `datasets` | <ul><li><code>True</code>：データセット であるインスタンス変数をログに記録します</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                         |
| `models`   | <ul><li><code>True</code>：model であるインスタンス変数をログに記録します</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                           |
| `others`   | <ul><li><code>True</code>：シリアル化可能なその他すべてを pickle としてログに記録します</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                |
| `settings` | <ul><li><code>wandb.Settings(...)</code>：このステップまたはフローの独自の <code>wandb</code> 設定を指定します</li><li><code>None</code>：<code>wandb.Settings()</code> を渡すことと同じです</li></ul><p>デフォルトでは、次の場合：</p><ul><li><code>settings.run_group</code> が <code>None</code> の場合、<code>\{flow_name\}/\{run_id\}</code> に設定されます</li><li><code>settings.run_job_type</code> が <code>None</code> の場合、<code>\{run_job_type\}/\{step_name\}</code> に設定されます</li></ul> |

## よくある質問

### ログに記録する内容は正確には何ですか？すべてのインスタンス変数とローカル変数をログに記録しますか？

`wandb_log` はインスタンス変数のみをログに記録します。ローカル変数は NEVER ログに記録されません。これは、不要なデータのログ記録を回避するのに役立ちます。

### どのデータ型がログに記録されますか？

現在、次の型をサポートしています。

| ログ設定     | 型                                                                                                                        |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| デフォルト（常にオン） | <ul><li><code>dict、list、set、str、int、float、bool</code></li></ul>                                                       |
| `datasets`          | <ul><li><code>pd.DataFrame</code></li><li><code>pathlib.Path</code></li></ul>                                               |
| `models`            | <ul><li><code>nn.Module</code></li><li><code>sklearn.base.BaseEstimator</code></li></ul>                                    |
| `others`            | <ul><li><a href="https://wiki.python.org/moin/UsingPickle">pickle 対応</a>であり、JSON シリアル化可能なもの</li></ul> |

### ログ記録の振る舞い を構成するにはどうすればよいですか？

| 変数の種類 | 振る舞い                      | 例         | データ型      |
| ---------------- | ------------------------------ | --------------- | -------------- |
| インスタンス         | 自動ログ                    | `self.accuracy` | `float`        |
| インスタンス         | `datasets=True` の場合ログに記録される      | `self.df`       | `pd.DataFrame` |
| インスタンス         | `datasets=False` の場合ログに記録されない | `self.df`       | `pd.DataFrame` |
| ローカル            | ログに記録されない                   | `accuracy`      | `float`        |
| ローカル            | ログに記録されない                   | `df`            | `pd.DataFrame` |

### Artifact のリネージ は追跡されますか？

はい。Artifact がステップ A の出力であり、ステップ B の入力である場合、リネージ DAG が自動的に構築されます。

この振る舞い の例については、この[ノートブック](https://colab.research.google.com/drive/1wZG-jYzPelk8Rs2gIM3a71uEoG46u_nG#scrollTo=DQQVaKS0TmDU)とその対応する [W&B Artifacts ページ](https://wandb.ai/megatruong/metaflow_integration/artifacts/dataset/raw_df/7d14e6578d3f1cfc72fe/graph)を参照してください。
