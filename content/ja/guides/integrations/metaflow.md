---
title: Metaflow
description: W&B を Metaflow と統合する方法
menu:
  default:
    identifier: ja-guides-integrations-metaflow
    parent: integrations
weight: 200
---

## 概要

[Metaflow](https://docs.metaflow.org) は、[Netflix](https://netflixtechblog.com) によって開発された ML ワークフローを作成・実行するためのフレームワークです。

このインテグレーションにより、Metaflow の [steps と flows](https://docs.metaflow.org/metaflow/basics) にデコレーターを適用し、W&B へパラメータや Artifacts の自動ログが可能になります。

* ステップをデコレートすると、そのステップ内の特定タイプごとにログ取得の ON/OFF を切り替えられます。
* フロー全体をデコレートすると、フロー内のすべてのステップでログ取得の ON/OFF を一括設定できます。

## クイックスタート

### サインアップと API キーの作成

API キーは、あなたのマシンを W&B へ認証するためのものです。API キーはユーザープロフィールから発行できます。

{{% alert %}}
よりシンプルな方法として、[W&B 認可ページ](https://wandb.ai/authorize) に直接アクセスして API キーを発行することが可能です。表示された API キーをコピーして、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
2. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
3. **Reveal** をクリックします。表示された API キーをコピーしてください。API キーを非表示にしたい場合はページをリロードしてください。

### `wandb` ライブラリのインストールとログイン

ローカルで `wandb` ライブラリをインストールし、ログインする手順です。

{{% alert %}}
`wandb` バージョン 0.19.8 以下の場合は、`plum-dispatch` の代わりに `fastcore` バージョン 1.8.0 以前（`fastcore<1.8.0`）をインストールしてください。
{{% /alert %}}


{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) に API キーを設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` ライブラリをインストールし、ログインします。



    ```shell
    pip install -Uqqq metaflow "plum-dispatch<3.0.0" wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install -Uqqq metaflow "plum-dispatch<3.0.0" wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install -Uqqq metaflow "plum-dispatch<3.0.0" wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

### フローやステップの装飾

{{< tabpane text=true >}}
{{% tab header="Step" value="step" %}}

ステップをデコレートすることで、そのステップ内の特定タイプのログの ON/OFF を切り替えられます。

この例では、`start` 内の全ての datasets と models がログされます。

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

フロー全体をデコレートすると、全てのステップにデフォルトで適用されます。

この場合、`WandbExampleFlow` の全てのステップがデフォルトで datasets および models をログします。これは各ステップに `@wandb_log(datasets=True, models=True)` を書くのと同じです。

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # 全ての @step を装飾
class WandbExampleFlow(FlowSpec):
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> データセットとしてアップロード
        self.model_file = torch.load(...)  # nn.Module    -> モデルとしてアップロード
        self.next(self.transform)
```
{{% /tab %}}

{{% tab header="Flow and Steps" value="flow_and_steps" %}}

フローを装飾すると、すべてのステップにデフォルト設定が適用されます。つまり、後から個別の Step に `@wandb_log` を付与すると、そのステップではフローレベルの設定を上書きします。

この例では：

* `start` と `mid` は datasets と models の両方をログします。
* `end` は datasets も models もログしません。

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # start と mid をデコレートしたのと同じ
class WandbExampleFlow(FlowSpec):
  # このステップは datasets と models をログします
  @step
  def start(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> データセットとしてアップロード
    self.model_file = torch.load(...)  # nn.Module    -> モデルとしてアップロード
    self.next(self.mid)

  # このステップも datasets と models をログします
  @step
  def mid(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> データセットとしてアップロード
    self.model_file = torch.load(...)  # nn.Module    -> モデルとしてアップロード
    self.next(self.end)

  # このステップでは設定を上書きし、datasets も models もログしません
  @wandb_log(datasets=False, models=False)
  @step
  def end(self):
    self.raw_df = pd.read_csv(...).    
    self.model_file = torch.load(...)
```
{{% /tab %}}
{{< /tabpane >}}

## データへのプログラム的アクセス

記録された情報には、下記の3通りの方法でアクセス可能です：  
- ログ中の Python プロセス内から [`wandb` クライアントライブラリ]({{< relref path="/ref/python/" lang="ja" >}})
- [Web アプリの UI]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})
- [Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を使ったプログラム的アクセス

`Parameter` は W&B の [`config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) に保存され、[Overviewタブ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}}) で確認できます。`datasets`, `models`, `others` は [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) に保存され、[Artifacts タブ]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ja" >}}) で確認できます。  
Pythonの基本型は W&B の [`summary`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) ディクショナリに保存され、Overview タブで確認できます。  
API を通じて外部からこの情報を取得する方法については [Public API ガイド]({{< relref path="/guides/models/track/public-api-guide.md" lang="ja" >}}) をご覧ください。

### クイックリファレンス

| データ                                         | クライアントライブラリ                            | UI                          |
| ----------------------------------------------- | -----------------------------------------         | --------------------------- |
| `Parameter(...)`                               | `wandb.Run.config`                               | Overview タブ, Config       |
| `datasets`, `models`, `others`                 | `wandb.Run.use_artifact("{var_name}:latest")`     | Artifacts タブ              |
| Python基本型 (`dict`, `list`, `str` など)      | `wandb.Run.summary`                              | Overview タブ, Summary      |

### `wandb_log` のキーワード引数

| kwarg      | オプション                                                                         |
| ---------- | ------------------------------------------------------------------------------- |
| `datasets` | <ul><li><code>True</code>: データセット型のインスタンスメンバをログ</li><li><code>False</code></li></ul>                                   |
| `models`   | <ul><li><code>True</code>: モデル型のインスタンスメンバをログ</li><li><code>False</code></li></ul>                                         |
| `others`   | <ul><li><code>True</code>: ピックルできるシリアライズ可能なその他の値もログ</li><li><code>False</code></li></ul>                          |
| `settings` | <ul><li><code>wandb.Settings(...)</code>: このステップやフロー用の <code>wandb</code> 設定を指定</li><li><code>None</code>: <code>wandb.Settings()</code> を渡すのと同じ</li></ul><p>デフォルトでは:</p><ul><li><code>settings.run_group</code> が <code>None</code> の場合は <code>\{flow_name\}/\{run_id\}</code> になる</li><li><code>settings.run_job_type</code> が <code>None</code> の場合は <code>\{run_job_type\}/\{step_name\}</code> になる</li></ul> |

## よくある質問

### 具体的にどんな情報をログしますか？全てのインスタンス変数やローカル変数がログされますか？

`wandb_log` ではインスタンス変数のみがログされます。ローカル変数は**絶対に**ログされません。これは不要なデータの記録を避けるために有用です。

### どの型がログ対象ですか？

現時点でサポートしている型はこちらです：

| ログ設定           | 型                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| デフォルト (常にオン) | <ul><li><code>dict, list, set, str, int, float, bool</code></li></ul>                            |
| `datasets`           | <ul><li><code>pd.DataFrame</code></li><li><code>pathlib.Path</code></li></ul>                    |
| `models`             | <ul><li><code>nn.Module</code></li><li><code>sklearn.base.BaseEstimator</code></li></ul>         |
| `others`             | <ul><li><a href="https://wiki.python.org/moin/UsingPickle">pickle 化</a>可能 & JSON シリアライズ可能なもの</li></ul> |

### ログの振る舞いを設定するには？

| 変数の種類 | 振る舞い                        | 例                  | データ型               |
| ------------ | ------------------------------ | ------------------ | ---------------------- |
| インスタンス   | 自動でログされる               | `self.accuracy`     | `float`                |
| インスタンス   | `datasets=True` の時に記録     | `self.df`           | `pd.DataFrame`         |
| インスタンス   | `datasets=False` の時は記録されない | `self.df`           | `pd.DataFrame`         |
| ローカル       | ログされない                   | `accuracy`          | `float`                |
| ローカル       | ログされない                   | `df`                | `pd.DataFrame`         |

### Artifact のリネージ（由来）は追跡されますか？

はい。もしある artifact が step A の出力かつ step B の入力であれば、リネージの DAG（有向非巡回グラフ）を自動で構成します。

この挙動の例については、この [notebook](https://colab.research.google.com/drive/1wZG-jYzPelk8Rs2gIM3a71uEoG46u_nG#scrollTo=DQQVaKS0TmDU) と対応する [W&B Artifacts ページ](https://wandb.ai/megatruong/metaflow_integration/artifacts/dataset/raw_df/7d14e6578d3f1cfc72fe/graph) をご参照ください。