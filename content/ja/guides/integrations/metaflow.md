---
title: Metaflow
description: W&B を Metaflow と統合する方法
menu:
  default:
    identifier: metaflow
    parent: integrations
weight: 200
---

## 概要

[Metaflow](https://docs.metaflow.org) は[Netflix](https://netflixtechblog.com) が開発した ML ワークフローの作成と実行のためのフレームワークです。

このインテグレーションにより、Metaflow の [steps and flows](https://docs.metaflow.org/metaflow/basics) にデコレーターを適用し、パラメータや Artifacts を W&B に自動でログできます。

* ステップをデコレーションすると、そのステップ内の特定のタイプのログをオン／オフできます。
* フロー全体をデコレーションすると、そのフロー内のすべてのステップのログをオン／オフできます。

## クイックスタート

### サインアップと API キーの作成

API キーは、お使いのマシンを W&B で認証するものです。API キーはユーザープロファイルから発行できます。

{{% alert %}}
もっとシンプルに設定するには、[W&B 認証ページ](https://wandb.ai/authorize) にアクセスして API キーを発行してください。表示された API キーをコピーし、パスワードマネージャなど安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロファイルアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックし、表示された API キーをコピーします。API キーを非表示にするにはページをリロードしてください。

### `wandb` ライブラリのインストールとログイン

ローカル環境に `wandb` ライブラリをインストールし、ログインします。

{{% alert %}}
`wandb` バージョン 0.19.8 以前では、`plum-dispatch` の代わりに `fastcore` バージョン 1.8.0 以下（`fastcore<1.8.0`）をインストールしてください。
{{% /alert %}}


{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref "/guides/models/track/environment-variables.md" >}}) をご自身の API キーに設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールし、ログインします。

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

### フローやステップをデコレートする

{{< tabpane text=true >}}
{{% tab header="Step" value="step" %}}

ステップをデコレーションすると、そのステップ内の特定タイプのログがオン／オフになります。

この例では、`start` 内のすべての datasets と models がログされます。

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

フローをデコレーションすると、全ての含まれるステップをデフォルトでデコレーションしたのと同じ効果があります。

この場合、`WandbExampleFlow` のすべてのステップは、デフォルトで datasets と models をログします。個別に各ステップに `@wandb_log(datasets=True, models=True)` を付与したのと同じです。

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # すべての @step にデコレーション
class WandbExampleFlow(FlowSpec):
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> データセットとしてアップロード
        self.model_file = torch.load(...)  # nn.Module    -> モデルとしてアップロード
        self.next(self.transform)
```
{{% /tab %}}

{{% tab header="Flow and Steps" value="flow_and_steps" %}}

フローをデコレーションすると、すべてのステップがデフォルトでデコレーションされた状態と同じになります。その後個別の Step に別の `@wandb_log` を付与すると、フローレベルの設定が上書きされます。

この例では:

* `start` と `mid` は datasets と models の両方をログします。
* `end` は datasets と models のどちらもログしません。

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # start と mid をデコレーションするのと同じ
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

  # このステップは上書きされ、datasets と models をログしません
  @wandb_log(datasets=False, models=False)
  @step
  def end(self):
    self.raw_df = pd.read_csv(...).    
    self.model_file = torch.load(...)
```
{{% /tab %}}
{{< /tabpane >}}

## データへプログラムからアクセスする

記録済みの情報には以下の 3 つの方法でアクセスできます：ログ中の Python プロセス内から [`wandb` クライアントライブラリ]({{< relref "/ref/python/" >}}) を使って、[Web アプリ UI]({{< relref "/guides/models/track/workspaces.md" >}}) で、または [Public API]({{< relref "/ref/python/public-api/" >}}) をプログラムから利用して取得できます。`Parameter` は W&B の [`config`]({{< relref "/guides/models/track/config.md" >}}) に保存され、 [Overviewタブ]({{< relref "/guides/models/track/runs/#overview-tab" >}}) で確認できます。`datasets`, `models`, `others` は [W&B Artifacts]({{< relref "/guides/core/artifacts/" >}}) に保存され、 [Artifactsタブ]({{< relref "/guides/models/track/runs/#artifacts-tab" >}}) で確認できます。Python の基本型は W&B の [`summary`]({{< relref "/guides/models/track/log/" >}}) 辞書に保存され、 Overviewタブ に表示されます。API を使って外部から情報取得する方法は [Public API ガイド]({{< relref "/guides/models/track/public-api-guide.md" >}}) を参照してください。

### クイックリファレンス

| データ                                  | クライアントライブラリ                            | UI                   |
| ----------------------------------------- | ------------------------------------------ | -------------------- |
| `Parameter(...)`                         | `wandb.Run.config`                         | Overviewタブ, Config |
| `datasets`, `models`, `others`           | `wandb.Run.use_artifact("{var_name}:latest")` | Artifactsタブ        |
| Python の基本型 (`dict`, `list`, `str` 等) | `wandb.Run.summary`                        | Overviewタブ, Summary|

### `wandb_log` の kwargs

| kwarg      | オプション                                                                              |
| ---------- | -------------------------------------------------------------------------------------- |
| `datasets` | <ul><li><code>True</code>: データセットであるインスタンス変数をログ</li><li><code>False</code></li></ul>                               |
| `models`   | <ul><li><code>True</code>: モデルであるインスタンス変数をログ</li><li><code>False</code></li></ul>                                    |
| `others`   | <ul><li><code>True</code>: pickle でシリアライズ可能な他の変数をなんでもログ</li><li><code>False</code></li></ul>                         |
| `settings` | <ul><li><code>wandb.Settings(...)</code>: このステップまたはフロー用の <code>wandb</code> 設定を指定</li><li><code>None</code>: <code>wandb.Settings()</code> を渡すのと同じ</li></ul><p>デフォルトでは：</p><ul><li><code>settings.run_group</code> が <code>None</code> の場合 <code>\{flow_name\}/\{run_id\}</code> に設定</li><li><code>settings.run_job_type</code> が <code>None</code> の場合 <code>\{run_job_type\}/\{step_name\}</code> に設定</li></ul> |

## よくある質問

### 実際にどんなものをログするのですか？全てのインスタンス変数やローカル変数がログされますか？

`wandb_log` でログされるのはインスタンス変数のみです。ローカル変数は**絶対に**ログされません。これにより、不要なデータが自動でログされるのを防げます。

### どのようなデータ型がログされますか？

現時点のサポート状況は以下の通りです:

| ログ設定            | データ型                                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| default （常に有効） | <ul><li><code>dict, list, set, str, int, float, bool</code></li></ul>                            |
| `datasets`          | <ul><li><code>pd.DataFrame</code></li><li><code>pathlib.Path</code></li></ul>                     |
| `models`            | <ul><li><code>nn.Module</code></li><li><code>sklearn.base.BaseEstimator</code></li></ul>          |
| `others`            | <ul><li><a href="https://wiki.python.org/moin/UsingPickle">pickle</a> 可能かつ JSON シリアライズ可能な任意のもの</li></ul> |

### ログの振る舞いをどのように設定できますか？

| 変数の種類  | 振る舞い                            | 例                | データ型         |
| ----------- | ------------------------------------ | ----------------- | --------------- |
| インスタンス | 自動でログされる                     | `self.accuracy`   | `float`         |
| インスタンス | `datasets=True` 指定時にログされる   | `self.df`         | `pd.DataFrame`  |
| インスタンス | `datasets=False` 指定時にログされない | `self.df`         | `pd.DataFrame`  |
| ローカル    | ログされない                         | `accuracy`        | `float`         |
| ローカル    | ログされない                         | `df`              | `pd.DataFrame`  |

### アーティファクトのリネージはトラッキングされていますか？

はい。あるアーティファクトがステップ A の出力で、次のステップ B の入力になっている場合には、自動的にリネージ DAG を構成します。

この振る舞いの例は、[このノートブック](https://colab.research.google.com/drive/1wZG-jYzPelk8Rs2gIM3a71uEoG46u_nG#scrollTo=DQQVaKS0TmDU) と対応する [W&B Artifacts ページ](https://wandb.ai/megatruong/metaflow_integration/artifacts/dataset/raw_df/7d14e6578d3f1cfc72fe/graph) をご覧ください。