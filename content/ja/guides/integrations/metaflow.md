---
title: Metaflow
description: W&B を Metaflow と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-metaflow
    parent: integrations
weight: 200
---

## 概要

[Metaflow](https://docs.metaflow.org) は、ML のワークフローを作成・実行するために [Netflix](https://netflixtechblog.com) が作ったフレームワークです。

このインテグレーションを使うと、Metaflow の [steps と flows](https://docs.metaflow.org/metaflow/basics) にデコレータを適用して、パラメータや Artifacts を W&B に自動でログできます。

* Step をデコレートすると、その step 内で特定の種類のログをオン/オフできます。
* Flow をデコレートすると、その flow 内のすべての step に対してログのオン/オフをまとめて設定できます。

## クイックスタート

### サインアップして API キーを作成

API キーは、あなたのマシンを W&B に認証します。API キーはユーザー プロファイルから作成できます。

{{% alert %}}
よりシンプルな方法として、[W&B authorization page](https://wandb.ai/authorize) に直接アクセスして API キーを生成できます。表示された API キーをコピーし、パスワード マネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザー プロファイル アイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示された API キーをコピーします。API キーを非表示にするには、ページを再読み込みします。

### `wandb` ライブラリをインストールしてログイン

ローカルに `wandb` ライブラリをインストールしてログインします。

{{% alert %}}
`wandb` のバージョンが 0.19.8 以下の場合は、`plum-dispatch` の代わりに `fastcore` の 1.8.0 以下（`fastcore<1.8.0`）をインストールしてください。
{{% /alert %}}


{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` の[環境 変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})にあなたの API キーを設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールしてログインします。



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

### Flow と Step をデコレートする

{{< tabpane text=true >}}
{{% tab header="Step" value="step" %}}

Step をデコレートすると、その step 内で特定の種類のログをオン/オフできます。

この例では、`start` 内のすべての datasets と models がログされます。

```python
from wandb.integration.metaflow import wandb_log

class WandbExampleFlow(FlowSpec):
    @wandb_log(datasets=True, models=True, settings=wandb.Settings(...))
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> dataset としてアップロード
        self.model_file = torch.load(...)  # nn.Module    -> model としてアップロード
        self.next(self.transform)
```
{{% /tab %}}

{{% tab header="Flow" value="flow" %}}

Flow をデコレートすることは、その flow を構成するすべての step をデフォルト設定でデコレートするのと同等です。

この場合、`WandbExampleFlow` のすべての step は datasets と models をログするのがデフォルトになり、各 step に `@wandb_log(datasets=True, models=True)` を付けるのと同じ挙動になります。

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # すべての @step をデコレート
class WandbExampleFlow(FlowSpec):
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> dataset としてアップロード
        self.model_file = torch.load(...)  # nn.Module    -> model としてアップロード
        self.next(self.transform)
```
{{% /tab %}}

{{% tab header="Flow and Steps" value="flow_and_steps" %}}

Flow をデコレートすると、すべての step にデフォルトが適用されます。つまり、後からある Step に別の `@wandb_log` を付ければ、flow レベルの設定を上書きできます。

この例では:

* `start` と `mid` は datasets と models を両方ログします。
* `end` は datasets も models もログしません。

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # start と mid をデコレートするのと同じ
class WandbExampleFlow(FlowSpec):
  # この step は datasets と models をログします
  @step
  def start(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> dataset としてアップロード
    self.model_file = torch.load(...)  # nn.Module    -> model としてアップロード
    self.next(self.mid)

  # この step も datasets と models をログします
  @step
  def mid(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> dataset としてアップロード
    self.model_file = torch.load(...)  # nn.Module    -> model としてアップロード
    self.next(self.end)

  # この step は上書きされ、datasets も models もログしません
  @wandb_log(datasets=False, models=False)
  @step
  def end(self):
    self.raw_df = pd.read_csv(...).    
    self.model_file = torch.load(...)
```
{{% /tab %}}
{{< /tabpane >}}

## データにプログラムでアクセスする

取得した情報には 3 つの方法でアクセスできます。ログしている元の Python プロセス内の[`wandb` クライアント ライブラリ]({{< relref path="/ref/python/" lang="ja" >}})、[Web アプリ UI]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})、あるいは[Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を使ってプログラムから取得します。`Parameter` は W&B の [`config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) に保存され、[Overview タブ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}})で確認できます。`datasets`、`models`、`others` は [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) に保存され、[Artifacts タブ]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ja" >}})で確認できます。基本的な Python 型は W&B の [`summary`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) の dict に保存され、Overview タブで確認できます。外部からこの情報をプログラムで取得する方法の詳細は、[Public API のガイド]({{< relref path="/guides/models/track/public-api-guide.md" lang="ja" >}})を参照してください。

### クイックリファレンス

| データ                                         | クライアント ライブラリ                           | UI                    |
| ---------------------------------------------- | ----------------------------------------------- | --------------------- |
| `Parameter(...)`                                | `wandb.Run.config`                              | Overview タブ、Config |
| `datasets`, `models`, `others`                  | `wandb.Run.use_artifact("{var_name}:latest")`   | Artifacts タブ        |
| 基本的な Python 型（`dict`, `list`, `str` など） | `wandb.Run.summary`                             | Overview タブ、Summary |

### `wandb_log` の kwargs

| kwarg      | オプション                                                                         |
| ---------- | ---------------------------------------------------------------------------------- |
| `datasets` | <ul><li><code>True</code>: dataset であるインスタンス 変数をログする</li><li><code>False</code></li></ul> |
| `models`   | <ul><li><code>True</code>: model であるインスタンス 変数をログする</li><li><code>False</code></li></ul>   |
| `others`   | <ul><li><code>True</code>: pickle でシリアライズ可能なその他すべてをログする</li><li><code>False</code></li></ul> |
| `settings` | <ul><li><code>wandb.Settings(...)</code>: この step または flow 用の <code>wandb</code> 設定を指定する</li><li><code>None</code>: <code>wandb.Settings()</code> を渡すのと同じ</li></ul><p>デフォルトでは、次のように動作します:</p><ul><li><code>settings.run_group</code> が <code>None</code> の場合、<code>\{flow_name\}/\{run_id\}</code> に設定されます</li><li><code>settings.run_job_type</code> が <code>None</code> の場合、<code>\{run_job_type\}/\{step_name\}</code> に設定されます</li></ul> |

## よくある質問

### 具体的に何をログしますか？インスタンス 変数とローカル 変数はすべてログされますか？

`wandb_log` はインスタンス 変数のみをログします。ローカル 変数は一切ログしません。不要なデータのログを避けるのに有用です。

### どのデータ型がログされますか？

現在、次の型をサポートしています。

| ログ設定              | 型                                                                                                                         |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| デフォルト（常にオン） | <ul><li><code>dict, list, set, str, int, float, bool</code></li></ul>                                                      |
| `datasets`            | <ul><li><code>pd.DataFrame</code></li><li><code>pathlib.Path</code></li></ul>                                               |
| `models`              | <ul><li><code>nn.Module</code></li><li><code>sklearn.base.BaseEstimator</code></li></ul>                                    |
| `others`              | <ul><li><a href="https://wiki.python.org/moin/UsingPickle">pickle</a> 可能かつ JSON シリアライズ可能なもの</li></ul>       |

### ログの挙動はどのように設定できますか？

| 変数の種類  | 振る舞い                         | 例              | データ型        |
| ----------- | -------------------------------- | --------------- | --------------- |
| インスタンス | 自動でログされる                 | `self.accuracy` | `float`         |
| インスタンス | `datasets=True` のときにログ     | `self.df`       | `pd.DataFrame`  |
| インスタンス | `datasets=False` のときは非ログ  | `self.df`       | `pd.DataFrame`  |
| ローカル     | ログされない                     | `accuracy`      | `float`         |
| ローカル     | ログされない                     | `df`            | `pd.DataFrame`  |

### Artifacts のリネージは追跡されますか？

はい。ある artifact が step A の出力で、step B の入力である場合、そのリネージ DAG を自動で構築します。

この挙動の例については、この[ notebook](https://colab.research.google.com/drive/1wZG-jYzPelk8Rs2gIM3a71uEoG46u_nG#scrollTo=DQQVaKS0TmDU) と対応する [W&B Artifacts ページ](https://wandb.ai/megatruong/metaflow_integration/artifacts/dataset/raw_df/7d14e6578d3f1cfc72fe/graph) を参照してください。