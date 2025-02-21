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

[Metaflow](https://docs.metaflow.org) は、MLワークフローを作成および実行するために [Netflix](https://netflixtechblog.com) によって作成されたフレームワークです。

このインテグレーションを使用すると、ユーザーは Metaflow の[ステップとフロー](https://docs.metaflow.org/metaflow/basics)にデコレータを適用して、パラメータとアーティファクトを自動的に W&B にログすることができます。

* ステップをデコレートすると、そのステップ内の特定のタイプに対するログをオンまたはオフにします。
* フローをデコレートすると、フロー内のすべてのステップに対するログをオンまたはオフにします。

## クイックスタート

### サインアップして API キーを作成

APIキーは、W&Bに対してマシンを認証します。ユーザープロフィールから API キーを生成できます。

{{% alert %}}
よりスムーズなアプローチのために、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして API キーを生成できます。表示された API キーをコピーして、パスワードマネージャーのような安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
1. **ユーザー設定**を選択し、**API キー** のセクションまでスクロールします。
1. **表示**をクリックします。表示された API キーをコピーします。API キーを隠すためには、ページを再読み込みしてください。

### `wandb`ライブラリのインストールとログイン

ローカルに`wandb`ライブラリをインストールしてログインする方法:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})をあなたの API キーに設定します。

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
{{% tab header="ステップ" value="step" %}}

ステップをデコレートすることで、そのステップ内の特定のタイプに対するログをオンまたはオフにします。

この例では、`start` 内のすべての Datasets と Models がログされます。

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

{{% tab header="フロー" value="flow" %}}

フローをデコレートすることは、すべての構成ステップをデフォルトでデコレートすることと同等です。

この場合、`WandbExampleFlow` のすべてのステップは、`@wandb_log(datasets=True, models=True)` で各ステップをデコレートするのと同様に、デフォルトで Datasets と Models をログするようになります。

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # すべての @step をデコレート 
class WandbExampleFlow(FlowSpec):
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> データセットとしてアップロード
        self.model_file = torch.load(...)  # nn.Module    -> モデルとしてアップロード
        self.next(self.transform)
```
{{% /tab %}}

{{% tab header="フローとステップ" value="flow_and_steps" %}}

フローをデコレートすることは、すべてのステップをデフォルトでデコレートすることと同等です。つまり、その後に別の `@wandb_log` でステップをデコレートした場合、それがフローレベルのデコレーションをオーバーライドします。

この例では:

* `start` と `mid` は Datasets と Models の両方をログします。
* `end` は Datasets も Models もログしません。

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # start と mid をデコレートするのと同じ
class WandbExampleFlow(FlowSpec):
  # このステップは Datasets と Models をログします
  @step
  def start(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> データセットとしてアップロード
    self.model_file = torch.load(...)  # nn.Module    -> モデルとしてアップロード
    self.next(self.mid)

  # このステップも Datasets と Models をログします
  @step
  def mid(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> データセットとしてアップロード
    self.model_file = torch.load(...)  # nn.Module    -> モデルとしてアップロード
    self.next(self.end)

  # このステップは上書きされ、DatasetsもModelsもログしません
  @wandb_log(datasets=False, models=False)
  @step
  def end(self):
    self.raw_df = pd.read_csv(...).    
    self.model_file = torch.load(...)
```
{{% /tab %}}
{{< /tabpane >}}

## データにプログラムでアクセスする

私たちがキャプチャした情報にアクセスする方法は3つあります。最初の方法は、[wandbクライアントライブラリ]({{< relref path="/ref/python/" lang="ja" >}})を使用してログされた元の Python プロセス内からです。次に、[Web アプリ UI]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})を使用する方法、または[パブリック API]({{< relref path="/ref/python/public-api/" lang="ja" >}})を使用してプログラムでアクセスする方法があります。 `パラメータ` は W&B の [`config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) に保存され、[Overviewタブ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}}) で見つけることができます。 `datasets`、`models`、および `others` は、[W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) に保存され、[Artifactsタブ]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ja" >}}) で見つけることができます。基本的な python タイプは、W&B の [`summary`]({{< relref path="/guides/models/track/log/" lang="ja" >}}) dict に保存され、Overview タブで見つけることができます。 この情報をプログラムで外部から取得する方法についての詳細は、[パブリック APIガイド]({{< relref path="/guides/models/track/public-api-guide.md" lang="ja" >}}) を参照してください。

### クイックリファレンス

| データ                                         | クライアントライブラリ               | UI                    |
| ---------------------------------------------- | ----------------------------------- | --------------------- |
| `Parameter(...)`                               | `wandb.config`                      | Overviewタブ、Config  |
| `datasets`, `models`, `others`                 | `wandb.use_artifact("{var_name}:latest")` | Artifactsタブ         |
| 基本的な Python タイプ (`dict`, `list`, `str`など) | `wandb.summary`                     | Overviewタブ、Summary |

### `wandb_log` の kwargs

| kwarg      | オプション                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `datasets` | <ul><li><code>True</code>: インスタンス変数としてデータセットをログする</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                         |
| `models`   | <ul><li><code>True</code>: インスタンス変数としてモデルをログする</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                           |
| `others`   | <ul><li><code>True</code>: 他のシリアライズ可能なものをpickleとしてログする</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                |
| `settings` | <ul><li><code>wandb.Settings(...)</code>: このステップまたはフローのために独自の<code>wandb</code>設定を指定する</li><li><code>None</code>: <code>wandb.Settings()</code> を渡したのと同等</li></ul><p>デフォルトでは、以下の場合:</p><ul><li><code>settings.run_group</code> は<code>None</code> の場合、<code>\{flow_name\}/\{run_id\}</code> に設定されます</li><li><code>settings.run_job_type</code>は<code>None</code> の場合、 <code>\{run_job_type\}/\{step_name\}</code>に設定されます</li></ul> |

## よくある質問

### 正確には何をログするのですか? インスタンス変数とローカル変数のすべてをログするのですか?

`wandb_log` は、インスタンス変数のみをログします。 ローカル変数は決してログされません。これは、不要なデータをログしないようにするのに役立ちます。

### どのデータ型がログされますか?

現在、以下のタイプをサポートしています:

| ログ設定          | タイプ                                                                                                                 |
| ----------------- | --------------------------------------------------------------------------------------------------------------------- |
| デフォルト (常にオン) | <ul><li><code>dict, list, set, str, int, float, bool</code></li></ul>                                              |
| `datasets`        | <ul><li><code>pd.DataFrame</code></li><li><code>pathlib.Path</code></li></ul>                                        |
| `models`          | <ul><li><code>nn.Module</code></li><li><code>sklearn.base.BaseEstimator</code></li></ul>                             |
| `others`          | <ul><li>何でも <a href="https://wiki.python.org/moin/UsingPickle">pickle可能</a> で JSON シリアライズ可能なもの</li></ul> |

### ログの振る舞いをどのように設定できますか?

| 変数の種類        | 振る舞い               | 例               | データ型      |
| ----------------- | --------------------- | ---------------- | ------------ |
| インスタンス      | 自動ログ              | `self.accuracy`  | `float`      |
| インスタンス      | `datasets=True`の場合はログされる | `self.df`       | `pd.DataFrame` |
| インスタンス      | `datasets=False`の場合はログされない | `self.df`       | `pd.DataFrame` |
| ローカル          | 決してログされない    | `accuracy`       | `float`      |
| ローカル          | 決してログされない    | `df`             | `pd.DataFrame` |

### アーティファクトのリネージは追跡されますか?

はい。 ステップ A の出力であり、ステップ B の入力であるアーティファクトがある場合、自動的にリネージ DAG を構築します。

この振る舞いの例については、この [notebook](https://colab.research.google.com/drive/1wZG-jYzPelk8Rs2gIM3a71uEoG46u_nG#scrollTo=DQQVaKS0TmDU) と対応する [W&B Artifacts ページ](https://wandb.ai/megatruong/metaflow_integration/artifacts/dataset/raw_df/7d14e6578d3f1cfc72fe/graph) を参照してください。