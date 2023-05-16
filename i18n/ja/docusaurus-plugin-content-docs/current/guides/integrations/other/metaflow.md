---
slug: /guides/integrations/metaflow
description: W&BとMetaflowの連携方法
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Metaflow

## 概要

[Metaflow](https://docs.metaflow.org)は、[Netflix](https://netflixtechblog.com)によって開発されたMLワークフローを作成および実行するためのフレームワークです。

この連携により、ユーザーは Metaflow の[ステップとフロー](https://docs.metaflow.org/metaflow/basics)にデコレータを適用して、パラメータやアーティファクトをW&Bに自動的にログすることができます。

* ステップにデコレータを適用することで、そのステップ内の特定のタイプのログの有効化または無効化が可能です。
* フローにデコレータを適用することで、フロー内のすべてのステップのログの有効化または無効化が可能です。

## クイックスタート

### W&Bをインストールし、ログインする

<Tabs
  defaultValue="notebook"
  values={[
    {label: 'ノートブック', value: 'notebook'},
    {label: 'コマンドライン', value: 'cli'},
  ]}>
  <TabItem value="notebook">

```python
!pip install -Uqqq metaflow fastcore wandb

import wandb
wandb.login()
```

  </TabItem>
  <TabItem value="cli">

```
pip install -Uqqq metaflow fastcore wandb
wandb login
```
  </TabItem>
</Tabs>

### フローとステップをデコレートする

<Tabs
  defaultValue="step"
  values={[
    {label: 'ステップ', value: 'step'},
    {label: 'フロー', value: 'flow'},
    {label: 'フローとステップ', value: 'flow_and_steps'},
  ]}>
  <TabItem value="step">

ステップをデコレートすることで、そのステップ内の特定のタイプに対してログの有効化または無効化ができます。

この例では、`start`内のすべてのデータセットとモデルがログに記録されます。
```python
from wandb.integration.metaflow import wandb_log

class WandbExampleFlow(FlowSpec):
    @wandb_log(datasets=True, models=True, settings=wandb.Settings(...))
    @step
    def start(self):
        self.raw_df = pd.read_csv(...)  # pd.DataFrame -> データセットとしてアップロード
        self.model_file = torch.load(...)  # nn.Module    -> モデルとしてアップロード
        self.next(self.transform)
```
  </TabItem>
  <TabItem value="flow">

フローをデコレートすることは、デフォルトで構成要素のすべてのステップをデコレートすることと同じです。

この場合、`WandbExampleFlow`のすべてのステップでデータセットとモデルがデフォルトで記録されます。これは、各ステップに`@wandb_log(datasets=True, models=True)`を装飾するのと同じです。

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # すべての@stepをデコレート
class WandbExampleFlow(FlowSpec):
    @step
    def start(self):
        self.raw_df = pd.read_csv(...)  # pd.DataFrame -> データセットとしてアップロード
        self.model_file = torch.load(...)  # nn.Module    -> モデルとしてアップロード
        self.next(self.transform)
```
  </TabItem>
  <TabItem value="flow_and_steps">
フローをデコレートすることは、すべてのステップをデフォルトでデコレートすることと同じです。つまり、後でステップに別の `@wandb_log` をデコレートすると、フローレベルのデコレーションが上書きされます。

以下の例では：

* `start` と `mid` はデータセットとモデルをログに記録しますが、
* `end` はデータセットもモデルもログに記録しません。

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # start と mid をデコレートするのと同じ
class WandbExampleFlow(FlowSpec):
  # このステップでは、データセットとモデルがログに記録されます
  @step
  def start(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> データセットとしてアップロード
    self.model_file = torch.load(...)  # nn.Module    -> モデルとしてアップロード
    self.next(self.mid)

  # このステップでもデータセットとモデルがログに記録されます
  @step
  def mid(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> データセットとしてアップロード
    self.model_file = torch.load(...)  # nn.Module    -> モデルとしてアップロード
    self.next(self.end)

  # このステップは上書きされており、データセットもモデルもログに記録されません
  @wandb_log(datasets=False, models=False)
  @step
  def end(self):
    self.raw_df = pd.read_csv(...).    
    self.model_file = torch.load(...)
```
  </TabItem>
</Tabs>
## データはどこにありますか？プログラムでアクセスできますか？

私たちがキャプチャした情報には、以下の3つの方法でアクセスできます：ログに使用されているオリジナルのPythonプロセス内の[`wandb`クライアントライブラリ](../../../ref/python/README.md)、[WebアプリのUI](../../app/intro.md)、または[パブリックAPI](../../../ref/python/public-api/README.md)を使ってプログラムで。 `Parameter`はW＆Bの[`config`](../../track/config.md)に保存され、[概要タブ](../../app/pages/run-page.md#overview-tab)で見つけることができます。`datasets`、`models`、`others`は[W＆Bアーティファクト](../../artifacts/intro.md)に保存され、[アーティファクトタブ](../../app/pages/run-page.md#artifacts-tab)で見つけることができます。基本的なPythonの型はW＆Bの[`summary`](../../track/log/intro.md)辞書に保存され、概要タブで見つけることができます。プログラムでこの情報を取得する方法については、[パブリックAPIのガイド](../../track/public-api-guide.md)を参照してください。

以下はチートシートです：

| データ                                             | クライアントライブラリ                            | UI                  |
| -------------------------------------------------- | -------------------------------------------------  | ------------------- |
| `Parameter(...)`                                   | `wandb.config`                                    | 概要タブ, コンフィグ|
| `datasets`, `models`, `others`                     | `wandb.use_artifact("{var_name}:latest")`         | アーティファクトタブ |
| 基本的なPythonの型（`dict`、`list`、`str`など）   | `wandb.summary`                                   | 概要タブ, サマリー   |

### `wandb_log` kwargs

| kwarg      | オプション                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `datasets` | <ul><li><code>True</code>: データセットのインスタンス変数をログに記録します</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                        |
| `models`   | <ul><li><code>True</code>: モデルのインスタンス変数をログに記録します</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                              |
| `others`   | <ul><li><code>True</code>: ピクルとしてシリアライズ可能な他のものをログに記録します</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                 |
| `settings` | <ul><li><code>wandb.Settings(...)</code>: このステップやフローのために自分で定義した<code>wandb</code>設定を指定します</li><li><code>None</code>: <code>wandb.Settings()</code>を渡すことと同じです</li></ul><p>デフォルトでは、以下の場合：</p><ul><li><code>settings.run_group</code>が<code>None</code>の場合、<code>{flow_name}/{run_id}</code>に設定されます</li><li><code>settings.run_job_type</code>が<code>None</code>の場合、<code>{run_job_type}/{step_name}</code>に設定されます</li></ul> |

## よくある質問

### 具体的に何をログに記録しますか？インスタンス変数とローカル変数のすべてをログに記録しますか？

`wandb_log`はインスタンス変数のみを記録します。ローカル変数は決して記録されません。これにより、不要なデータのログ記録を回避できます。

### どのデータ型が記録されますか？

現在、以下のタイプに対応しています：
| ロギング設定         | タイプ                                                                                                                              |

| ----------------- | --------------------------------------------------------------------------------------------------------------------------------- |

| デフォルト（常時オン） | <ul><li><code>dict, list, set, str, int, float, bool</code></li></ul>                                                             |

| `datasets`        | <ul><li><code>pd.DataFrame</code></li><li><code>pathlib.Path</code></li></ul>                                                     |

| `models`          | <ul><li><code>nn.Module</code></li><li><code>sklearn.base.BaseEstimator</code></li></ul>                                          |

| `others`          | <ul><li>何か<a href="https://wiki.python.org/moin/UsingPickle">pickle可能</a>でJSONシリアライズ可能なもの</li></ul>                |



### ロギング振る舞いの例



| 変数の種類          | 振る舞い                       | 例                | データタイプ     |

| ----------------- | ----------------------------- | ---------------- | -------------  |

| インスタンス       | 自動ロギング                  | `self.accuracy`  | `float`         |

| インスタンス       | `datasets=True`の場合にロギング | `self.df`       | `pd.DataFrame`  |

| インスタンス       | `datasets=False`の場合にはロギングされない | `self.df`       | `pd.DataFrame`  |

| ローカル            | ロギングされません          | `accuracy`      | `float`         |

| ローカル            | ロギングされません          | `df`             | `pd.DataFrame`  |



### アーティファクトの履歴をトラックしますか？



はい！ステップAの出力であり、ステップBへの入力であるアーティファクトがある場合、自動的に履歴のDAGを作成します。



この振る舞いの例として、この[ノートブック](https://colab.research.google.com/drive/1wZG-jYzPelk8Rs2gIM3a71uEoG46u\_nG#scrollTo=DQQVaKS0TmDU)と対応する[W&Bアーティファクトページ](https://wandb.ai/megatruong/metaflow\_integration/artifacts/dataset/raw\_df/7d14e6578d3f1cfc72fe/graph)をご覧ください。