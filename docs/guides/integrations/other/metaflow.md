---
description: W&B を Metaflow と統合する方法
slug: /guides/integrations/metaflow
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Metaflow

## Overview

[Metaflow](https://docs.metaflow.org)は、MLワークフローを作成および実行するために[Netflix](https://netflixtechblog.com)によって作成されたフレームワークです。

このインテグレーションにより、ユーザーはMetaflowの[ステップとフロー](https://docs.metaflow.org/metaflow/basics)にデコレータを適用して、パラメータとArtifactsをW&Bに自動的にログすることができます。

* ステップにデコレートすると、そのステップ内の特定の型のログを有効または無効にできます。
* フローにデコレートすると、フロー内のすべてのステップのログを有効または無効にできます。

## クイックスタート

### W&Bをインストールしてログインする

<Tabs
  defaultValue="notebook"
  values={[
    {label: 'Notebook', value: 'notebook'},
    {label: 'Command Line', value: 'cli'},
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
    {label: 'Step', value: 'step'},
    {label: 'Flow', value: 'flow'},
    {label: 'Flow and Steps', value: 'flow_and_steps'},
  ]}>
  <TabItem value="step">

ステップをデコレートすると、そのステップ内の特定の型のログを有効または無効にできます。

この例では、`start`内のすべてのDatasetsとModelsがログされます。

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
  </TabItem>
  <TabItem value="flow">

フローをデコレートすることは、すべての構成ステップをデフォルトでデコレートすることと同等です。

この場合、`WandbExampleFlow`内のすべてのステップはデフォルトでDatasetsとModelsをログします -- 各ステップを`@wandb_log(datasets=True, models=True)`でデコレートするのと同じです。

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # すべての@stepをデコレートする
class WandbExampleFlow(FlowSpec):
    @step
    def start(self):
        self.raw_df = pd.read_csv(...).    # pd.DataFrame -> データセットとしてアップロード
        self.model_file = torch.load(...)  # nn.Module    -> モデルとしてアップロード
        self.next(self.transform)
```
  </TabItem>
  <TabItem value="flow_and_steps">

フローをデコレートすることは、すべてのステップをデフォルトでデコレートすることと同等です。つまり、後でステップを別の`@wandb_log`でデコレートすると、フローレベルのデコレーションが上書きされます。

以下の例では：

* `start`と`mid`はDatasetsとModelsをログしますが、
* `end`はDatasetsもModelsもログしません。

```python
from wandb.integration.metaflow import wandb_log

@wandb_log(datasets=True, models=True)  # startとmidをデコレートするのと同じ
class WandbExampleFlow(FlowSpec):
  # このステップはDatasetsとModelsをログします
  @step
  def start(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> データセットとしてアップロード
    self.model_file = torch.load(...)  # nn.Module    -> モデルとしてアップロード
    self.next(self.mid)

  # このステップもDatasetsとModelsをログします
  @step
  def mid(self):
    self.raw_df = pd.read_csv(...).    # pd.DataFrame -> データセットとしてアップロード
    self.model_file = torch.load(...)  # nn.Module    -> モデルとしてアップロード
    self.next(self.end)

  # このステップは上書きされ、DatasetsやModelsはログされません
  @wandb_log(datasets=False, models=False)
  @step
  def end(self):
    self.raw_df = pd.read_csv(...).    
    self.model_file = torch.load(...)
```
  </TabItem>
</Tabs>

## データはどこにありますか？プログラムでアクセスできますか？

キャプチャした情報には、元のPythonプロセス内で[`wandb`クライアントライブラリ](../../../ref/python/README.md)を使用して、[webアプリUI](../../app/intro.md)を介して、または[公開API](../../../ref/python/public-api/README.md)を使用してプログラムでアクセスできます。`パラメータ`はW&Bの[`config`](../../track/config.md)に保存され、[Overviewタブ](../../app/pages/run-page.md#overview-tab)で見つけることができます。`datasets`、`models`、およびその他のものは[W&B Artifacts](../../artifacts/intro.md)に保存され、[Artifactsタブ](../../app/pages/run-page.md#artifacts-tab)で見つけることができます。基本的なPythonタイプはW&Bの[`summary`](../../track/log/intro.md)辞書に保存され、Overviewタブで見つけることができます。APIを使用してこの情報を外部からプログラムで取得する方法については、[公開APIガイド](../../track/public-api-guide.md)をご覧ください。

チートシートはこちらです：

| データ                                | クライアントライブラリ                  | UI                     |
| ------------------------------------- | --------------------------------------- | --------------------- |
| `Parameter(...)`                      | `wandb.config`                          | Overviewタブ, Config  |
| `datasets`, `models`, `others`        | `wandb.use_artifact("{var_name}:latest")` | Artifactsタブ         |
| 基本的なPythonタイプ (`dict`, `list`, `str` など) | `wandb.summary`                         | Overviewタブ, Summary |

### `wandb_log`のキーワード引数 (kwargs)

| kwarg       | オプション                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `datasets`  | <ul><li><code>True</code>: データセットであるインスタンス変数をログする</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                         |
| `models`    | <ul><li><code>True</code>: モデルであるインスタンス変数をログする</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                                               |
| `others`    | <ul><li><code>True</code>: その他のシリアライズ可能なものをpickleとしてログする</li><li><code>False</code></li></ul>                                                                                                                                                                                                                                                                                                                                                               |
| `settings`  | <ul><li><code>wandb.Settings(...)</code>: このステップまたはフローのために独自の<code>wandb</code>設定を指定する</li><li><code>None</code>: <code>wandb.Settings()</code>を渡すのと同等</li></ul><p>デフォルトでは次のようになります:</p><ul><li><code>settings.run_group</code>が<code>None</code>の場合、<code>{flow_name}/{run_id}</code>に設定されます</li><li><code>settings.run_job_type</code>が<code>None</code>の場合、<code>{run_job_type}/{step_name}</code>に設定されます</li></ul> |

## よくある質問

### 具体的にどのようなログを取りますか？すべてのインスタンス変数とローカル変数をログしますか？

`wandb_log`はインスタンス変数のみをログします。ローカル変数は絶対にログされません。これは不要なデータをログしないようにするために役立ちます。

### どのデータ型をログしますか？

現在サポートしている型は以下の通りです：

| ログ設定           | タイプ                                                                                                                         |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| デフォルト（常にオン） | <ul><li><code>dict, list, set, str, int, float, bool</code></li></ul>                                                       |
| `datasets`         | <ul><li><code>pd.DataFrame</code></li><li><code>pathlib.Path</code></li></ul>                                               |
| `models`           | <ul><li><code>nn.Module</code></li><li><code>sklearn.base.BaseEstimator</code></li></ul>                                    |
| `others`           | <ul><li>pickle可能でJSONシリアライズ可能なもの</li></ul>                                                                     |

### ログの振る舞いの例

| 変数の種類 | 振る舞い                   | 例              | データ型        |
| ---------- | -------------------------- | --------------- | -------------- |
| インスタンス | 自動ログ                  | `self.accuracy`  | `float`        |
| インスタンス | `datasets=True`の場合にログ | `self.df`        | `pd.DataFrame` |
| インスタンス | `datasets=False`の場合にログしない | `self.df`        | `pd.DataFrame` |
| ローカル    | ログしない                 | `accuracy`       | `float`        |
| ローカル    | ログしない                 | `df`             | `pd.DataFrame` |

### これはアーティファクトのリネージを追跡しますか？

はい！ステップAの出力でありステップBの入力であるアーティファクトがある場合、自動的にリネージDAGを構築します。

この振る舞いの例については、この[ノートブック](https://colab.research.google.com/drive/1wZG-jYzPelk8Rs2gIM3a71uEoG46u_nG#scrollTo=DQQVaKS0TmDU)とそれに対応する[W&B Artifactsページ](https://wandb.ai/megatruong/metaflow_integration/artifacts/dataset/raw_df/7d14e6578d3f1cfc72fe/graph)をご覧ください。