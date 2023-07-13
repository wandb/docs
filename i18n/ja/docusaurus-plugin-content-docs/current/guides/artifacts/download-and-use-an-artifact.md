---
description: Download and use Artifacts from multiple projects.
displayed_sidebar: ja
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# アーティファクトのダウンロードと使用

<head>
  <title>アーティファクトのダウンロードと使用</title>
</head>
すでにWeights & Biasesサーバーに保存されているアーティファクトをダウンロードして使用するか、アーティファクトオブジェクトを構築して必要に応じて重複排除を行ってください。

:::note
閲覧専用席のチームメンバーは、アーティファクトをダウンロードできません。
:::
### Weights & Biasesに保存されたアーティファクトのダウンロードと使用

Weights & BiasesのW&B Runの内部または外部に保存されたアーティファクトをダウンロードして使用します。Public API ([`wandb.Api`](https://docs.wandb.ai/ref/python/public-api/api)) を使って、Weights & Biasesに既に保存されているデータをエクスポート（または更新）します。詳細については、Weights & Biases [Public API Reference ガイド](https://docs.wandb.ai/ref/python/public-api) を参照してください。


<Tabs
  defaultValue="insiderun"
  values={[
    {label: 'ラン時', value: 'insiderun'},
    {label: 'Outside of a run', value: 'outsiderun'},
    {label: 'wandb CLI', value: 'cli'},
  ]}>
  <TabItem value="insiderun">

まず、W&B Python SDKをインポートしてください。次に、W&B [Run](https://docs.wandb.ai/ref/python/run) を作成します:

```python
import wandb

run = wandb.init(project="<例>", job_type="<ジョブタイプ>")
```

以下のように、[`use_artifact`](https://docs.wandb.ai/ref/python/run#use_artifact)メソッドを使用して、使用するアーティファクトを指定します。これはrunオブジェクトを返します。次のコードスニペットでは、エイリアスが`'latest'`の`'bike-dataset'`というアーティファクトを指定しています：

```python
artifact = run.use_artifact('bike-dataset:latest')
```

返されたオブジェクトを使用して、アーティファクトの内容をすべてダウンロードします：

```python
datadir = artifact.download()
```

必要に応じて、rootパラメータにパスを渡して、アーティファクトの内容を特定のディレクトリにダウンロードできます。詳細については、[Python SDKリファレンスガイド](https://docs.wandb.ai/ref/python/artifact#download)を参照してください。

ファイルのサブセットのみをダウンロードするには、[`get_path`](https://docs.wandb.ai/ref/python/artifact#get\_path) メソッドを使用してください:


```python
path = artifact.get_path(name)
```

これは、パス`name`にあるファイルだけを取得します。`Entry`オブジェクトを返し、次のメソッドがあります：

* `Entry.download`：アーティファクトからパス`name`のファイルをダウンロードする
* `Entry.ref`：エントリが`add_reference`を使って参照として保存された場合、URIを返す

Weights & Biasesが扱い方を知っているスキームを持つ参照は、アーティファクトファイルと同様にダウンロードできます。詳細については、[外部ファイルのトラッキング](https://docs.wandb.ai/guides/artifacts/track-external-files)を参照してください。

  </TabItem>
  <TabItem value="outsiderun">
  
  まず、Weights & Biases SDKをインポートします。次に、Public APIクラスからアーティファクトを作成します。そのアーティファクトに関連するエンティティ、プロジェクト、アーティファクト、エイリアスを提供してください：

  ```python
  import wandb

  api = wandb.Api()
  artifact = api.artifact('entity/project/artifact:alias')
  ```

  アーティファクトの内容をダウンロードするために、返されたオブジェクトを使用します：

  ```python
  artifact.download()
  ```

  必要に応じて、`root`パラメータにパスを渡して、アーティファクトの内容を特定のディレクトリにダウンロードできます。詳細については、[APIリファレンスガイド](https://docs.wandb.ai/ref/python/public-api/artifact#download)を参照してください。

  </TabItem>
  <TabItem value="cli">

`wandb artifact get`コマンドを使用して、Weights & Biasesサーバーからアーティファクトをダウンロードします。

```
$ wandb artifact get project/artifact:alias --root mnist/
```

  </TabItem>
</Tabs>

### 別のプロジェクトからのアーティファクトを使用する

アーティファクトを参照するには、そのアーティファクトの名前とプロジェクト名を指定してください。また、エンティティ名を指定して、アーティファクトを横断的に参照することもできます。

以下のコード例は、別のプロジェクトからアーティファクトを取得し、現在のW&B runへの入力として使用する方法を示しています。


```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
# W&Bから別のプロジェクトのアーティファクトを取得し、
# このrunの入力としてマークする。
artifact = run.use_artifact('my-project/artifact:エイリアス')
# 他のエンティティのアーティファクトを使用し、それをこのrunの入力としてマークする
artifact = run.use_artifact('my-entity/my-project/artifact:エイリアス')
```

### アーティファクトの同時構築と使用
同時にアーティファクトを構築し、使用します。アーティファクトオブジェクトを作成し、use\_artifactに渡します。これにより、Weights & Biasesでアーティファクトがまだ存在していない場合、作成されます。[`use_artifact`](https://docs.wandb.ai/ref/python/run#use\_artifact) APIは冪等性がありますので、何度でも呼び出すことができます。

```python
import wandb
artifact = wandb.Artifact('reference model')
artifact.add_file('model.h5')
run.use_artifact(artifact)
```
アーティファクトの構築に関する詳細は、[アーティファクトの構築](https://docs.wandb.ai/guides/artifacts/construct-an-artifact)を参照してください。