---
description: 複数のProjectsからArtifactsをダウンロードして使用します。
displayed_sidebar: default
---


# Download and use artifacts

<head>
  <title>Download and use artifacts</title>
</head>

W&B サーバーに既に保存されているアーティファクトをダウンロードして使用するか、アーティファクトオブジェクトを作成して必要に応じて重複排除のために渡してください。

:::note
閲覧専用のシートを持っているチームメンバーはアーティファクトをダウンロードできません。
:::

### W&B に保存されているアーティファクトをダウンロードして使用する

W&B に保存されているアーティファクトを W&B Run の内外で使用してダウンロードします。Public API ([`wandb.Api`](../../ref/python/public-api/api.md)) を使って、既に W&B に保存されているデータのエクスポート (または更新) を行います。詳細については、W&B [Public API Reference guide](../../ref/python/public-api/README.md) を参照してください。

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs
  defaultValue="insiderun"
  values={[
    {label: 'During a run', value: 'insiderun'},
    {label: 'Outside of a run', value: 'outsiderun'},
    {label: 'W&B CLI', value: 'CLI'},
  ]}>
  <TabItem value="insiderun">

まず、W&B Python SDK をインポートします。次に、W&B [Run](../../ref/python/run.md) を作成します。

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
```

[`use_artifact`](../../ref/python/run.md#use_artifact) メソッドを使って使用したいアーティファクトを指定します。これにより、run オブジェクトが返されます。以下のコードスニペットでは、`latest` というエイリアスを持つ `'bike-dataset'` というアーティファクトを指定しています。

```python
artifact = run.use_artifact("bike-dataset:latest")
```

返されるオブジェクトを使って、アーティファクトのすべての内容をダウンロードします。

```python
datadir = artifact.download()
```

オプションで、root パラメータにパスを渡して、アーティファクトの内容を特定のディレクトリーにダウンロードすることもできます。詳細については、[Python SDK Reference Guide](../../ref/python/artifact.md#download) を参照してください。

[`get_path`](../../ref/python/artifact.md#get_path) メソッドを使って、一部のファイルのみをダウンロードすることもできます。

```python
path = artifact.get_path(name)
```

これは、`name` というパスにあるファイルのみを取得します。次のメソッドを持つ `Entry` オブジェクトを返します。

* `Entry.download`: パス `name` にあるアーティファクトからファイルをダウンロードする
* `Entry.ref`: `add_reference` がエントリを参照として保存した場合、その URI を返す

W&B が処理方法を知っているスキームを持つ参照は、アーティファクトファイルと同じようにダウンロードされます。詳細については、[Track external files](../../guides/artifacts/track-external-files.md) を参照してください。

  </TabItem>
  <TabItem value="outsiderun">

まず、W&B SDK をインポートします。次に、Public API クラスからアーティファクトを作成します。そのアーティファクトに関連づけられた entity、project、artifact、および alias を提供してください。

```python
import wandb

api = wandb.Api()
artifact = api.artifact("entity/project/artifact:alias")
```

返されるオブジェクトを使用して、アーティファクトの内容をダウンロードします。

```python
artifact.download()
```

オプションで、root パラメータにパスを渡して、アーティファクトの内容を特定のディレクトリーにダウンロードすることもできます。詳細については、[API Reference Guide](../../ref/python/artifact.md#download) を参照してください。
  
  </TabItem>
  <TabItem value="CLI">

`wandb artifact get` コマンドを使用して、W&B サーバーからアーティファクトをダウンロードします。

```
$ wandb artifact get project/artifact:alias --root mnist/
```
  </TabItem>
</Tabs>

### アーティファクトを部分的にダウンロードする

path_prefix パラメータを使用して、プレフィックスに基づいてアーティファクトの一部をダウンロードすることができます。単一のファイルやサブフォルダーの内容をダウンロードできます。

```python
artifact = run.use_artifact("bike-dataset:latest")

artifact.download(path_prefix="bike.png") # bike.png のみをダウンロードする
```

別の方法として、特定のディレクトリーからファイルをダウンロードすることもできます。

```python
artifact.download(path_prefix="images/bikes/") # images/bikes ディレクトリー内のファイルをダウンロードする
```

### 別のプロジェクトからアーティファクトを使用する

アーティファクトの名前とプロジェクト名を指定して参照します。エンティティ名と共にアーティファクトの名前を指定して、他のエンティティのアーティファクトを参照することもできます。

以下のコード例は、他のプロジェクトからアーティファクトをクエリし、現在の W&B run の入力として使用する方法を示しています。

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
# 他のプロジェクトからアーティファクトを W&B にクエリして、
# この run の入力としてマークします。
artifact = run.use_artifact("my-project/artifact:alias")

# 他のエンティティのアーティファクトを使って、
# この run の入力としてマークします。
artifact = run.use_artifact("my-entity/my-project/artifact:alias")
```

### 同時にアーティファクトを作成して使用する

同時にアーティファクトを作成して使用します。アーティファクトオブジェクトを作成し、それを use_artifact に渡します。これにより、W&B にアーティファクトがまだ存在しない場合には作成されます。[`use_artifact`](../../ref/python/run.md#use_artifact) API は冪等であるため、必要なだけ何度でも呼び出すことができます。

```python
import wandb

artifact = wandb.Artifact("reference model")
artifact.add_file("model.h5")
run.use_artifact(artifact)
```

アーティファクトの作成について詳しくは、[Construct an artifact](../../guides/artifacts/construct-an-artifact.md) を参照してください。