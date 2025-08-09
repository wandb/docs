---
title: アーティファクトのダウンロードと利用
description: 複数の Projects から Artifacts をダウンロードして利用することができます。
menu:
  default:
    identifier: ja-guides-core-artifacts-download-and-use-an-artifact
    parent: artifacts
weight: 3
---

すでに W&B サーバーに保存されているアーティファクトをダウンロードして利用する、または必要に応じてアーティファクトオブジェクトを構築して重複除外に渡すことができます。

{{% alert %}}
閲覧のみ権限のチームメンバーはアーティファクトをダウンロードできません。
{{% /alert %}}


### W&B に保存されているアーティファクトをダウンロード・利用する

W&B に保存されているアーティファクトは、W&B の Run 内外どちらからでもダウンロードして利用できます。Public API（[`wandb.Api`]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}})）を使って、すでに W&B に保存されているデータをエクスポート（または更新）できます。より詳しくは W&B の [Public API Reference guide]({{< relref path="/ref/python/public-api/index.md" lang="ja" >}}) を参照してください。

{{< tabpane text=true >}}
  {{% tab header="Run 中の場合" %}}
まず、W&B の Python SDK をインポートします。次に、W&B の [Run]({{< relref path="/ref/python/sdk/classes/run.md" lang="ja" >}}) を作成します。

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
```

使いたいアーティファクトを、[`use_artifact`]({{< relref path="/ref/python/sdk/classes/run.md#use_artifact" lang="ja" >}}) メソッドで指定します。このメソッドは run オブジェクトを返します。以下のコードスニペットでは、`'bike-dataset'` という名前のアーティファクトと `'latest'` というエイリアスを指定しています。

```python
artifact = run.use_artifact("bike-dataset:latest")
```

返されたオブジェクトを使って、アーティファクトの全コンテンツをダウンロードします。

```python
datadir = artifact.download()
```

root パラメータにパスを指定することで、ダウンロード先ディレクトリーを指定することもできます。詳細は [Python SDK Reference Guide]({{< relref path="/ref/python/sdk/classes/artifact.md#download" lang="ja" >}}) をご覧ください。

[`get_path`]({{< relref path="/ref/python/sdk/classes/artifact.md#get_path" lang="ja" >}}) メソッドを使うと、一部のファイルのみをダウンロードできます。

```python
path = artifact.get_path(name)
```

これは `name` で指定したパスのファイルだけを取得します。返されるのは `Entry` オブジェクトで、以下のメソッドが利用できます。

* `Entry.download`: 指定パスのファイルをアーティファクトからダウンロードします
* `Entry.ref`: もし `add_reference` で参照として格納された場合、URI を返します

W&B で扱えるスキームの参照は、アーティファクトファイル同様にダウンロード可能です。詳しくは [外部ファイルのトラッキング]({{< relref path="/guides/core/artifacts/track-external-files.md" lang="ja" >}}) を参照してください。  
  {{% /tab %}}
  {{% tab header="Run 外の場合" %}}
まず、W&B SDK をインポートします。次に Public API クラスを使ってアーティファクトを取得します。entity、project、artifact、alias を指定してください。

```python
import wandb

api = wandb.Api()
artifact = api.artifact("entity/project/artifact:alias")
```

返されたオブジェクトを使って、アーティファクトの中身をダウンロードします。

```python
artifact.download()
```

`root` パラメータにパスを渡すことで、アーティファクトを特定のディレクトリーにダウンロードすることもできます。詳細は [API Reference Guide]({{< relref path="/ref/python/sdk/classes/artifact.md#download" lang="ja" >}}) をご覧ください。  
  {{% /tab %}}
  {{% tab header="W&B CLI" %}}
`wandb artifact get` コマンドを用いて、W&B サーバーからアーティファクトをダウンロードできます。

```
$ wandb artifact get project/artifact:alias --root mnist/
```  
  {{% /tab %}}
{{< /tabpane >}}


### アーティファクトを部分的にダウンロードする

プレフィックスを指定すれば、アーティファクトの一部のみダウンロードすることも可能です。`path_prefix` パラメータを使うと、特定のファイルやサブフォルダーの内容をダウンロードできます。

```python
artifact = run.use_artifact("bike-dataset:latest")

artifact.download(path_prefix="bike.png") # bike.png のみをダウンロード
```

また、特定のディレクトリからファイルをダウンロードすることもできます。

```python
artifact.download(path_prefix="images/bikes/") # images/bikes ディレクトリ内のファイルをダウンロード
```
### 別プロジェクトのアーティファクトを利用する

利用したいアーティファクト名と共にプロジェクト名を指定して参照できます。また、entity 名と合わせて指定すれば、エンティティ間で跨ぐアーティファクトの参照も可能です。

以下のコード例は、他のプロジェクトからアーティファクトをクエリして、現在の W&B Run の入力として利用する方法を示しています。

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
# 他プロジェクトのアーティファクトを W&B から取得し、
# この run へのインプットとしてマークします。
artifact = run.use_artifact("my-project/artifact:alias")

# 他エンティティのアーティファクトも利用可能です。
# その場合、run へのインプットとしてマークされます。
artifact = run.use_artifact("my-entity/my-project/artifact:alias")
```

### アーティファクトを同時に構築・利用する

アーティファクトを構築しつつ、同時に利用することもできます。アーティファクトオブジェクトを作成し、それを use_artifact に渡してください。まだ W&B 上に存在していない場合は新規に作成されます。[`use_artifact`]({{< relref path="/ref/python/sdk/classes/run.md#use_artifact" lang="ja" >}}) API は冪等性があるため、何度呼んでも問題ありません。

```python
import wandb

artifact = wandb.Artifact("reference model")
artifact.add_file("model.h5")
run.use_artifact(artifact)
```

アーティファクトの構築方法についてさらに詳しくは、[アーティファクトの構築]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ja" >}}) をご覧ください。