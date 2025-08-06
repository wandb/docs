---
title: Artifacts をダウンロードして使用する
description: 複数の Projects から Artifacts をダウンロードして利用することができます。
menu:
  default:
    identifier: download-and-use-an-artifact
    parent: artifacts
weight: 3
---

W&B サーバーにすでに保存されているアーティファクトをダウンロードして使用する、またはアーティファクトオブジェクトを作成して重複排除のために渡すことができます。

{{% alert %}}
閲覧のみの権限を持つチームメンバーはアーティファクトをダウンロードできません。
{{% /alert %}}


### W&B に保存されているアーティファクトのダウンロードと利用

W&B に保存されているアーティファクトは、W&B Run の内外どちらからでもダウンロードして利用できます。Public API（[`wandb.Api`]({{< relref "/ref/python/public-api/api.md" >}})）を使って、W&B にすでに保存されているデータのエクスポート（または更新）を行います。詳細については、W&B の [Public API リファレンスガイド]({{< relref "/ref/python/public-api/index.md" >}}) を参照してください。

{{< tabpane text=true >}}
  {{% tab header="Run中の場合" %}}
まず、W&B の Python SDK をインポートします。続いて、W&B の [Run]({{< relref "/ref/python/sdk/classes/run.md" >}}) を作成します。

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
```

使用したいアーティファクトを [`use_artifact`]({{< relref "/ref/python/sdk/classes/run.md#use_artifact" >}}) メソッドで指定します。これにより run オブジェクトが返ります。以下のコードスニペットでは `'bike-dataset'` というアーティファクトを `'latest'` というエイリアスとともに指定しています。

```python
artifact = run.use_artifact("bike-dataset:latest")
```

返されたオブジェクトを使って、アーティファクトの全コンテンツをダウンロードします。

```python
datadir = artifact.download()
```

`root` パラメータにパスを指定することで、アーティファクトを特定のディレクトリーにダウンロードすることも可能です。詳細は [Python SDK リファレンスガイド]({{< relref "/ref/python/sdk/classes/artifact.md#download" >}}) をご覧ください。

ファイルのサブセットのみダウンロードしたい場合は [`get_path`]({{< relref "/ref/python/sdk/classes/artifact.md#get_path" >}}) メソッドを使います。

```python
path = artifact.get_path(name)
```

これは `name` で指定したパス上のファイルだけを取得します。返されるのは次のメソッドを持つ `Entry` オブジェクトです。

* `Entry.download`: 指定したパス name のファイルをアーティファクトからダウンロードします
* `Entry.ref`: もし `add_reference` でエントリーが参照として保存されていれば、その URI を返します

W&B がハンドリングできるスキームを持つ参照は、アーティファクトファイルと同じようにダウンロードされます。詳細は [外部ファイルを追跡する]({{< relref "/guides/core/artifacts/track-external-files.md" >}}) を参照してください。  
  {{% /tab %}}
  {{% tab header="Runの外で利用する場合" %}}
まず、W&B SDK をインポートします。次に、Public API クラスからアーティファクトを作成します。その際、そのアーティファクトに関連する entity、project、artifact、alias を指定します。

```python
import wandb

api = wandb.Api()
artifact = api.artifact("entity/project/artifact:alias")
```

返されたオブジェクトを使って、アーティファクトの内容をダウンロードします。

```python
artifact.download()
```

`root` パラメータにパスを指定することで、アーティファクトの内容を特定のディレクトリーにダウンロードすることもできます。詳しくは [API リファレンスガイド]({{< relref "/ref/python/sdk/classes/artifact.md#download" >}}) をご覧ください。  
  {{% /tab %}}
  {{% tab header="W&B CLI" %}}
`wandb artifact get` コマンドを使って、W&B サーバーからアーティファクトをダウンロードします。

```
$ wandb artifact get project/artifact:alias --root mnist/
```  
  {{% /tab %}}
{{< /tabpane >}}


### アーティファクトの一部のみダウンロードする

プレフィックスを指定して、アーティファクトの一部のみをダウンロードすることができます。`path_prefix` パラメータを使うことで、単一ファイルやサブフォルダの内容を取得できます。

```python
artifact = run.use_artifact("bike-dataset:latest")

artifact.download(path_prefix="bike.png") # bike.png だけをダウンロード
```

または、特定のディレクトリ内のファイルをダウンロードする場合：

```python
artifact.download(path_prefix="images/bikes/") # images/bikes ディレクトリ内のファイルをダウンロード
```
### 別プロジェクトのアーティファクトを利用する

参照したいアーティファクト名とそのプロジェクト名を合わせて指定すると、他のプロジェクトのアーティファクトを参照できます。さらに、entity 名をあわせて記載することで、entity 間でもアーティファクトを参照可能です。

以下のコード例は、他のプロジェクトのアーティファクトを現在の W&B run 用の入力として参照する方法を示しています。

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
# 他のプロジェクトのアーティファクトを W&B から取得し、
# この run の入力としてマークします。
artifact = run.use_artifact("my-project/artifact:alias")

# 他の entity のアーティファクトも、
# この run の入力として利用できます。
artifact = run.use_artifact("my-entity/my-project/artifact:alias")
```

### アーティファクトを同時に作成して利用する

アーティファクトオブジェクトを作成し、同時に use_artifact へ渡して利用することも可能です。アーティファクトがまだ W&B に存在していなければ新たに作成されます。[`use_artifact`]({{< relref "/ref/python/sdk/classes/run.md#use_artifact" >}}) API は冪等なので、何回呼び出しても問題ありません。

```python
import wandb

artifact = wandb.Artifact("reference model")
artifact.add_file("model.h5")
run.use_artifact(artifact)
```

アーティファクトの作成方法の詳細は [アーティファクトの作成]({{< relref "/guides/core/artifacts/construct-an-artifact.md" >}}) をご覧ください。