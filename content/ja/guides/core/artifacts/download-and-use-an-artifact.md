---
title: Download and use artifacts
description: 複数のプロジェクトからArtifactsをダウンロードして使用します。
menu:
  default:
    identifier: ja-guides-core-artifacts-download-and-use-an-artifact
    parent: artifacts
weight: 3
---

既に W&B サーバーに保存されているアーティファクトをダウンロードして使用するか、アーティファクトオブジェクトを構築して、重複排除が必要な場合に渡します。

{{% alert %}}
閲覧のみの席のチームメンバーはアーティファクトをダウンロードできません。
{{% /alert %}}

### W&B に保存されたアーティファクトのダウンロードと使用

W&B に保存されたアーティファクトを W&B Run の内部または外部でダウンロードして使用します。 W&B に既に保存されたデータをエクスポート（または更新）するには、Public API ([`wandb.Api`]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}})) を使用します。詳細については、W&B [Public API Reference guide]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を参照してください。

{{< tabpane text=true >}}
  {{% tab header="During a run" %}}
まず、W&B Python SDK をインポートします。次に、W&B [Run]({{< relref path="/ref/python/run.md" lang="ja" >}}) を作成します：

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
```

[`use_artifact`]({{< relref path="/ref/python/run.md#use_artifact" lang="ja" >}}) メソッドを使用して使用するアーティファクトを指定します。これにより、run オブジェクトが返されます。次のコードスニペットでは、`'bike-dataset'` という名前のアーティファクトを `'latest'` というエイリアスで指定します：

```python
artifact = run.use_artifact("bike-dataset:latest")
```

返されたオブジェクトを使用して、アーティファクトのすべての内容をダウンロードします：

```python
datadir = artifact.download()
```

オプションで、特定のディレクトリーにアーティファクトの内容をダウンロードするためにルートパラメータにパスを渡すことができます。詳細については、[Python SDK リファレンスガイド]({{< relref path="/ref/python/artifact.md#download" lang="ja" >}}) を参照してください。

[`get_path`]({{< relref path="/ref/python/artifact.md#get_path" lang="ja" >}}) メソッドを使用して、ファイルのサブセットのみをダウンロードします：

```python
path = artifact.get_path(name)
```

これは、パス `name` のファイルのみを取得します。以下のメソッドを持つ `Entry` オブジェクトを返します：

* `Entry.download`: パス `name` のアーティファクトからファイルをダウンロード
* `Entry.ref`: `add_reference` がエントリをリファレンスとして保存した場合、URI を返す

W&B が処理する方法を知っているスキームを持つリファレンスは、アーティファクトファイルと同様にダウンロードされます。詳細については、[外部ファイルの追跡]({{< relref path="/guides/core/artifacts/track-external-files.md" lang="ja" >}})を参照してください。  
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
まず、W&B SDK をインポートします。次に、Public API クラスからアーティファクトを作成します。そのアーティファクトに関連付けられている entity、project、artifact、およびエイリアスを提供します：

```python
import wandb

api = wandb.Api()
artifact = api.artifact("entity/project/artifact:alias")
```

返されたオブジェクトを使用して、アーティファクトの内容をダウンロードします：

```python
artifact.download()
```

オプションで、`root` パラメータにパスを渡して、特定のディレクトリーにアーティファクトの内容をダウンロードすることができます。詳細については、[API リファレンスガイド]({{< relref path="/ref/python/artifact.md#download" lang="ja" >}}) を参照してください。  
  {{% /tab %}}
  {{% tab header="W&B CLI" %}}
`wandb artifact get` コマンドを使用して、W&B サーバーからアーティファクトをダウンロードします。

```
$ wandb artifact get project/artifact:alias --root mnist/
```  
  {{% /tab %}}
{{< /tabpane >}}

### アーティファクトの部分的なダウンロード

prefix に基づいて、アーティファクトの一部をオプションでダウンロードすることができます。`path_prefix` パラメータを使用して、単一ファイルまたはサブフォルダーの内容をダウンロードできます。

```python
artifact = run.use_artifact("bike-dataset:latest")

artifact.download(path_prefix="bike.png") # bike.png のみをダウンロード
```

あるいは、特定のディレクトリーからファイルをダウンロードすることも可能です：

```python
artifact.download(path_prefix="images/bikes/") # images/bikes ディレクトリーのファイルをダウンロード
```

### 別のプロジェクトからアーティファクトを使用する

アーティファクトの名前とそれに関連するプロジェクト名を指定してアーティファクトを参照します。artifact の名前とそれに関連する entity 名を指定することで、異なる entity 間でアーティファクトを参照することもできます。

以下のコード例は、現在の W&B run の入力として別のプロジェクトからアーティファクトをクエリーする方法を示しています。

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
# 別のプロジェクトからアーティファクトを W&B にクエリーし、それを
# この run の入力としてマークします。
artifact = run.use_artifact("my-project/artifact:alias")

# 別の entity からアーティファクトを使用し、それを
# この run の入力としてマークします。
artifact = run.use_artifact("my-entity/my-project/artifact:alias")
```

### アーティファクトの構築と使用を同時に行う

アーティファクトを同時に構築し、使用します。アーティファクトオブジェクトを作成し、それを `use_artifact` に渡します。これにより、アーティファクトが W&B にまだ存在しない場合は作成されます。 [`use_artifact`]({{< relref path="/ref/python/run.md#use_artifact" lang="ja" >}}) APIは冪等 (べいとう) 性を持っているため、何度でも呼び出すことができます。

```python
import wandb

artifact = wandb.Artifact("reference model")
artifact.add_file("model.h5")
run.use_artifact(artifact)
```

アーティファクトの構築に関する詳細については、[アーティファクトの構築]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ja" >}}) を参照してください。