---
title: アーティファクトをダウンロードして使用する
description: 複数の Projects から Artifacts をダウンロードして使用する。
menu:
  default:
    identifier: ja-guides-core-artifacts-download-and-use-an-artifact
    parent: artifacts
weight: 3
---

すでに W&B サーバーに保存されているアーティファクトをダウンロードして使用するか、アーティファクト オブジェクトを構築して、必要に応じて重複排除のためにそれを渡します。

{{% alert %}}
閲覧専用シートのチームメンバーは、アーティファクトをダウンロードできません。
{{% /alert %}}

### W&B に保存されているアーティファクトをダウンロードして使用する

W&B に保存されているアーティファクトを W&B Run の内外でダウンロードして使用します。Public API（[`wandb.Api`]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}})）を使用して、W&B にすでに保存されているデータをエクスポート（または更新）します。詳細については、W&B [Public API Reference guide]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を参照してください。

{{< tabpane text=true >}}
  {{% tab header="During a run" %}}
まず、W&B Python SDK をインポートします。次に、W&B [Run]({{< relref path="/ref/python/run.md" lang="ja" >}}) を作成します。

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
```

使用したいアーティファクトを [`use_artifact`]({{< relref path="/ref/python/run.md#use_artifact" lang="ja" >}}) メソッドで指定します。これにより run オブジェクトが返されます。次のコードスニペットでは、`'bike-dataset'` というアーティファクトを `'latest'` というエイリアスで指定しています。

```python
artifact = run.use_artifact("bike-dataset:latest")
```

戻されたオブジェクトを使って、アーティファクトの内容をすべてダウンロードします。

```python
datadir = artifact.download()
```

アーティファクトの内容を特定のディレクトリーにダウンロードするには、`root` パラメータにパスをオプションで渡すことができます。詳細については、[Python SDK Reference Guide]({{< relref path="/ref/python/artifact.md#download" lang="ja" >}}) を参照してください。

[`get_path`]({{< relref path="/ref/python/artifact.md#get_path" lang="ja" >}}) メソッドを使用して、ファイルのサブセットのみをダウンロードできます。

```python
path = artifact.get_path(name)
```

これにより、パス `name` のファイルのみが取得されます。次のメソッドを持つ `Entry` オブジェクトが返されます。

* `Entry.download`: パス `name` のアーティファクトからファイルをダウンロード
* `Entry.ref`: `add_reference` がエントリーを参照として保存している場合、URI を返します。

W&B が処理方法を知っているスキームを持つ参照は、アーティファクトファイルと同様にダウンロードされます。詳細については、[Track external files]({{< relref path="/guides/core/artifacts/track-external-files.md" lang="ja" >}}) を参照してください。  
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
まず、W&B SDK をインポートします。次に、Public API クラスからアーティファクトを作成します。エンティティ、プロジェクト、アーティファクト、およびエイリアスをそのアーティファクトに関連付けます。

```python
import wandb

api = wandb.Api()
artifact = api.artifact("entity/project/artifact:alias")
```

戻されたオブジェクトを使って、アーティファクトの内容をダウンロードします。

```python
artifact.download()
```

アーティファクトの内容を特定のディレクトリーにダウンロードするために `root` パラメータにパスをオプションで渡すことができます。詳細については、[API Reference Guide]({{< relref path="/ref/python/artifact.md#download" lang="ja" >}}) を参照してください。  
  {{% /tab %}}
  {{% tab header="W&B CLI" %}}
`wandb artifact get` コマンドを使用して、W&B サーバーからアーティファクトをダウンロードします。

```
$ wandb artifact get project/artifact:alias --root mnist/
```  
  {{% /tab %}}
{{< /tabpane >}}

### アーティファクトの一部をダウンロード

プレフィックスを基にアーティファクトの一部をダウンロードすることができます。`path_prefix` パラメータを使用して、単一のファイルまたはサブフォルダーの内容をダウンロードします。

```python
artifact = run.use_artifact("bike-dataset:latest")

artifact.download(path_prefix="bike.png") # bike.png のみをダウンロード
```

または、特定のディレクトリーからファイルをダウンロードすることもできます。

```python
artifact.download(path_prefix="images/bikes/") # images/bikes ディレクトリー内のファイルをダウンロード
```

### 別のプロジェクトからアーティファクトを使用する

アーティファクトの名前とともにそのプロジェクト名を指定して、アーティファクトを参照します。また、エンティティ名でアーティファクトの名前を指定して、エンティティを超えてアーティファクトを参照することもできます。

次のコード例は、現在の W&B run に他のプロジェクトからのアーティファクトを入力としてクエリする方法を示しています。

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
# 他のプロジェクトからアーティファクトを W&B でクエリして、それを入力として
# この run にマークします。
artifact = run.use_artifact("my-project/artifact:alias")

# 別のエンティティからアーティファクトを使用し、それを入力として
# この run にマークします。
artifact = run.use_artifact("my-entity/my-project/artifact:alias")
```

### アーティファクトを同時に構築して使用する

アーティファクトを同時に構築して使用します。アーティファクト オブジェクトを作成して、それを use_artifact に渡します。これにより、W&B にアーティファクトが存在しない場合は作成されます。[`use_artifact`]({{< relref path="/ref/python/run.md#use_artifact" lang="ja" >}}) API は冪等性があり、あなたが好きなだけ何度も呼び出すことができます。

```python
import wandb

artifact = wandb.Artifact("reference model")
artifact.add_file("model.h5")
run.use_artifact(artifact)
```

アーティファクトの構築に関する詳細については、[Construct an artifact]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ja" >}}) を参照してください。