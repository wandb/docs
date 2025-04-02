---
title: Download and use artifacts
description: 複数のプロジェクトから Artifacts をダウンロードして使用します。
menu:
  default:
    identifier: ja-guides-core-artifacts-download-and-use-an-artifact
    parent: artifacts
weight: 3
---

W&B サーバーにすでに保存されている Artifacts をダウンロードして使用するか、必要に応じて重複排除のために Artifacts オブジェクトを構築して渡します。

{{% alert %}}
表示専用のシートを持つチームメンバーは、Artifacts をダウンロードできません。
{{% /alert %}}

### W&B に保存されている Artifacts のダウンロードと使用

W&B Run の内部または外部で、W&B に保存されている Artifacts をダウンロードして使用します。Public API ([`wandb.Api`]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}})) を使用して、W&B にすでに保存されているデータをエクスポート (または更新) します。詳細については、W&B の[Public API Reference guide]({{< relref path="/ref/python/public-api/" lang="ja" >}})を参照してください。

{{< tabpane text=true >}}
  {{% tab header="Run 中" %}}
まず、W&B Python SDK をインポートします。次に、W&B [Run]({{< relref path="/ref/python/run.md" lang="ja" >}})を作成します。

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
```

[`use_artifact`]({{< relref path="/ref/python/run.md#use_artifact" lang="ja" >}}) メソッドを使用して、使用する Artifacts を指定します。これにより、run オブジェクトが返されます。次のコードスニペットでは、エイリアス `'latest'` を持つ `'bike-dataset'` という Artifacts を指定します。

```python
artifact = run.use_artifact("bike-dataset:latest")
```

返されたオブジェクトを使用して、Artifacts のすべてのコンテンツをダウンロードします。

```python
datadir = artifact.download()
```

オプションで、ルートパラメータにパスを渡して、Artifacts のコンテンツを特定のディレクトリーにダウンロードできます。詳細については、[Python SDK Reference Guide]({{< relref path="/ref/python/artifact.md#download" lang="ja" >}})を参照してください。

[`get_path`]({{< relref path="/ref/python/artifact.md#get_path" lang="ja" >}}) メソッドを使用して、ファイルのサブセットのみをダウンロードします。

```python
path = artifact.get_path(name)
```

これは、パス `name` にあるファイルのみをフェッチします。これは、次のメソッドを持つ `Entry` オブジェクトを返します。

* `Entry.download`: パス `name` にある Artifacts からファイルをダウンロードします
* `Entry.ref`: `add_reference` がエントリをリファレンスとして保存した場合、URI を返します

W&B が処理する方法を知っているスキームを持つリファレンスは、Artifacts ファイルとまったく同じようにダウンロードされます。詳細については、[Track external files]({{< relref path="/guides/core/artifacts/track-external-files.md" lang="ja" >}})を参照してください。
  {{% /tab %}}
  {{% tab header="Run の外部" %}}
まず、W&B SDK をインポートします。次に、Public API クラスから Artifacts を作成します。その Artifacts に関連付けられたエンティティ、プロジェクト、Artifacts、エイリアスを指定します。

```python
import wandb

api = wandb.Api()
artifact = api.artifact("entity/project/artifact:alias")
```

返されたオブジェクトを使用して、Artifacts のコンテンツをダウンロードします。

```python
artifact.download()
```

オプションで、`root` パラメータにパスを渡して、Artifacts のコンテンツを特定のディレクトリーにダウンロードできます。詳細については、[API Reference Guide]({{< relref path="/ref/python/artifact.md#download" lang="ja" >}})を参照してください。
  {{% /tab %}}
  {{% tab header="W&B CLI" %}}
`wandb artifact get` コマンドを使用して、W&B サーバーから Artifacts をダウンロードします。

```
$ wandb artifact get project/artifact:alias --root mnist/
```
  {{% /tab %}}
{{< /tabpane >}}

### Artifacts の部分的なダウンロード

オプションで、プレフィックスに基づいて Artifacts の一部をダウンロードできます。`path_prefix` パラメータを使用すると、単一のファイルまたはサブフォルダーのコンテンツをダウンロードできます。

```python
artifact = run.use_artifact("bike-dataset:latest")

artifact.download(path_prefix="bike.png") # bike.png のみをダウンロードします
```

または、特定のディレクトリーからファイルをダウンロードすることもできます。

```python
artifact.download(path_prefix="images/bikes/") # images/bikes ディレクトリー内のファイルをダウンロードします
```
### 別のプロジェクトの Artifacts の使用

Artifacts の名前をプロジェクト名とともに指定して、Artifacts を参照します。エンティティ名とともに Artifacts の名前を指定することで、エンティティ間で Artifacts を参照することもできます。

次のコード例は、別のプロジェクトから Artifacts をクエリして、現在の W&B run への入力として使用する方法を示しています。

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
# 別のプロジェクトから Artifacts の W&B をクエリして、それをマークします
# この run への入力として。
artifact = run.use_artifact("my-project/artifact:alias")

# 別のエンティティからの Artifacts を使用して、入力としてマークします
# この run に。
artifact = run.use_artifact("my-entity/my-project/artifact:alias")
```

### Artifacts の同時構築と使用

Artifacts を同時に構築して使用します。Artifacts オブジェクトを作成し、それを use_artifact に渡します。これにより、Artifacts がまだ存在しない場合は W&B に作成されます。[`use_artifact`]({{< relref path="/ref/python/run.md#use_artifact" lang="ja" >}}) API はべき等であるため、必要な回数だけ呼び出すことができます。

```python
import wandb

artifact = wandb.Artifact("reference model")
artifact.add_file("model.h5")
run.use_artifact(artifact)
```

Artifacts の構築の詳細については、[Construct an artifact]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ja" >}})を参照してください。
