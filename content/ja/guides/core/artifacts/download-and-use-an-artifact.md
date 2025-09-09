---
title: Artifacts をダウンロードして使用する
description: 複数の Projects から Artifacts をダウンロードして使用します。
menu:
  default:
    identifier: ja-guides-core-artifacts-download-and-use-an-artifact
    parent: artifacts
weight: 3
---

W&B サーバーに既に保存されている Artifact をダウンロードして使用するか、Artifact オブジェクトを構築し、必要に応じて重複排除のために渡します。
{{% alert %}}
閲覧専用シートを持つチームメンバーは、Artifacts をダウンロードできません。
{{% /alert %}}

### W&B に保存されている Artifact をダウンロードして使用する
W&B Run の内部または外部で、W&B に保存されている Artifact をダウンロードして使用します。Public API ([`wandb.Api`]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}})) を使用して、W&B に既に保存されているデータをエクスポート (または更新) します。詳細については、W&B の [Public API リファレンスガイド]({{< relref path="/ref/python/public-api/index.md" lang="ja" >}}) を参照してください。

{{< tabpane text=true >}}
  {{% tab header="run 中" %}}
まず、W&B Python SDK をインポートします。次に、W&B [Run]({{< relref path="/ref/python/sdk/classes/run.md" lang="ja" >}}) を作成します。
```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
```
使用したい Artifact を [`use_artifact`]({{< relref path="/ref/python/sdk/classes/run.md#use_artifact" lang="ja" >}}) メソッドで指定します。これにより、run オブジェクトが返されます。次のコードスニペットでは、`'bike-dataset'` という Artifact を `'latest'` というエイリアスで指定しています。
```python
artifact = run.use_artifact("bike-dataset:latest")
```
返されたオブジェクトを使用して、Artifact のすべてのコンテンツをダウンロードします。
```python
datadir = artifact.download()
```
オプションで、`root` パラメータにパスを渡すことで、Artifact のコンテンツを特定のディレクトリーにダウンロードできます。詳細については、[Python SDK リファレンスガイド]({{< relref path="/ref/python/sdk/classes/artifact.md#download" lang="ja" >}}) を参照してください。
[`get_path`]({{< relref path="/ref/python/sdk/classes/artifact.md#get_path" lang="ja" >}}) メソッドを使用して、ファイルのサブセットのみをダウンロードします。
```python
path = artifact.get_path(name)
```
これは、パス `name` にあるファイルのみを取得します。次のメソッドを持つ `Entry` オブジェクトを返します。
* `Entry.download`: パス `name` にある Artifact からファイルをダウンロードします
* `Entry.ref`: `add_reference` がエントリーをリファレンスとして保存した場合、URI を返します
W&B が処理する方法を知っているスキームを持つリファレンスは、Artifact ファイルと同じようにダウンロードされます。詳細については、[外部ファイルを追跡する]({{< relref path="/guides/core/artifacts/track-external-files.md" lang="ja" >}}) を参照してください。
  {{% /tab %}}
  {{% tab header="run 外" %}}
まず、W&B SDK をインポートします。次に、Public API クラスから Artifact を作成します。その Artifact に関連付けられた Entity、Project、Artifact、および alias を指定します。
```python
import wandb

api = wandb.Api()
artifact = api.artifact("entity/project/artifact:alias")
```
返されたオブジェクトを使用して、Artifact のコンテンツをダウンロードします。
```python
artifact.download()
```
オプションで、`root` パラメータにパスを渡すことで、Artifact のコンテンツを特定のディレクトリーにダウンロードできます。詳細については、[API リファレンスガイド]({{< relref path="/ref/python/sdk/classes/artifact.md#download" lang="ja" >}}) を参照してください。
  {{% /tab %}}
  {{% tab header="W&B CLI" %}}
`wandb artifact get` コマンドを使用して、W&B サーバーから Artifact をダウンロードします。
```
$ wandb artifact get project/artifact:alias --root mnist/
```
  {{% /tab %}}
{{< /tabpane >}}

### Artifact を部分的にダウンロードする
オプションで、プレフィックスに基づいて Artifact の一部をダウンロードできます。`path_prefix` パラメータを使用すると、単一のファイルまたはサブフォルダーのコンテンツをダウンロードできます。
```python
artifact = run.use_artifact("bike-dataset:latest")

artifact.download(path_prefix="bike.png") # bike.png のみをダウンロードします
```
あるいは、特定のディレクトリーからファイルをダウンロードできます。
```python
artifact.download(path_prefix="images/bikes/") # images/bikes ディレクトリー内のファイルをダウンロードします
```

### 別の Project から Artifact を使用する
Artifact を参照するには、Project 名とともに Artifact の名前を指定します。Entity 名とともに Artifact の名前を指定することで、複数の Entity にわたって Artifact を参照することもできます。
次のコード例は、別の Project からの Artifact を現在の W&B run への入力としてクエリする方法を示しています。
```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
# 別の project からの artifact を W&B にクエリし、それを
# この run への入力としてマークします。
artifact = run.use_artifact("my-project/artifact:alias")

# 別の entity からの artifact を使用し、それを入力としてマークします
# この run へ。
artifact = run.use_artifact("my-entity/my-project/artifact:alias")
```

### Artifact を同時に構築して使用する
Artifact を同時に構築して使用します。Artifact オブジェクトを作成し、`use_artifact` に渡します。これにより、まだ存在しない場合は W&B に Artifact が作成されます。[`use_artifact`]({{< relref path="/ref/python/sdk/classes/run.md#use_artifact" lang="ja" >}}) API は冪等であるため、好きなだけ呼び出すことができます。
```python
import wandb

artifact = wandb.Artifact("reference model")
artifact.add_file("model.h5")
run.use_artifact(artifact)
```
Artifact の構築の詳細については、[Artifact を構築する]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ja" >}}) を参照してください。