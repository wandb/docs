---
title: Create an artifact version
description: 単一の run 、または分散された プロセス から新しい アーティファクト の バージョン を作成します。
menu:
  default:
    identifier: ja-guides-core-artifacts-create-a-new-artifact-version
    parent: artifacts
weight: 6
---

単一の [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) で、または分散されたrunと共同で、新しいアーティファクトのバージョンを作成します。オプションで、[インクリメンタルアーティファクト]({{< relref path="#create-a-new-artifact-version-from-an-existing-version" lang="ja" >}}) として知られる、以前のバージョンから新しいアーティファクトのバージョンを作成できます。

{{% alert %}}
元のアーティファクトのサイズが著しく大きい場合に、アーティファクト内のファイルのサブセットに変更を適用する必要がある場合は、インクリメンタルアーティファクトを作成することをお勧めします。
{{% /alert %}}

## 新しいアーティファクトのバージョンをゼロから作成する
新しいアーティファクトのバージョンを作成する方法は2つあります。単一のrunから作成する方法と、分散されたrunから作成する方法です。それらは次のように定義されます。

* **単一のrun**: 単一のrunは、新しいバージョンのすべてのデータを提供します。これは最も一般的なケースであり、runが必要なデータを完全に再作成する場合に最適です。例：分析のために保存されたモデルまたはモデルの予測をテーブルに出力するなど。
* **分散されたrun**: runのセットが集合的に新しいバージョンのすべてのデータを提供します。これは、多くの場合並行してデータを生成する複数のrunを持つ分散ジョブに最適です。例：分散方式でモデルを評価し、予測を出力するなど。

W&B は、プロジェクトに存在しない名前を `wandb.Artifact` API に渡すと、新しいアーティファクトを作成し、`v0` エイリアスを割り当てます。同じアーティファクトに再度ログを記録すると、W&B はコンテンツのチェックサムを計算します。アーティファクトが変更された場合、W&B は新しいバージョン `v1` を保存します。

`wandb.Artifact` API に名前とアーティファクトのタイプを渡し、それがプロジェクト内の既存のアーティファクトと一致する場合、W&B は既存のアーティファクトを取得します。取得されたアーティファクトのバージョンは1より大きくなります。

{{< img src="/images/artifacts/single_distributed_artifacts.png" alt="" >}}

### 単一のrun
アーティファクト内のすべてのファイルを生成する単一のrunで、Artifactの新しいバージョンをログに記録します。このケースは、単一のrunがアーティファクト内のすべてのファイルを生成する場合に発生します。

ユースケースに基づいて、以下のタブのいずれかを選択して、runの内側または外側に新しいアーティファクトのバージョンを作成します。

{{< tabpane text=true >}}
  {{% tab header="runの内側" %}}
W&B run内でアーティファクトのバージョンを作成します。

1. `wandb.init` でrunを作成します。
2. `wandb.Artifact` で新しいアーティファクトを作成するか、既存のアーティファクトを取得します。
3. `.add_file` でアーティファクトにファイルを追加します。
4. `.log_artifact` でアーティファクトをrunにログ記録します。

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # `.add`、`.add_file`、`.add_dir`、および `.add_reference` を使用して、
    # ファイルとアセットをアーティファクトに追加します
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```
  {{% /tab %}}
  {{% tab header="runの外側" %}}
W&B runの外側でアーティファクトのバージョンを作成します。

1. `wanb.Artifact` で新しいアーティファクトを作成するか、既存のアーティファクトを取得します。
2. `.add_file` でアーティファクトにファイルを追加します。
3. `.save` でアーティファクトを保存します。

```python
artifact = wandb.Artifact("artifact_name", "artifact_type")
# `.add`、`.add_file`、`.add_dir`、および `.add_reference` を使用して、
# ファイルとアセットをアーティファクトに追加します
artifact.add_file("image1.png")
artifact.save()
```
  {{% /tab %}}
{{< /tabpane >}}

### 分散されたrun

コミットする前に、runのコレクションがバージョンで共同作業できるようにします。これは、1つのrunが新しいバージョンのすべてのデータを提供する上記の単一runモードとは対照的です。

{{% alert %}}
1. コレクション内の各runは、同じバージョンで共同作業するために、同じ一意のID（`distributed_id` と呼ばれます）を認識している必要があります。デフォルトでは、存在する場合、W&B は `wandb.init(group=GROUP)` によって設定されたrunの `group` を `distributed_id` として使用します。
2. バージョンの状態を永続的にロックする、バージョンを「コミット」する最終的なrunが必要です。
3. 共同アーティファクトに追加するには `upsert_artifact` を使用し、コミットを完了するには `finish_artifact` を使用します。
{{% /alert %}}

次の例を検討してください。異なるrun（以下では **Run 1**、**Run 2**、および **Run 3** としてラベル付け）は、`upsert_artifact` で同じアーティファクトに異なる画像ファイルを追加します。

#### Run 1:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`、`.add_file`、`.add_dir`、および `.add_reference` を使用して、
    # ファイルとアセットをアーティファクトに追加します
    artifact.add_file("image1.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 2:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`、`.add_file`、`.add_dir`、および `.add_reference` を使用して、
    # ファイルとアセットをアーティファクトに追加します
    artifact.add_file("image2.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 3

Run 1とRun 2が完了した後に実行する必要があります。`finish_artifact` を呼び出すRunはアーティファクトにファイルを含めることができますが、必須ではありません。

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # ファイルとアセットをアーティファクトに追加します
    # `.add`、`.add_file`、`.add_dir`、および `.add_reference`
    artifact.add_file("image3.png")
    run.finish_artifact(artifact, distributed_id="my_dist_artifact")
```

## 既存のバージョンから新しいアーティファクトのバージョンを作成する

変更されていないファイルを再インデックスする必要なく、以前のアーティファクトのバージョンからファイルのサブセットを追加、変更、または削除します。以前のアーティファクトのバージョンからファイルのサブセットを追加、変更、または削除すると、*インクリメンタルアーティファクト*として知られる新しいアーティファクトのバージョンが作成されます。

{{< img src="/images/artifacts/incremental_artifacts.png" alt="" >}}

発生する可能性のあるインクリメンタルな変更のタイプごとのシナリオを次に示します。

- add: 新しいバッチを収集した後、データセットに新しいファイルのサブセットを定期的に追加します。
- remove: 複数の重複ファイルを発見し、アーティファクトから削除したいとします。
- update: ファイルのサブセットのアノテーションを修正し、古いファイルを正しいファイルに置き換えたいとします。

インクリメンタルアーティファクトと同じ機能を実行するために、アーティファクトをゼロから作成できます。ただし、アーティファクトをゼロから作成する場合、アーティファクトのすべてのコンテンツをローカルディスクに用意する必要があります。インクリメンタルな変更を行う場合、以前のアーティファクトのバージョンからファイルを変更せずに、単一のファイルを追加、削除、または変更できます。

{{% alert %}}
インクリメンタルアーティファクトは、単一のrun内またはrunのセット（分散モード）で作成できます。
{{% /alert %}}

以下の手順に従って、アーティファクトをインクリメンタルに変更します。

1. インクリメンタルな変更を実行するアーティファクトのバージョンを取得します。

{{< tabpane text=true >}}
{{% tab header="runの内側" %}}

```python
saved_artifact = run.use_artifact("my_artifact:latest")
```

{{% /tab %}}
{{% tab header="runの外側" %}}

```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")
```

{{% /tab %}}
{{< /tabpane >}}

2. 次を使用してドラフトを作成します。

```python
draft_artifact = saved_artifact.new_draft()
```

3. 次のバージョンで表示したいインクリメンタルな変更を実行します。既存のエントリを追加、削除、または変更できます。

これらの変更を実行する方法の例については、以下のタブのいずれかを選択してください。

{{< tabpane text=true >}}
  {{% tab header="Add" %}}
`add_file` メソッドを使用して、既存のアーティファクトのバージョンにファイルを追加します。

```python
draft_artifact.add_file("file_to_add.txt")
```

{{% alert %}}
`add_dir` メソッドを使用してディレクトリーを追加することで、複数のファイルを追加することもできます。
{{% /alert %}}
  {{% /tab %}}
  {{% tab header="Remove" %}}
`remove` メソッドを使用して、既存のアーティファクトのバージョンからファイルを削除します。

```python
draft_artifact.remove("file_to_remove.txt")
```

{{% alert %}}
ディレクトリーパスを渡すことで、`remove` メソッドを使用して複数のファイルを削除することもできます。
{{% /alert %}}
  {{% /tab %}}
  {{% tab header="Modify" %}}
ドラフトから古いコンテンツを削除し、新しいコンテンツを再度追加して、コンテンツを変更または置き換えます。

```python
draft_artifact.remove("modified_file.txt")
draft_artifact.add_file("modified_file.txt")
```
  {{% /tab %}}
{{< /tabpane >}}

4. 最後に、変更をログに記録または保存します。以下のタブは、W&B runの内側と外側で変更を保存する方法を示しています。ユースケースに合ったタブを選択してください。

{{< tabpane text=true >}}
  {{% tab header="runの内側" %}}
```python
run.log_artifact(draft_artifact)
```

  {{% /tab %}}
  {{% tab header="runの外側" %}}
```python
draft_artifact.save()
```
  {{% /tab %}}
{{< /tabpane >}}

まとめると、上記のコード例は次のようになります。

{{< tabpane text=true >}}
  {{% tab header="runの内側" %}}
```python
with wandb.init(job_type="データセットの変更") as run:
    saved_artifact = run.use_artifact(
        "my_artifact:latest"
    )  # アーティファクトをフェッチしてrunに入力します
    draft_artifact = saved_artifact.new_draft()  # ドラフトバージョンを作成します

    # ドラフトバージョン内のファイルのサブセットを変更します
    draft_artifact.add_file("file_to_add.txt")
    draft_artifact.remove("dir_to_remove/")
    run.log_artifact(
        artifact
    )  # 変更をログに記録して新しいバージョンを作成し、runへの出力としてマークします
```
  {{% /tab %}}
  {{% tab header="runの外側" %}}
```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")  # アーティファクトをロードします
draft_artifact = saved_artifact.new_draft()  # ドラフトバージョンを作成します

# ドラフトバージョン内のファイルのサブセットを変更します
draft_artifact.remove("deleted_file.txt")
draft_artifact.add_file("modified_file.txt")
draft_artifact.save()  # ドラフトへの変更をコミットします
```
  {{% /tab %}}
{{< /tabpane >}}