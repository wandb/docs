---
title: アーティファクト バージョンを作成する
description: 新しいアーティファクト バージョンを単一の run または分散プロセスから作成します。
menu:
  default:
    identifier: ja-guides-core-artifacts-create-a-new-artifact-version
    parent: artifacts
weight: 6
---

新しいアーティファクトバージョンをシングル [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) で作成するか、分散 run を使って共同で作成します。以前のバージョンから新しいアーティファクトバージョンを作成することもできます。これを [インクリメンタルアーティファクト]({{< relref path="#create-a-new-artifact-version-from-an-existing-version" lang="ja" >}}) と呼びます。

{{% alert %}}
アーティファクト内のファイルの一部に変更を加える必要がある場合、元のアーティファクトのサイズがかなり大きい場合は、インクリメンタルアーティファクトを作成することをお勧めします。
{{% /alert %}}

## 新しいアーティファクトバージョンをゼロから作成する
新しいアーティファクトバージョンを作成する方法は、シングル run と分散 run による2つがあります。それぞれ次のように定義されています:

* **シングル run**: シングル run が新しいバージョンのすべてのデータを提供します。これは最も一般的なケースで、run が必要なデータを完全に再現する場合に最適です。例: 保存されたモデルやモデル予測を分析用のテーブルに出力する。
* **分散 run**: 複数の run のセットが共同して新しいバージョンのすべてのデータを提供します。これは、複数の run が並行してデータを生成する分散ジョブに最適です。例: モデルを分散的に評価し、予測を出力する。

W&B は、プロジェクト内に存在しない名前を `wandb.Artifact` API に渡すと、新しいアーティファクトを作成し、それに `v0` エイリアスを割り当てます。同じアーティファクトに再度ログを記録する際に内容をチェックサムします。アーティファクトが変更されている場合、W&B は新しいバージョン `v1` を保存します。

プロジェクト内に既存のアーティファクトと一致する名前とアーティファクトタイプを `wandb.Artifact` API に渡すと、W&B は既存のアーティファクトを取得します。取得されたアーティファクトはバージョンが 1 より大きくなります。

{{< img src="/images/artifacts/single_distributed_artifacts.png" alt="" >}}

### シングル run
アーティファクト内のすべてのファイルを生成するシングル run によって、新しいバージョンのアーティファクトをログします。このケースは、シングル run がアーティファクト内のすべてのファイルを生成する場合に発生します。

ユースケースに基づいて、以下のタブのいずれかを選択して、run 内または run 外で新しいアーティファクトバージョンを作成してください:

{{< tabpane text=true >}}
  {{% tab header="Inside a run" %}}
W&B run 内でアーティファクトバージョンを作成します:

1. `wandb.init` を使って run を作成。
2. `wandb.Artifact` で新しいアーティファクトを作成するか、既存のアーティファクトを取得。
3. `.add_file` を使用してファイルをアーティファクトに追加。
4. `.log_artifact` を使ってアーティファクトを run にログ。

```python 
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # Add Files and Assets to the artifact using
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```  
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
W&B run の外でアーティファクトバージョンを作成します:

1. `wanb.Artifact` で新しいアーティファクトを作成するか、既存のアーティファクトを取得。
2. `.add_file` を使用してファイルをアーティファクトに追加。
3. `.save` でアーティファクトを保存。

```python 
artifact = wandb.Artifact("artifact_name", "artifact_type")
# Add Files and Assets to the artifact using
# `.add`, `.add_file`, `.add_dir`, and `.add_reference`
artifact.add_file("image1.png")
artifact.save()
```    
  {{% /tab %}}
{{< /tabpane  >}}

### 分散 run

バージョンをコミットする前に、複数の run が共同で作業します。これは、上記のシングル run モードとは対照的です。こちらは1つの run が新しいバージョンのすべてのデータを提供します。

{{% alert %}}
1. コレクション内の各 run は、同じバージョンで共同作業をするために、同じユニークな ID ( `distributed_id` と呼ばれる) を認識している必要があります。デフォルトでは、存在する場合、W&B は run の `group` を、`wandb.init(group=GROUP)` によって設定された `distributed_id` として使用します。
2. バージョンを「コミット」し、その状態を永続的にロックする最終 run が必要です。
3. 協調的なアーティファクトに追加するには `upsert_artifact` を使用し、コミットを最終的にするには `finish_artifact` を使用します。
{{% /alert %}}

以下の例を考えてみてください。異なる run (以下で **Run 1**、**Run 2**、**Run 3** とラベル付けされている) が `upsert_artifact` を使って同じアーティファクトに異なる画像ファイルを追加します。

#### Run 1

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # Add Files and Assets to the artifact using
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`
    artifact.add_file("image1.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 2

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # Add Files and Assets to the artifact using
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`
    artifact.add_file("image2.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 3

Run 1 と Run 2 が完了した後に実行する必要があります。`finish_artifact` を呼び出す Run は、アーティファクトにファイルを含めることができますが、必須ではありません。

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # Add Files and Assets to the artifact
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`
    artifact.add_file("image3.png")
    run.finish_artifact(artifact, distributed_id="my_dist_artifact")
```

## 既存のバージョンから新しいアーティファクトバージョンを作成する

前のアーティファクトバージョンからファイルのサブセットを追加、変更、または削除して、変更されていないファイルを再インデックスする必要はありません。前のアーティファクトバージョンからファイルのサブセットを追加、変更、または削除すると、新しいアーティファクトバージョンが作成され、これを*インクリメンタルアーティファクト*と呼びます。

{{< img src="/images/artifacts/incremental_artifacts.png" alt="" >}}

以下は、遭遇する可能性のあるインクリメンタルな変更の各タイプに対するシナリオです:

- add: 新しいバッチを収集した後、定期的にデータセットに新しいファイルのサブセットを追加します。
- remove: 重複ファイルをいくつか発見し、アーティファクトからそれらを削除することを希望します。
- update: ファイルのサブセットに対する注釈を修正し、古いファイルを正しいものと置き換えます。

インクリメンタルアーティファクトとしての同じ機能を実行するためにアーティファクトをゼロから作成することもできます。しかし、アーティファクトをゼロから作成する場合、アーティファクトのすべての内容をローカルディスクに持っている必要があります。インクリメンタルな変更を行う場合、前のアーティファクトバージョンのファイルを変更せずに、個々のファイルを追加、削除、または変更できます。

{{% alert %}}
単一の run で、または複数の run (分散モード) でインクリメンタルアーティファクトを作成できます。
{{% /alert %}}

以下の手順に従って、アーティファクトをインクリメンタルに変更します:

1. インクリメンタル変更を行いたいアーティファクトバージョンを取得します:

{{< tabpane text=true >}}
{{% tab header="Inside a run" %}}

```python
saved_artifact = run.use_artifact("my_artifact:latest")
```

{{% /tab %}}
{{% tab header="Outside of a run" %}}

```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")
```

{{% /tab %}}
{{< /tabpane >}}

2. 以下の方法でドラフトを作成します:

```python
draft_artifact = saved_artifact.new_draft()
```

3. 次のバージョンで見たいインクリメンタルな変更を行います。既存のエントリーを追加、削除、または変更することができます。

各変更を行うための例については、以下のいずれかのタブを選択してください:

{{< tabpane text=true >}}
  {{% tab header="Add" %}}
`add_file` メソッドで既存のアーティファクトバージョンにファイルを追加します:

```python
draft_artifact.add_file("file_to_add.txt")
```

{{% alert %}}
`add_dir` メソッドを使用してディレクトリを追加することで、複数のファイルを追加することもできます。
{{% /alert %}}  
  {{% /tab %}}
  {{% tab header="Remove" %}}
`remove` メソッドで既存のアーティファクトバージョンからファイルを削除します:

```python
draft_artifact.remove("file_to_remove.txt")
```

{{% alert %}}
`remove` メソッドにディレクトリパスを渡すことで、複数のファイルを削除することもできます。
{{% /alert %}}  
  {{% /tab %}}
  {{% tab header="Modify" %}}
ドラフトから古い内容を削除し、新しい内容を追加することで、内容を変更または置き換えます:

```python
draft_artifact.remove("modified_file.txt")
draft_artifact.add_file("modified_file.txt")
```  
  {{% /tab %}}
{{< /tabpane >}}

4. 最後に、変更をログまたは保存します。以下のタブは、W&B run の内外で変更を保存する方法を示しています。適切なユースケースに応じてタブを選択してください:

{{< tabpane text=true >}}
  {{% tab header="Inside a run" %}}
```python
run.log_artifact(draft_artifact)
```

  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
```python
draft_artifact.save()
```  
  {{% /tab %}}
{{< /tabpane >}}

以上のコード例をまとめると、以下のようになります:

{{< tabpane text=true >}}
  {{% tab header="Inside a run" %}}
```python
with wandb.init(job_type="modify dataset") as run:
    saved_artifact = run.use_artifact(
        "my_artifact:latest"
    )  # fetch artifact and input it into your run
    draft_artifact = saved_artifact.new_draft()  # create a draft version

    # modify a subset of files in the draft version
    draft_artifact.add_file("file_to_add.txt")
    draft_artifact.remove("dir_to_remove/")
    run.log_artifact(
        artifact
    )  # log your changes to create a new version and mark it as output to your run
```  
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")  # load your artifact
draft_artifact = saved_artifact.new_draft()  # create a draft version

# modify a subset of files in the draft version
draft_artifact.remove("deleted_file.txt")
draft_artifact.add_file("modified_file.txt")
draft_artifact.save()  # commit changes to the draft
```  
  {{% /tab %}}
{{< /tabpane >}}