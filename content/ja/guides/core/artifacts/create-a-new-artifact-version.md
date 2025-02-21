---
title: Create an artifact version
description: 単一の run または分散プロセスから新しいアーティファクトバージョンを作成します。
menu:
  default:
    identifier: ja-guides-core-artifacts-create-a-new-artifact-version
    parent: artifacts
weight: 6
---

新しいアーティファクト バージョンを、1 つの [Run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) または共同で分散 Run で作成します。既存のバージョンから新しいアーティファクト バージョンを作成することもできます。これは [増分アーティファクト]({{< relref path="#create-a-new-artifact-version-from-an-existing-version" lang="ja" >}}) として知られています。

{{% alert %}}
元のアーティファクトのサイズが大きい場合にアーティファクト内のファイルの一部を変更する必要がある場合は、増分アーティファクトを作成することをお勧めします。
{{% /alert %}}

## 新しいアーティファクト バージョンをゼロから作成する
新しいアーティファクト バージョンを作成する方法は 2 つあります: 単一の Run から作成する方法と、分散 Run から作成する方法です。それらは次のように定義されます。

* **単一の run**: 単一の Run が新しいバージョンのためのすべてのデータを提供します。これは最も一般的なケースであり、Run が必要なデータを完全に再現する場合に最適です。例えば、テーブルに保存されたモデルやモデルの予測を出力して分析を行う場合などです。
* **分散 run**: 一連の Run は共同で新しいバージョンのすべてのデータを提供します。これは、複数の Run がデータを生成する分散ジョブに最適です。たとえば、モデルを分散方式で評価し、予測を出力する場合などです。

W&B は、プロジェクトに存在しない名前を `wandb.Artifact` API に渡すと、新しいアーティファクトを作成して `v0` エイリアスを割り当てます。同じアーティファクトに再度ログする際に、W&B は内容をチェックサムします。アーティファクトが変更された場合、W&B は新しいバージョン `v1` を保存します。

W&B は、プロジェクト内の既存のアーティファクトに一致する名前とアーティファクト タイプを `wandb.Artifact` API に渡すと、既存のアーティファクトを取得します。取得されたアーティファクトは、1 より大きいバージョンを持ちます。

{{< img src="/images/artifacts/single_distributed_artifacts.png" alt="" >}}

### 単一の run
1 つの Run でアーティファクト内のすべてのファイルを生成して、新しいバージョンのアーティファクトをログします。このケースは、単一の Run がアーティファクト内のすべてのファイルを生成する場合に発生します。

ユースケースに応じて、Run の内外で新しいアーティファクト バージョンを作成するには、以下のタブのいずれかを選択してください。

{{< tabpane text=true >}}
  {{% tab header="Run 内" %}}
W&B の Run 内でアーティファクト バージョンを作成します:

1. `wandb.init` を使用して Run を作成します。 (行 1)
2. `wandb.Artifact` を使用して新しいアーティファクトを作成するか、既存のものを取得します。 (行 2)
3. `.add_file` を使用してアーティファクトにファイルを追加します。 (行 9)
4. `.log_artifact` を使用してアーティファクトを Run にログします。 (行 10)

```python showLineNumbers
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # `.add`, `.add_file`, `.add_dir`, and `.add_reference` を使用してファイルとアセットをアーティファクトに追加します
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```  
  {{% /tab %}}
  {{% tab header="Run 外" %}}
W&B の Run 外でアーティファクト バージョンを作成します:

1. `wanb.Artifact` を使用して新しいアーティファクトを作成するか、既存のものを取得します。 (行 1)
2. `.add_file` を使用してアーティファクトにファイルを追加します。 (行 4)
3. `.save` を使用してアーティファクトを保存します。 (行 5)

```python showLineNumbers
artifact = wandb.Artifact("artifact_name", "artifact_type")
# `.add`, `.add_file`, `.add_dir`, and `.add_reference` を使用してファイルとアセットをアーティファクトに追加します
artifact.add_file("image1.png")
artifact.save()
```    
  {{% /tab %}}
{{< /tabpane  >}}

### 分散 run

バージョンをコミットする前に、一連の Run が共同で作業できるようにします。これは、上記で説明した単一 Run モードとは対照的で、1 つの Run がすべてのデータを新しいバージョンに提供します。

{{% alert %}}
1. コレクションの各 Run は同じ一意のID (`distributed_id` と呼ばれる) を認識している必要があります。同じバージョンで共同作業を行うためです。デフォルトでは、W&B は存在する場合、`wandb.init(group=GROUP)` によって設定された Run の `group` を `distributed_id` として使用します。
2. バージョンを「コミット」する最終的な Run が必要です。この Run は永久にその状態をロックします。
3. `upsert_artifact` を使用して共同アーティファクトに追加し、`finish_artifact` を使用してコミットを確定します。
{{% /alert %}}

次の例を考えてみてください。異なる Run (以下で **Run 1**、**Run 2**、および **Run 3** とラベル付けされています) が、それぞれ異なる画像ファイルを同じアーティファクトに `upsert_artifact` を使用して追加します。

#### Run 1:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference` を使用してファイルとアセットをアーティファクトに追加します
    artifact.add_file("image1.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 2:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference` を使用してファイルとアセットをアーティファクトに追加します
    artifact.add_file("image2.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 3

Run 1 と Run 2 が完了した後に実行する必要があります。`finish_artifact` を呼び出す Run は、アーティファクトにファイルを含めることができますが、含める必要はありません。

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # ファイルとアセット `.add`, `.add_file`, `.add_dir`, and `.add_reference` をアーティファクトに追加します
    artifact.add_file("image3.png")
    run.finish_artifact(artifact, distributed_id="my_dist_artifact")
```

## 既存のバージョンから新しいアーティファクト バージョンを作成する

以前のアーティファクト バージョンからファイルの一部を追加、変更、または削除して、変更されていないファイルを再インデックスする必要はありません。以前のアーティファクト バージョンからファイルの一部を追加、変更、または削除すると、新しいアーティファクト バージョンが作成され、*増分アーティファクト* として知られます。

{{< img src="/images/artifacts/incremental_artifacts.png" alt="" >}}

次に、遭遇する可能性のある増分変更の各タイプのシナリオを示します。

- 追加: 新しいバッチを収集した後、定期的にデータセットに新しいファイルのサブセットを追加します。
- 削除: 複数の重複ファイルを発見し、それらをアーティファクトから削除したい場合です。
- 更新: ファイルのサブセットのアノテーションを修正し、古いファイルを正しいものと置き換えたい場合です。

同じ機能を増分アーティファクトで行うために、アーティファクトをゼロから作成することもできます。ただし、アーティファクトをゼロから作成すると、アーティファクトのすべてのコンテンツをローカルディスクに持っている必要があります。増分変更を行う際には、以前のアーティファクト バージョンのファイルを変更せずに、単一のファイルを追加、削除、または変更することができます。

{{% alert %}}
増分アーティファクトは単一の Run 内、または一連の Run (分散モード) で作成できます。
{{% /alert %}}

以下の手順に従ってアーティファクトを増分的に変更します。

1. 増分変更を行いたいアーティファクト バージョンを取得します。

{{< tabpane text=true >}}
{{% tab header="Run 内" %}}

```python
saved_artifact = run.use_artifact("my_artifact:latest")
```

{{% /tab %}}
{{% tab header="Run 外" %}}

```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")
```

{{% /tab %}}
{{< /tabpane >}}

2. 下記を使用してドラフトを作成します。

```python
draft_artifact = saved_artifact.new_draft()
```

3. 次のバージョンで見たい増分変更を行います。既存のエントリを追加、削除、または変更できます。

これらの変更を実行する方法については、各タブを選択してください。

{{< tabpane text=true >}}
  {{% tab header="追加" %}}
`add_file` メソッドを使用して、既存のアーティファクト バージョンにファイルを追加します。

```python
draft_artifact.add_file("file_to_add.txt")
```

{{% alert %}}
ディレクトリを追加することで `add_dir` メソッドを使用して複数のファイルを追加することもできます。
{{% /alert %}}  
  {{% /tab %}}
  {{% tab header="削除" %}}
`remove` メソッドを使用して既存のアーティファクト バージョンからファイルを削除します。

```python
draft_artifact.remove("file_to_remove.txt")
```

{{% alert %}}
ディレクトリ パスを指定することで `remove` メソッドを使用して複数のファイルを削除することもできます。
{{% /alert %}}  
  {{% /tab %}}
  {{% tab header="変更" %}}
下記のように、ドラフトから古い内容を削除し、新しい内容を再度追加することで、内容を変更または置き換えます。

```python
draft_artifact.remove("modified_file.txt")
draft_artifact.add_file("modified_file.txt")
```  
  {{% /tab %}}
{{< /tabpane >}}

4. 最後に、変更をログまたは保存します。以下のタブは、W&B Run の中および外で変更を保存する方法を示しているので、それぞれのユースケースに適したタブを選択してください。

{{< tabpane text=true >}}
  {{% tab header="Run 内" %}}
```python
run.log_artifact(draft_artifact)
```

  {{% /tab %}}
  {{% tab header="Run 外" %}}
```python
draft_artifact.save()
```  
  {{% /tab %}}
{{< /tabpane >}}

上記のコード例をすべてまとめると次のようになります。

{{< tabpane text=true >}}
  {{% tab header="Run 内" %}}
```python
with wandb.init(job_type="modify dataset") as run:
    saved_artifact = run.use_artifact(
        "my_artifact:latest"
    )  # アーティファクトをフェッチしてそれを Run に入力します
    draft_artifact = saved_artifact.new_draft()  # ドラフト バージョンを作成します

    # ドラフト バージョンのファイルのサブセットを変更します
    draft_artifact.add_file("file_to_add.txt")
    draft_artifact.remove("dir_to_remove/")
    run.log_artifact(
        artifact
    )  # 変更をログに記録して新しいバージョンを作成し、それを Run の出力としてマークします
```  
  {{% /tab %}}
  {{% tab header="Run 外" %}}
```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")  # アーティファクトをロードします
draft_artifact = saved_artifact.new_draft()  # ドラフト バージョンを作成します

# ドラフト バージョンのファイルのサブセットを変更します
draft_artifact.remove("deleted_file.txt")
draft_artifact.add_file("modified_file.txt")
draft_artifact.save()  # ドラフトへの変更をコミットします
```  
  {{% /tab %}}
{{< /tabpane >}}