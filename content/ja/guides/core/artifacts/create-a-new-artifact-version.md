---
title: Create an artifact version
description: 単一の run 、または分散された プロセス から、新しい アーティファクト バージョン を作成します。
menu:
  default:
    identifier: ja-guides-core-artifacts-create-a-new-artifact-version
    parent: artifacts
weight: 6
---

単一の [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) で、または分散された run と共同で、新しい Artifact の バージョンを作成します。オプションで、[インクリメンタル Artifact]({{< relref path="#create-a-new-artifact-version-from-an-existing-version" lang="ja" >}}) と呼ばれる以前のバージョンから新しい Artifact の バージョンを作成できます。

{{% alert %}}
元の Artifact のサイズが大幅に大きい場合に、Artifact 内のファイルのサブセットに変更を適用する必要がある場合は、インクリメンタル Artifact を作成することをお勧めします。
{{% /alert %}}

## Artifact の新しいバージョンをゼロから作成する

Artifact の新しいバージョンを作成する方法は、単一の run から作成する方法と、分散された run から作成する方法の2つがあります。それらは次のように定義されます。

*   **単一の run**: 単一の run は、新しいバージョンのすべてのデータを提供します。これは最も一般的なケースであり、run が必要なデータを完全に再作成する場合に最適です。例：分析のために、保存された model または model の 予測 をテーブルに出力するなど。
*   **分散された run**: run のセットが、共同で新しいバージョンのすべてのデータを提供します。これは、分散ジョブで複数の run がデータを生成する場合に最適であり、多くの場合並行して行われます。例：分散方式で model を評価し、 予測 を出力するなど。

W&B は、プロジェクト に存在しない名前を `wandb.Artifact` API に渡すと、新しい Artifact を作成し、`v0` エイリアス を割り当てます。同じ Artifact に再度 ログ を記録すると、W&B はコンテンツのチェックサムを作成します。Artifact が変更された場合、W&B は新しいバージョン `v1` を保存します。

`wandb.Artifact` API に名前と Artifact タイプを渡して、プロジェクト に既存の Artifact と一致する場合、W&B は既存の Artifact を取得します。取得された Artifact のバージョンは1より大きくなります。

{{< img src="/images/artifacts/single_distributed_artifacts.png" alt="" >}}

### 単一の run

Artifact 内のすべてのファイルを生成する単一の run で、Artifact の新しいバージョンを ログ に記録します。このケースは、単一の run が Artifact 内のすべてのファイルを生成する場合に発生します。

ユースケース に基づいて、以下のタブのいずれかを選択して、run の内外で新しい Artifact バージョンを作成します。

{{< tabpane text=true >}}
  {{% tab header="Run の内部" %}}
W&B の run 内に Artifact バージョンを作成します。

1.  `wandb.init` で run を作成します。（1行目）
2.  `wandb.Artifact` で新しい Artifact を作成するか、既存の Artifact を取得します。（2行目）
3.  `.add_file` で Artifact にファイルを追加します。（9行目）
4.  `.log_artifact` で Artifact を run に ログ します。（10行目）

```python showLineNumbers
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # `.add`、`.add_file`、`.add_dir`、および `.add_reference` を使用して、Artifact にファイルとアセットを追加します
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```
  {{% /tab %}}
  {{% tab header="Run の外部" %}}
W&B の run の外部に Artifact バージョンを作成します。

1.  `wanb.Artifact` で新しい Artifact を作成するか、既存の Artifact を取得します。（1行目）
2.  `.add_file` で Artifact にファイルを追加します。（4行目）
3.  `.save` で Artifact を保存します。（5行目）

```python showLineNumbers
artifact = wandb.Artifact("artifact_name", "artifact_type")
# `.add`、`.add_file`、`.add_dir`、および `.add_reference` を使用して、Artifact にファイルとアセットを追加します
artifact.add_file("image1.png")
artifact.save()
```
  {{% /tab %}}
{{< /tabpane >}}

### 分散された run

コミットする前に、run のコレクションがバージョンで共同作業できるようにします。これは、単一の run が新しいバージョンのすべてのデータを提供する上記の単一 run モードとは対照的です。

{{% alert %}}
1.  コレクション内の各 run は、同じバージョンで共同作業するために、同じ一意のID（`distributed_id` と呼ばれます）を認識している必要があります。デフォルトでは、存在する場合、W&B は `wandb.init(group=GROUP)` で設定された run の `group` を `distributed_id` として使用します。
2.  バージョンを「コミット」し、その状態を永続的にロックする最終 run が必要です。
3.  共同 Artifact に追加するには `upsert_artifact` を使用し、コミットを完了するには `finish_artifact` を使用します。
{{% /alert %}}

次の例を考えてみましょう。異なる run （以下では **Run 1**、**Run 2**、**Run 3** としてラベル付けされています）は、`upsert_artifact` を使用して、異なるイメージファイルを同じ Artifact に追加します。

#### Run 1:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`、`.add_file`、`.add_dir`、および `.add_reference` を使用して、Artifact にファイルとアセットを追加します
    artifact.add_file("image1.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 2:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`、`.add_file`、`.add_dir`、および `.add_reference` を使用して、Artifact にファイルとアセットを追加します
    artifact.add_file("image2.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 3

Run 1 と Run 2 が完了した後で実行する必要があります。`finish_artifact` を呼び出す Run は、Artifact にファイルを含めることができますが、含める必要はありません。

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # Artifact にファイルとアセットを追加します
    # `.add`、`.add_file`、`.add_dir`、および `.add_reference`
    artifact.add_file("image3.png")
    run.finish_artifact(artifact, distributed_id="my_dist_artifact")
```

## 既存のバージョンから新しい Artifact バージョンを作成する

変更されていないファイルを再インデックスする必要なく、以前の Artifact バージョンからファイルの サブセット を追加、変更、または削除します。以前の Artifact バージョンからファイルの サブセット を追加、変更、または削除すると、*インクリメンタル Artifact* と呼ばれる新しい Artifact バージョンが作成されます。

{{< img src="/images/artifacts/incremental_artifacts.png" alt="" >}}

考えられるインクリメンタルな変更のタイプごとのシナリオを以下に示します。

*   add: 新しいバッチを収集した後、データセット に新しいファイルの サブセット を定期的に追加します。
*   remove: 複数の重複ファイルを発見し、Artifact から削除したいと考えています。
*   update: ファイルの サブセット のアノテーションを修正し、古いファイルを正しいファイルに置き換えたいと考えています。

Artifact をゼロから作成して、インクリメンタル Artifact と同じ機能を実行できます。ただし、Artifact をゼロから作成する場合、ローカルディスクに Artifact のすべてのコンテンツが必要です。インクリメンタルな変更を行う場合、以前の Artifact バージョンのファイルを変更せずに、単一のファイルを追加、削除、または変更できます。

{{% alert %}}
インクリメンタル Artifact は、単一の run 内で、または run のセット（分散モード）で作成できます。
{{% /alert %}}

以下の手順に従って、Artifact をインクリメンタルに変更します。

1.  インクリメンタルな変更を実行する Artifact バージョンを取得します。

{{< tabpane text=true >}}
{{% tab header="Run の内部" %}}

```python
saved_artifact = run.use_artifact("my_artifact:latest")
```

{{% /tab %}}
{{% tab header="Run の外部" %}}

```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")
```

{{% /tab %}}
{{< /tabpane >}}

2.  以下を使用して下書きを作成します。

```python
draft_artifact = saved_artifact.new_draft()
```

3.  次のバージョンで確認したいインクリメンタルな変更を実行します。既存のエントリを追加、削除、または変更できます。

これらの変更を実行する方法の例については、以下のタブのいずれかを選択してください。

{{< tabpane text=true >}}
  {{% tab header="追加" %}}
`add_file` メソッドを使用して、既存の Artifact バージョンにファイルを追加します。

```python
draft_artifact.add_file("file_to_add.txt")
```

{{% alert %}}
`add_dir` メソッドを使用してディレクトリーを追加することで、複数のファイルを追加することもできます。
{{% /alert %}}
  {{% /tab %}}
  {{% tab header="削除" %}}
`remove` メソッドを使用して、既存の Artifact バージョンからファイルを削除します。

```python
draft_artifact.remove("file_to_remove.txt")
```

{{% alert %}}
ディレクトリーパスを渡すことで、`remove` メソッドを使用して複数のファイルを削除することもできます。
{{% /alert %}}
  {{% /tab %}}
  {{% tab header="変更" %}}
下書きから古いコンテンツを削除し、新しいコンテンツを再度追加することで、コンテンツを変更または置換します。

```python
draft_artifact.remove("modified_file.txt")
draft_artifact.add_file("modified_file.txt")
```
  {{% /tab %}}
{{< /tabpane >}}

4.  最後に、変更を ログ または保存します。以下のタブは、W&B の run の内外で変更を保存する方法を示しています。ユースケース に適したタブを選択してください。

{{< tabpane text=true >}}
  {{% tab header="Run の内部" %}}
```python
run.log_artifact(draft_artifact)
```

  {{% /tab %}}
  {{% tab header="Run の外部" %}}
```python
draft_artifact.save()
```
  {{% /tab %}}
{{< /tabpane >}}

すべてをまとめると、上記のコード例は次のようになります。

{{< tabpane text=true >}}
  {{% tab header="Run の内部" %}}
```python
with wandb.init(job_type="modify dataset") as run:
    saved_artifact = run.use_artifact(
        "my_artifact:latest"
    )  # Artifact をフェッチし、run に入力します
    draft_artifact = saved_artifact.new_draft()  # 下書きバージョンを作成します

    # 下書きバージョンのファイルの サブセット を変更します
    draft_artifact.add_file("file_to_add.txt")
    draft_artifact.remove("dir_to_remove/")
    run.log_artifact(
        artifact
    )  # 変更を ログ して新しいバージョンを作成し、run への出力としてマークします
```
  {{% /tab %}}
  {{% tab header="Run の外部" %}}
```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")  # Artifact をロードします
draft_artifact = saved_artifact.new_draft()  # 下書きバージョンを作成します

# 下書きバージョンのファイルの サブセット を変更します
draft_artifact.remove("deleted_file.txt")
draft_artifact.add_file("modified_file.txt")
draft_artifact.save()  # 下書きに変更をコミットします
```
  {{% /tab %}}
{{< /tabpane >}}
