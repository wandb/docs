---
title: Artifacts のバージョンを作成する
description: 単一の run または分散プロセスから、新しい Artifacts のバージョンを作成します。
menu:
  default:
    identifier: ja-guides-core-artifacts-create-a-new-artifact-version
    parent: artifacts
weight: 6
---

単一の [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) で、または分散された run と協調して、新しい Artifact バージョンを作成します。必要に応じて、以前のバージョンから新しい Artifact バージョンを作成できます。これは [Incremental Artifact]({{< relref path="#create-a-new-artifact-version-from-an-existing-version" lang="ja" >}}) と呼ばれます。
{{% alert %}}
オリジナルの Artifact のサイズが著しく大きく、Artifact 内のファイルのサブセットにのみ変更を適用したい場合は、Incremental Artifact の作成をお勧めします。
{{% /alert %}}
## 新しい Artifact バージョンをゼロから作成する
新しい Artifact バージョンを作成する方法は、単一の run と分散された run の 2 通りがあります。定義は次のとおりです。
*   **Single run**: 単一の run が新しいバージョンに必要なすべてのデータを提供します。これは最も一般的なケースであり、run が必要なデータを完全に再作成する場合に最適です。例: 保存されたモデルやモデルの予測を分析用のテーブルとして出力する場合。
*   **Distributed runs**: 複数の run が共同で新しいバージョンに必要なすべてのデータを提供します。これは、複数の run がしばしば並行してデータを生成する分散ジョブに最適です。例: モデルを分散方式で評価し、予測を出力する場合。
`wandb.Artifact` API に、お使いの project に存在しない名前を渡すと、W&B は新しい Artifact を作成し、`v0` エイリアスを割り当てます。同じ Artifact に再度ログを記録すると、W&B はそのコンテンツのチェックサムを作成します。Artifact が変更された場合、W&B は新しいバージョン `v1` を保存します。
`wandb.Artifact` API に、お使いの project に存在する Artifact と一致する名前と Artifact タイプを渡すと、W&B は既存の Artifact を取得します。取得された Artifact は、1 より大きいバージョンを持ちます.
{{< img src="/images/artifacts/single_distributed_artifacts.png" alt="Artifact ワークフローの比較" >}}
### Single run
Artifact 内のすべてのファイルを生成する単一の run で、Artifact の新しいバージョンをログに記録します。このケースは、単一の run が Artifact 内のすべてのファイルを生成する場合に発生します。
お使いのユースケースに基づいて、以下のタブのいずれかを選択して、run 内または run 外で新しい Artifact バージョンを作成してください。
{{< tabpane text=true >}}
  {{% tab header="run 内" %}}
W&B の run 内で Artifact バージョンを作成します:
1.  `wandb.init` で run を作成します。
2.  `wandb.Artifact` で新しい Artifact を作成するか、既存のものを取得します。
3.  `.add_file` で Artifact にファイルを追加します。
4.  `.log_artifact` で Artifact を run にログとして記録します。
```python 
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # `.add`, `.add_file`, `.add_dir`, and `.add_reference` を使用して
    # Artifact にファイルとアセットを追加します
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```  
  {{% /tab %}}
  {{% tab header="run 外" %}}
W&B の run 外で Artifact バージョンを作成します:
1.  `wandb.Artifact` で新しい Artifact を作成するか、既存のものを取得します。
2.  `.add_file` で Artifact にファイルを追加します。
3.  `.save` で Artifact を保存します。
```python 
artifact = wandb.Artifact("artifact_name", "artifact_type")
# `.add`, `.add_file`, `.add_dir`, and `.add_reference` を使用して
# Artifact にファイルとアセットを追加します
artifact.add_file("image1.png")
artifact.save()
```    
  {{% /tab %}}
{{< /tabpane  >}}
### Distributed runs
複数の run が共同でバージョンを作成し、コミット前に協力できるようにします。これは、1 つの run が新しいバージョンに必要なすべてのデータを提供する、上記の単一 run モードとは対照的です。
{{% alert %}}
1.  コレクション内の各 run は、同じバージョンで共同作業するために、同じユニーク ID (`distributed_id` と呼ばれます) を認識している必要があります。デフォルトでは、存在する場合、W&B は `wandb.init(group=GROUP)` で設定された run の `group` を `distributed_id` として使用します。
2.  バージョンを「コミット」し、その状態を永続的にロックする最終 run が存在する必要があります。
3.  共同 Artifact に追加するには `upsert_artifact` を使用し、コミットを確定するには `finish_artifact` を使用します。
{{% /alert %}}
次の例を考えてみましょう。異なる run (**Run 1**、**Run 2**、**Run 3** と下記に示されています) が、`upsert_artifact` を使用して、それぞれ異なる画像ファイルを同じ Artifact に追加します。
#### Run 1:
```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference` を使用して
    # Artifact にファイルとアセットを追加します
    artifact.add_file("image1.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```
#### Run 2:
```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference` を使用して
    # Artifact にファイルとアセットを追加します
    artifact.add_file("image2.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```
#### Run 3
Run 1 と Run 2 が完了した後で実行する必要があります。`finish_artifact` を呼び出す run は、Artifact にファイルを含めることができますが、必須ではありません。
```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # Artifact にファイルとアセットを追加します
    # `.add`, `.add_file`, `.add_dir`, および `.add_reference` を使用します
    artifact.add_file("image3.png")
    run.finish_artifact(artifact, distributed_id="my_dist_artifact")
```
## 既存のバージョンから新しい Artifact バージョンを作成する
変更されていないファイルを再インデックスすることなく、以前の Artifact バージョンからファイルのサブセットを追加、変更、または削除します。以前の Artifact バージョンからファイルのサブセットを追加、変更、または削除することで、*Incremental Artifact* と呼ばれる新しい Artifact バージョンが作成されます。
{{< img src="/images/artifacts/incremental_artifacts.png" alt="Incremental Artifact のバージョン管理" >}}
インクリメンタル変更の各タイプで考えられるシナリオをいくつか紹介します。
-   追加: 新しいバッチを収集した後、データセットにファイルの新しいサブセットを定期的に追加します。
-   削除: いくつかの重複ファイルを発見し、それらを Artifact から削除したい場合。
-   更新: ファイルのサブセットの注釈を修正し、古いファイルを正しいものに置き換えたい場合。
Incremental Artifact と同じ機能を実行するために、Artifact をゼロから作成することもできます。ただし、Artifact をゼロから作成する場合、Artifact のすべてのコンテンツをローカルディスクに保存しておく必要があります。インクリメンタルな変更を行う場合、以前の Artifact バージョンのファイルを変更することなく、単一のファイルを追加、削除、または変更できます。
{{% alert %}}
単一の run 内、または複数の run (分散モード) を使用して、Incremental Artifact を作成できます。
{{% /alert %}}
Artifact をインクリメンタルに変更するには、以下の手順に従ってください。
1.  インクリメンタルな変更を加えたい Artifact バージョンを取得します。
{{< tabpane text=true >}}
{{% tab header="run 内" %}}
```python
saved_artifact = run.use_artifact("my_artifact:latest")
```
{{% /tab %}}
{{% tab header="run 外" %}}
```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")
```
{{% /tab %}}
{{< /tabpane >}}
2.  次のようにドラフトを作成します。
```python
draft_artifact = saved_artifact.new_draft()
```
3.  次のバージョンで見たいインクリメンタルな変更を実行します。既存のエントリーの追加、削除、または変更が可能です。
これらの変更をそれぞれ実行する方法の例については、以下のタブのいずれかを選択してください。
{{< tabpane text=true >}}
  {{% tab header="追加" %}}
`add_file` メソッドを使用して、既存の Artifact バージョンにファイルを追加します。
```python
draft_artifact.add_file("file_to_add.txt")
```
{{% alert %}}
`add_dir` メソッドでディレクトリを追加することにより、複数のファイルを追加することもできます。
{{% /alert %}}  
  {{% /tab %}}
  {{% tab header="削除" %}}
`remove` メソッドを使用して、既存の Artifact バージョンからファイルを削除します。
```python
draft_artifact.remove("file_to_remove.txt")
```
{{% alert %}}
ディレクトリ パスを渡すことで、`remove` メソッドを使用して複数のファイルを削除することもできます。
{{% /alert %}}  
  {{% /tab %}}
  {{% tab header="変更" %}}
ドラフトから古いコンテンツを削除し、新しいコンテンツを再度追加することで、コンテンツを変更または置き換えます。
```python
draft_artifact.remove("modified_file.txt")
draft_artifact.add_file("modified_file.txt")
```  
  {{% /tab %}}
{{< /tabpane >}}
4.  最後に、変更をログとして記録または保存します。以下のタブでは、W&B run の内部と外部で変更を保存する方法を示します。お使いのユースケースに適したタブを選択してください。
{{< tabpane text=true >}}
  {{% tab header="run 内" %}}
```python
run.log_artifact(draft_artifact)
```
  {{% /tab %}}
  {{% tab header="run 外" %}}
```python
draft_artifact.save()
```  
  {{% /tab %}}
{{< /tabpane >}}
まとめると、上記のコード例は次のようになります。
{{< tabpane text=true >}}
  {{% tab header="run 内" %}}
```python
with wandb.init(job_type="modify dataset") as run:
    saved_artifact = run.use_artifact(
        "my_artifact:latest"
    )  # Artifact を取得し、run に取り込みます
    draft_artifact = saved_artifact.new_draft()  # ドラフトバージョンを作成します

    # ドラフトバージョン内のファイルのサブセットを変更します
    draft_artifact.add_file("file_to_add.txt")
    draft_artifact.remove("dir_to_remove/")
    run.log_artifact(
        draft_artifact
    )  # 変更をログに記録して新しいバージョンを作成し、それを run の出力としてマークします
```  
  {{% /tab %}}
  {{% tab header="run 外" %}}
```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")  # Artifact をロードします
draft_artifact = saved_artifact.new_draft()  # ドラフトバージョンを作成します

# ドラフトバージョン内のファイルのサブセットを変更します
draft_artifact.remove("deleted_file.txt")
draft_artifact.add_file("modified_file.txt")
draft_artifact.save()  # ドラフトへの変更をコミットします
```  
  {{% /tab %}}
{{< /tabpane >}}