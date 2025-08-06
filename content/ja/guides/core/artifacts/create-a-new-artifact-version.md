---
title: アーティファクトのバージョンを作成する
description: 単一の run または分散プロセスから新しいアーティファクトバージョンを作成します。
menu:
  default:
    identifier: create-a-new-artifact-version
    parent: artifacts
weight: 6
---

単一の [run]({{< relref "/guides/models/track/runs/" >}}) で新しい artifact バージョンを作成することも、分散環境の複数の run で協力して artifact バージョンを作成することもできます。また、[インクリメンタル artifact]({{< relref "#create-a-new-artifact-version-from-an-existing-version" >}}) と呼ばれる、既存バージョンから新しい artifact バージョンを作成することも可能です。

{{% alert %}}
元の artifact がかなり大きい場合に、一部のファイルだけ変更したい場合はインクリメンタル artifact の作成をおすすめします。
{{% /alert %}}

## 新しい artifact バージョンをゼロから作成する
新たな artifact バージョンの作成方法は 2 つあります。「単一の run から作成」と「分散 run から作成」です。

* **単一 run**: 1 つの run ですべてのデータを新バージョン用に提供します。必要なデータを run で一からすべて作成する場合に最も一般的な方法です（例: モデルの保存や、分析用にテーブルにモデルの予測結果を保存する場合）。
* **分散 run**: 複数の run を組み合わせて新バージョンのすべてのデータを集約します。複数の run が並行してデータを生成する分散ジョブに最適です（例: 分散評価による予測データの生成）。

`wandb.Artifact` API に Project 内でまだ存在しない名前を指定して artifact を作成した場合、W&B は新規 artifact を作成し、その artifact に `v0` エイリアスを割り当てます。同じ artifact に再びログした場合、内容のハッシュ値をチェックし、artifact 内容が変更されていれば新しいバージョン `v1` として保存されます。

Project 内の既存 artifact と同じ名前・タイプを `wandb.Artifact` API に指定した場合は、対応する artifact を取得します。この場合、バージョン番号は 1 より大きくなります。

{{< img src="/images/artifacts/single_distributed_artifacts.png" alt="Artifact workflow comparison" >}}

### 単一 run
1 つの run ですべてのファイルを含む新しい Artifact のバージョンをログします。このパターンは、単一の run で artifact のすべてのファイルが用意できる場合に当てはまります。

ユースケースに応じて、run の内側（Inside a run）、run の外側（Outside of a run）どちらで artifact バージョンを作成するか、以下のタブから選択してください。

{{< tabpane text=true >}}
  {{% tab header="Inside a run" %}}
W&B run 内で artifact バージョンを作成する場合：

1. `wandb.init` で run を作成します。
2. `wandb.Artifact` で新しい artifact を作成、または既存 artifact を取得します。
3. `.add_file` で artifact にファイルを追加します。
4. `.log_artifact` で artifact を run にログします。

```python 
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # ファイルやアセットを artifact に追加
    # `.add`, `.add_file`, `.add_dir`, `.add_reference` などを利用
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```  
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
W&B run の外側で artifact バージョンを作成する場合：

1. `wanb.Artifact` で新しい artifact を作成、または既存 artifact を取得します。
2. `.add_file` で artifact にファイルを追加します。
3. `.save` で artifact を保存します。

```python 
artifact = wandb.Artifact("artifact_name", "artifact_type")
# ファイルやアセットを artifact に追加
# `.add`, `.add_file`, `.add_dir`, `.add_reference` などを利用
artifact.add_file("image1.png")
artifact.save()
```    
  {{% /tab %}}
{{< /tabpane  >}}




### 分散 run

複数の run で共同して 1 バージョンの artifact を作成できます。これは前述の単一 run 方式とは異なり、複数 run からデータを寄せ集めて新バージョンを作成します。

{{% alert %}}
1. コレクション内の各 run が同じユニーク ID（`distributed_id`）を認識している必要があります。デフォルトでは、`wandb.init(group=GROUP)` でセットした run の `group` が `distributed_id` として使用されます。
2. 最後に「コミット（状態を確定）」する run が必要です。この run で最終的な artifact バージョンが確定されます。
3. 共同 artifact へ追加は `upsert_artifact` で、コミットは `finish_artifact` を利用してください。
{{% /alert %}}

以下の例では、異なる run（**Run 1**, **Run 2**, **Run 3**）が、それぞれ異なる画像ファイルを同じ artifact に `upsert_artifact` で追加しています。

#### Run 1:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # ファイルやアセットを artifact に追加
    # `.add`, `.add_file`, `.add_dir`, `.add_reference` などを利用
    artifact.add_file("image1.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 2:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # ファイルやアセットを artifact に追加
    # `.add`, `.add_file`, `.add_dir`, `.add_reference` などを利用
    artifact.add_file("image2.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 3

Run 1 と Run 2 が完了してから実行してください。`finish_artifact` を呼ぶ run で artifact にファイルを追加しても良いですが、必須ではありません。

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # ファイルやアセットを artifact に追加
    # `.add`, `.add_file`, `.add_dir`, `.add_reference` などを利用
    artifact.add_file("image3.png")
    run.finish_artifact(artifact, distributed_id="my_dist_artifact")
```




## 既存バージョンから新しい artifact バージョンを作成する

前の artifact バージョンから一部ファイルだけを追加・修正・削除し、変更のないファイルを再アップロードすることなく、新しい artifact バージョンを作成できます。こうした変更で作られる新しい artifact バージョンを *インクリメンタル artifact* と呼びます。

{{< img src="/images/artifacts/incremental_artifacts.png" alt="Incremental artifact versioning" >}}

インクリメンタルな変更のタイプごとに、いくつかの活用例を挙げます：

- add: 新しいバッチを収集した後、データセットに新しいサブセットのファイルを定期的に追加する場合。
- remove: 重複ファイルを発見し、artifact から削除したい場合。
- update: 一部のファイルのアノテーションを修正し、正しいファイルに置き換えたい場合。

一から artifact を作り直すことも可能ですが、その場合 artifact の全ファイルがローカルディスク上に必要です。インクリメンタル変更では、以前の artifact バージョンの未変更ファイルには手を加えず、追加・削除・修正したいファイルのみを対象にできます。

{{% alert %}}
インクリメンタル artifact は、単一の run 内でも、複数 run（分散モード）でも作成できます。
{{% /alert %}}

以下の手順に従って artifact をインクリメンタルに変更しましょう：

1. インクリメンタル変更を加えたい artifact バージョンを取得します：

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

2. 下書き（ドラフト）を作成します：

```python
draft_artifact = saved_artifact.new_draft()
```

3. 次のバージョンに反映したいインクリメンタルな変更を行います。ファイルの追加・削除・修正のいずれにも対応できます。

変更ごとの例は下記のタブから選択してください：

{{< tabpane text=true >}}
  {{% tab header="Add" %}}
既存 artifact バージョンにファイルを追加するには `add_file` メソッドを使います：

```python
draft_artifact.add_file("file_to_add.txt")
```

{{% alert %}}
`add_dir` メソッドでディレクトリごと複数ファイル追加も可能です。
{{% /alert %}}  
  {{% /tab %}}
  {{% tab header="Remove" %}}
既存 artifact バージョンからファイルを削除するには `remove` メソッドを使います：

```python
draft_artifact.remove("file_to_remove.txt")
```

{{% alert %}}
ディレクトリパスを `remove` メソッドに渡すことで、複数ファイルまとめて削除できます。
{{% /alert %}}  
  {{% /tab %}}
  {{% tab header="Modify" %}}
内容の修正や置き換えは、古い内容をドラフトから `remove` で削除し、新しい内容を `add_file` で追加します：

```python
draft_artifact.remove("modified_file.txt")
draft_artifact.add_file("modified_file.txt")
```  
  {{% /tab %}}
{{< /tabpane >}}

4. 最後に、変更をログまたは保存します。W&B run 内外、それぞれの場合の保存方法を選択してください：

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

全部まとめると、これらの手順をコードとして記述すると次の通りです：

{{< tabpane text=true >}}
  {{% tab header="Inside a run" %}}
```python
with wandb.init(job_type="modify dataset") as run:
    saved_artifact = run.use_artifact(
        "my_artifact:latest"
    )  # artifact を取得し run に入力する
    draft_artifact = saved_artifact.new_draft()  # 下書きバージョンを作成

    # 下書きバージョンの一部ファイルを変更
    draft_artifact.add_file("file_to_add.txt")
    draft_artifact.remove("dir_to_remove/")
    run.log_artifact(
        draft_artifact
    )  # 変更をログし新しいバージョンを作成、run の output としてマーク
```  
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")  # artifact を取得
draft_artifact = saved_artifact.new_draft()  # 下書きバージョンを作成

# 下書きバージョンの一部ファイルを変更
draft_artifact.remove("deleted_file.txt")
draft_artifact.add_file("modified_file.txt")
draft_artifact.save()  # 変更をコミット
```  
  {{% /tab %}}
{{< /tabpane >}}