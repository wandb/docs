---
title: アーティファクトのバージョンを作成する
description: 単一の run または分散プロセスから新しいアーティファクトバージョンを作成します。
menu:
  default:
    identifier: ja-guides-core-artifacts-create-a-new-artifact-version
    parent: artifacts
weight: 6
---

単一の [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) で、または分散 run で協力しながら新しい artifact バージョンを作成できます。また、既存のバージョンから新しい artifact バージョンを作成する（[インクリメンタル artifact]({{< relref path="#create-a-new-artifact-version-from-an-existing-version" lang="ja" >}}) と呼ばれる）ことも可能です。

{{% alert %}}
アーティファクト内の一部ファイルのみ変更が必要で、元のアーティファクトサイズが大きい場合には、インクリメンタル artifact の作成を推奨します。
{{% /alert %}}

## 新しい artifact バージョンをゼロから作成する
新しい artifact バージョンの作成方法は2つあります：単一の run から作成、分散 run から作成です。それぞれ以下のように定義されます。

* **Single run**: 1つの run が新しいバージョンのすべてのデータを提供します。最も一般的なケースで、run が必要なデータをすべて再生成する場合に最適です。例：保存済みのモデルや、分析用にテーブルで出力するモデル予測など。
* **Distributed runs**: 複数の run の集合が、新しいバージョンのすべてのデータを共同で提供します。複数の run が並列でデータを生成する分散ジョブに適しています。例：分散方式でモデルを評価し、予測結果を出力する場合など。

プロジェクト内に存在しない名前を `wandb.Artifact` API に渡すと、W&B は新しい artifact を作成し、`v0` というエイリアスを割り当てます。同じ artifact に再度ログする際、W&B は内容にチェックサムをかけ、内容が変わっていれば新しいバージョン `v1` が保存されます。

プロジェクト内に存在する artifact の名前と artifact タイプを `wandb.Artifact` API に渡すと、既存の artifact が取得され、取得した artifact のバージョンは 1 より大きくなります。

{{< img src="/images/artifacts/single_distributed_artifacts.png" alt="Artifact workflow comparison" >}}

### Single run
1つの run がアーティファクト内のすべてのファイルを生成する場合に、Artifact の新バージョンをログします。

ご自身のユースケースに合わせて、以下のタブから run 内または run 外で新しい artifact バージョンを作成する方法を選択してください。

{{< tabpane text=true >}}
  {{% tab header="Inside a run" %}}
W&B の run 内で artifact バージョンを作成する方法:

1. `wandb.init` で run を作成します。
2. `wandb.Artifact` で新しい artifact を作成、または既存の artifact を取得します。
3. `.add_file` で artifact にファイルを追加します。
4. `.log_artifact` で artifact を run にログします。

```python 
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # ファイルやアセットを artifact に追加
    # `.add`, `.add_file`, `.add_dir`, `.add_reference` を利用
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```  
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
W&B の run 外で artifact バージョンを作成する方法:

1. `wandb.Artifact` で新しい artifact を作成または既存の artifact を取得します。
2. `.add_file` で artifact にファイルを追加します。
3. `.save` で artifact を保存します。

```python 
artifact = wandb.Artifact("artifact_name", "artifact_type")
# ファイルやアセットを artifact に追加
# `.add`, `.add_file`, `.add_dir`, `.add_reference` を利用
artifact.add_file("image1.png")
artifact.save()
```    
  {{% /tab %}}
{{< /tabpane  >}}

### Distributed runs

複数の run で協力して artifact バージョンを作成し、最後に commit することができます。これは、1つの run で全てを処理する Single run モードとは対照的です。

{{% alert %}}
1. コレクション内の各 run は、同一の一意な ID（`distributed_id` と呼ばれる）を認識している必要があります。デフォルトでは、存在すれば、`wandb.init(group=GROUP)` でセットした run の `group` が `distributed_id` として使われます。
2. 最後に "commit" を行う run が必要で、この run によって artifact の状態が確定します。
3. 共同 artifact への追加には `upsert_artifact`、commit の確定には `finish_artifact` を使用します。
{{% /alert %}}

以下は例です。異なる run（**Run 1**, **Run 2**, **Run 3** とラベル付け）のそれぞれが、`upsert_artifact` を使って同じ artifact に異なる画像ファイルを追加しています。

#### Run 1:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # ファイルやアセットを artifact に追加
    # `.add`, `.add_file`, `.add_dir`, `.add_reference` を利用
    artifact.add_file("image1.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 2:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # ファイルやアセットを artifact に追加
    # `.add`, `.add_file`, `.add_dir`, `.add_reference` を利用
    artifact.add_file("image2.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 3

Run 1 と Run 2 の完了後に実行する必要があります。`finish_artifact` を呼ぶ Run もファイルを artifact に含められますが、含める必要はありません。

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # ファイルやアセットを artifact に追加
    # `.add`, `.add_file`, `.add_dir`, `.add_reference` を利用
    artifact.add_file("image3.png")
    run.finish_artifact(artifact, distributed_id="my_dist_artifact")
```

## 既存バージョンから新しい artifact バージョンを作成する

前の artifact バージョンに含まれるファイルの一部だけを追加・変更・削除して、新しい artifact バージョンを作成できます（このように生成されたものを *インクリメンタル artifact* と呼びます）。変更されていないファイルの再インデックスは不要です。

{{< img src="/images/artifacts/incremental_artifacts.png" alt="Incremental artifact versioning" >}}

各種インクリメンタルな変更のシナリオ例：

- add: 新しいバッチでデータを収集した後、データセットに新しいサブセットを定期的に追加する
- remove: 重複ファイルを発見し、artifact から削除したい
- update: ファイルのアノテーションに修正を施し、古いファイルを差し替えたい

インクリメンタル artifact と同等のことを、artifact をゼロから作成しても実現できます。しかしこの場合、artifact に含めたい全コンテンツをローカルディスク上に用意する必要があります。一方、インクリメンタルな変更であれば、前バージョンのファイルには触れず、1つのファイル単位で追加・削除・更新ができます。

{{% alert %}}
インクリメンタル artifact は、単一 run 内でも、複数 run（分散モード）でも作成できます。
{{% /alert %}}

以下の手順で artifact のインクリメンタルな変更を行います:

1. インクリメンタルな変更を行いたい artifact バージョンを取得します:

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

2. 下書きを作成します:

```python
draft_artifact = saved_artifact.new_draft()
```

3. インクリメンタルに反映したい変更（ファイルの追加・削除・修正）を行います。

各変更例は以下のタブから選択してください:

{{< tabpane text=true >}}
  {{% tab header="Add" %}}
`add_file` メソッドで既存の artifact バージョンにファイルを追加します:

```python
draft_artifact.add_file("file_to_add.txt")
```

{{% alert %}}
`add_dir` メソッドを使うことで、ディレクトリ単位で複数ファイルも追加できます。
{{% /alert %}}  
  {{% /tab %}}
  {{% tab header="Remove" %}}
`remove` メソッドで既存 artifact バージョンからファイルを削除します:

```python
draft_artifact.remove("file_to_remove.txt")
```

{{% alert %}}
ディレクトリパスを渡すことで、`remove` メソッドで複数ファイルも削除できます。
{{% /alert %}}  
  {{% /tab %}}
  {{% tab header="Modify" %}}
ドラフトから古い内容を削除し、新しい内容を追加することで、内容を変更または差し替えます:

```python
draft_artifact.remove("modified_file.txt")
draft_artifact.add_file("modified_file.txt")
```  
  {{% /tab %}}
{{< /tabpane >}}

4. 最後に、変更をログまたは保存します。run 内外で変更を保存する方法は、下記タブからご確認ください。

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

これらすべてをまとめると、上のコード例は次のようになります:

{{< tabpane text=true >}}
  {{% tab header="Inside a run" %}}
```python
with wandb.init(job_type="modify dataset") as run:
    saved_artifact = run.use_artifact(
        "my_artifact:latest"
    )  # artifact を取得し、run に入力
    draft_artifact = saved_artifact.new_draft()  # ドラフトバージョンを作成

    # ドラフトバージョンで一部ファイルを変更
    draft_artifact.add_file("file_to_add.txt")
    draft_artifact.remove("dir_to_remove/")
    run.log_artifact(
        draft_artifact
    )  # 変更をログして新しいバージョンを作成し、run の出力としてマーク
```  
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")  # artifact を読み込み
draft_artifact = saved_artifact.new_draft()  # ドラフトバージョンを作成

# ドラフトバージョンで一部ファイルを変更
draft_artifact.remove("deleted_file.txt")
draft_artifact.add_file("modified_file.txt")
draft_artifact.save()  # ドラフトへの変更を保存
```  
  {{% /tab %}}
{{< /tabpane >}}