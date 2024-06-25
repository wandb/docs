---
description: 単一の run から、または分散プロセスから新しいアーティファクト バージョンを作成します。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Create new artifact versions

<head>
    <title>Create new artifacts versions from single and multiprocess Runs.</title>
</head>

新しいアーティファクトバージョンを単一の [run](../runs/intro.md) または分散 run で共同作成します。前のバージョンから新しいアーティファクトバージョンを作成することもできます。このバージョンは [incremental artifact](#create-a-new-artifact-version-from-an-existing-version) として知られています。

:::tip
元のアーティファクトのサイズが大きく、一部のファイルに変更を加える必要がある場合は、インクリメンタルアーティファクトの作成をお勧めします。
:::

## 新たなアーティファクトバージョンをゼロから作成
新しいアーティファクトバージョンを作成するには、単一の run からと分散 run からの2つの方法があります。以下のように定義されています：

* **Single run**: 単一の run が新しいバージョンに必要なすべてのデータを提供します。これは最も一般的なケースで、run が必要なデータを完全に再生成する場合に最適です。例：保存されたモデルやモデル予測をテーブルに出力して分析する場合。
* **Distributed runs**: 複数の run が共同で新しいバージョンに必要なすべてのデータを提供します。これは、分散ジョブで複数の run がデータを生成する場合に最適です。例：分散方式でモデルを評価し、予測を出力する場合。

W&Bは、プロジェクト内で存在しない名前を `wandb.Artifact` API に渡すと新しいアーティファクトを作成し、それに `v0` エイリアスを割り当てます。再び同じアーティファクトにログを残すと、W&Bはその内容のチェックサムを計算し、アーティファクトが変更されていれば新しいバージョン `v1` を保存します。

プロジェクト内で既存のアーティファクトに一致する名前とアーティファクトタイプを `wandb.Artifact` API に渡すと、W&B は既存のアーティファクトを取得します。この取得されたアーティファクトは1より大きいバージョンを持ちます。

![](/images/artifacts/single_distributed_artifacts.png)

### Single run
単一の run でアーティファクトの新しいバージョンをログに記録します。これは、単一の run がアーティファクト内のすべてのファイルを生成する場合に発生します。

ユースケースに基づいて、次のタブのいずれかを選択して、run 内またはrun 外で新しいアーティファクトバージョンを作成します。

<Tabs
  defaultValue="inside"
  values={[
    {label: 'Inside a run', value: 'inside'},
    {label: 'Outside a run', value: 'outside'},
  ]}>
  <TabItem value="inside">

W&B run 内でアーティファクトバージョンを作成します：

1. `wandb.init` を使って run を作成します。（1行目）
2. `wandb.Artifact` を使って新しいアーティファクトを作成するか、既存のものを取得します。（2行目）
3. `.add_file` を使用してアーティファクトにファイルを追加します。（9行目）
4. `.log_artifact` を使用して run にアーティファクトをログに記録します。（10行目）

```python showLineNumbers
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # アーティファクトにファイルとアセットを追加
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference` を使用
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```

  </TabItem>
  <TabItem value="outside">

W&B run 外でアーティファクトバージョンを作成します：

1. `wanb.Artifact` を使って新しいアーティファクトを作成するか、既存のものを取得します。（1行目）
2. `.add_file` を使ってアーティファクトにファイルを追加します。（4行目）
3. `.save` を使ってアーティファクトを保存します。（5行目）

```python showLineNumbers
artifact = wandb.Artifact("artifact_name", "artifact_type")
# アーティファクトにファイルとアセットを追加
# `.add`, `.add_file`, `.add_dir`, and `.add_reference` を使用
artifact.add_file("image1.png")
artifact.save()
```  
  </TabItem>
</Tabs>

### Distributed runs

複数の run を使用してバージョンに協力し、それをコミットします。これは、上記の単一の run モードとは対照的で、一つの run が新しいバージョンに必要なすべてのデータを提供します。

:::info
1. コレクション内の各 run は、同じバージョンに協力するために、同じユニークID（`distributed_id` と呼ばれる）を認識している必要があります。デフォルトでは、存在する場合、W&Bは run の `group` を `wandb.init(group=GROUP)` によって `distributed_id` として使用します。
2. 「コミット」バージョンを実行する最終 run が存在する必要があります。これにより、その状態が永久にロックされます。
3. コラボレーティブアーティファクトに追加するために `upsert_artifact` を使用し、コミットを最終化するために `finish_artifact` を使用します。
:::

次の例を考えてみましょう。異なる run（以下で **Run 1**、**Run 2**、**Run 3** とラベル付けされています）が `upsert_artifact` を使用して同じアーティファクトに異なる画像ファイルを追加します。

#### Run 1:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # アーティファクトにファイルとアセットを追加
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference` を使用
    artifact.add_file("image1.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 2:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # アーティファクトにファイルとアセットを追加
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference` を使用
    artifact.add_file("image2.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 3

Run 1 と Run 2 が完了した後に実行される必要があります。`finish_artifact` を呼び出す run はアーティファクトにファイルを含めることもできますが、必ずしも含める必要はありません。

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # アーティファクトにファイルとアセットを追加
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference` を使用
    artifact.add_file("image3.png")
    run.finish_artifact(artifact, distributed_id="my_dist_artifact")
```

## Create a new artifact version from an existing version

前のアーティファクトバージョンからサブセットのファイルを追加、変更、または削除し、変更されていないファイルを再インデックス化することなく新しいアーティファクトバージョンを作成します。前のアーティファクトバージョンからサブセットのファイルを追加、変更、または削除することで、*インクリメンタルアーティファクト* として知られる新しいアーティファクトバージョンが作成されます。

![](/images/artifacts/incremental_artifacts.png)

以下に、各タイプのインクリメンタル変更に関するシナリオをいくつか示します：

- 追加: 新しいバッチを収集した後、定期的にデータセットに新しいサブセットのファイルを追加します。
- 削除: 重複したファイルを発見し、それらをアーティファクトから削除します。
- 更新: サブセットのファイルの注釈を修正し、古いファイルを新しいファイルに置き換えます。

インクリメンタルアーティファクトと同じ機能を実行するために、ゼロからアーティファクトを作成することもできます。しかし、ゼロからアーティファクトを作成する場合、アーティファクトのすべての内容をローカルディスクに持つ必要があります。インクリメンタル変更を行う場合、以前のアーティファクトバージョンのファイルを変更せずに、単一ファイルを追加、削除、または変更できます。

:::info
インクリメンタルアーティファクトは、単一の run 内または複数の run のセット（分散モード）で作成できます。
:::

以下の手順に従ってアーティファクトをインクリメンタルに変更しましょう：

1. インクリメンタル変更を行いたいアーティファクトバージョンを取得します：

<Tabs
  defaultValue="inside"
  values={[
    {label: 'Inside a run', value: 'inside'},
    {label: 'Outside of a run', value: 'outside'},
  ]}>
  <TabItem value="inside">

```python
saved_artifact = run.use_artifact("my_artifact:latest")
```

  </TabItem>
  <TabItem value="outside">

```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")
```

  </TabItem>
</Tabs>

2. 以下を使用してドラフトを作成します：

```python
draft_artifact = saved_artifact.new_draft()
```

3. 次のバージョンで見たい任意のインクリメンタル変更を行います。既存のエントリを追加、削除、または修正できます。

以下のタブのいずれかを選択して、各変更を行う方法の例を参照してください：

<Tabs
  defaultValue="add"
  values={[
    {label: 'Add', value: 'add'},
    {label: 'Remove', value: 'remove'},
    {label: 'Modify', value: 'modify'},
  ]}>
  <TabItem value="add">

`add_file` メソッドを使用して既存のアーティファクトバージョンにファイルを追加します：

```python
draft_artifact.add_file("file_to_add.txt")
```

:::note
`add_dir` メソッドを使用してディレクトリを追加することで、複数のファイルを追加することもできます。
:::

  </TabItem>
  <TabItem value="remove">

`remove` メソッドを使用して既存のアーティファクトバージョンからファイルを削除します：

```python
draft_artifact.remove("file_to_remove.txt")
```

:::note
ディレクトリパスを渡すことで、`remove` メソッドを使用して複数のファイルを削除することもできます。
:::

  </TabItem>
  <TabItem value="modify">

ドラフトから古い内容を削除し、新しい内容を再度追加することで内容を変更または置き換えます：

```python
draft_artifact.remove("modified_file.txt")
draft_artifact.add_file("modified_file.txt")
```

  </TabItem>
</Tabs>

4. 最後に、変更をログに記録または保存します。以下のタブは、それぞれのユースケースに応じた方法を示します。適切なタブを選択してください：

<Tabs
  defaultValue="inside"
  values={[
    {label: 'Inside a run', value: 'inside'},
    {label: 'Outside of a run', value: 'outside'},
  ]}>
  <TabItem value="inside">

```python
run.log_artifact(draft_artifact)
```

  </TabItem>
  <TabItem value="outside">

```python
draft_artifact.save()
```

  </TabItem>
</Tabs>

これらのコード例をまとめると、以下のようになります：

<Tabs
  defaultValue="inside"
  values={[
    {label: 'Inside a run', value: 'inside'},
    {label: 'Outside of a run', value: 'outside'},
  ]}>
  <TabItem value="inside">

```python
with wandb.init(job_type="modify dataset") as run:
    saved_artifact = run.use_artifact(
        "my_artifact:latest"
    )  # アーティファクトを取得してrunに入力
    draft_artifact = saved_artifact.new_draft()  # ドラフトバージョンを作成

    # ドラフトバージョンでサブセットのファイルを変更
    draft_artifact.add_file("file_to_add.txt")
    draft_artifact.remove("dir_to_remove/")
    run.log_artifact(
        artifact
    )  # 変更をログに記録して新しいバージョンを作成し、runの出力としてマーク
```

  </TabItem>
  <TabItem value="outside">

```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")  # アーティファクトを読み込み
draft_artifact = saved_artifact.new_draft()  # ドラフトバージョンを作成

# ドラフトバージョンでサブセットのファイルを変更
draft_artifact.remove("deleted_file.txt")
draft_artifact.add_file("modified_file.txt")
draft_artifact.save()  # ドラフトに変更をコミット
```

  </TabItem>
</Tabs>