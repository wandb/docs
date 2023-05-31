---
description: Create a new artifact version from a single run or from a distributed process.
displayed_sidebar: ja
---

# 新しいアーティファクトのバージョンを作成

<head>
    <title>単一およびマルチプロセスのRunsから新しいアーティファクトのバージョンを作成します。</title>
</head>
単一のrunを使って新しいアーティファクトバージョンを作成するか、分散ライターを共同で使用して作成するか、前のバージョンに対するパッチとして作成します。

3つの方法で新しいアーティファクトバージョンを作成します：

* **シンプル**：単一のrunが新しいバージョンのすべてのデータを提供します。これは最も一般的なケースであり、runが必要なデータを完全に再作成する場合に最適です。例えば：分析用のテーブルに保存されたモデルやモデル予測を出力する場合。
* **コラボレーティブ**：一連のrunが新しいバージョンのすべてのデータを共同で提供します。これは、データを生成する複数のrunを持つ分散ジョブに最適です。例えば：モデルを分散して評価し、予測を出力する場合。
* **パッチ**：（近日公開）単一のrunが適用すべき差分のパッチを提供します。これは、既存のデータをすべて再作成することなく、runがアーティファクトにデータを追加したい場合に最適です。例えば：毎日ウェブスクレイピングを実行して作成されるゴールデンデータセットがある場合、このケースではrunがデータセットに新しいデータを追加することを望みます。
![アーティファクト概要図](/images/artifacts/create_new_artifact_version.png)

### シンプルモード

アーティファクト内のすべてのファイルを生成する単一のrunで新しいバージョンのアーティファクトをログするには、シンプルモードを使用してください：

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # `.add`, `.add_file`, `.add_dir`, および`.add_reference` を使って
    # アーティファクトにファイルとアセットを追加
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```
`Artifact.save()` を使用して、runを開始せずにバージョンを作成します。

```python
artifact = wandb.Artifact("artifact_name", "artifact_type")
# `.add`, `.add_file`, `.add_dir`, そして `.add_reference` を使って
# アーティファクトにファイルやアセットを追加する
artifact.add_file("image1.png")
artifact.save()
```
### コラボレーティブモード

コラボレーティブモードを使用して、複数のrunsがバージョンを共同で作成することができます。コラボレーティブモードを使用する際には、次の2つの重要な点を理解しておく必要があります。

1. コレクション内の各Runは、同じ一意のID（`distributed_id`と呼ばれる）を認識して、同じバージョンで協力する必要があります。デフォルトでは、Weights & Biasesは、`wandb.init(group=GROUP)`で設定されたrunの`group`を`distributed_id`として使用します（もしある場合）。
2. 状態を永久にロックするバージョンを"コミット"する最終的なrunが必要です。
次の例を考えてみましょう。`log_artifact`を使う代わりに、`upsert_artifact`を使ってコラボレーションアーティファクトを追加し、`finish_artifact`を使ってコミットを確定します。

#### Run 1:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # アーティファクトにファイルやアセットを追加する方法は
    # `.add`, `.add_file`, `.add_dir`, そして `.add_reference` を使います
    artifact.add_file("image1.png")
    run.upsert_artifact(
        artifact, 
        distributed_id="my_dist_artifact"
        )     
```
#### Run 2:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # アーティファクトにファイルやアセットを追加するには、
    # `.add`、`.add_file`、`.add_dir`、`.add_reference`を使用します
    artifact.add_file("image2.png")
    run.upsert_artifact(
        artifact, 
        distributed_id="my_dist_artifact"
        )
```
#### Run 3

Run 1およびRun 2が完了した後に実行する必要があります。`finish_artifact`を呼び出すRunでは、アーティファクトにファイルを含めることができますが、必ずしも含める必要はありません。

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # アーティファクトにファイルやアセットを追加する
    # `.add`, `.add_file`, `.add_dir`, および `.add_reference`
    artifact.add_file("image3.png")
    run.finish_artifact(
        artifact, 
        distributed_id="my_dist_artifact"
        )
```