---
title: コレクションを作成
menu:
  default:
    identifier: create_collection
    parent: registry
weight: 4
---

*コレクション* とは、レジストリ内でリンクされたアーティファクト バージョンのセットです。各コレクションは、それぞれ独立したタスクやユースケースを表します。

たとえば、コアの Dataset レジストリ内には複数のコレクションを持つことができます。それぞれのコレクションには、MNIST、CIFAR-10、ImageNet など異なるデータセットが含まれます。

別の例として、「chatbot」というレジストリには、モデルアーティファクト用、データセットアーティファクト用、ファインチューン済みモデルアーティファクト用など、異なるコレクションを作成することもできます。

レジストリとそのコレクションの構成は、あなた自身で自由に決めることができます。

{{% alert %}}
W&B Model Registry に馴染みのある方は registered models という用語をご存じかもしれません。Model Registry 内の registered models は、W&B Registry ではコレクションと呼ばれるようになりました。
{{% /alert %}}

## コレクションのタイプ

各コレクションでは、*1 種類のみ* のアーティファクトタイプを受け付けます。指定したタイプによって、あなたやチームメンバーがそのコレクションにリンクできるアーティファクトの種類が制限されます。

{{% alert %}}
アーティファクトタイプは、プログラミング言語 (Python など) のデータ型に似ているとイメージすると分かりやすいです。この例えでは、コレクションは文字列・整数・浮動小数点のいずれかのみを保存できますが、これらを混在させることはできません。
{{% /alert %}}

たとえば、「dataset」タイプのアーティファクトのみ受け付けるコレクションを作成した場合、そのコレクションには今後「dataset」タイプを持つアーティファクトバージョンしかリンクできません。同様に、「model」タイプ専用のコレクションには「model」タイプのアーティファクトのみリンクできます。

{{% alert %}}
アーティファクトのタイプは、そのアーティファクトオブジェクトを作成する際に指定します。`wandb.Artifact()` の `type` フィールドにご注目ください。

```python
import wandb

# run を初期化
run = wandb.init(
  entity = "<team_entity>",
  project = "<project>"
  )

# アーティファクトオブジェクトを作成
artifact = wandb.Artifact(
    name="<artifact_name>", 
    type="<artifact_type>"
    )
```
{{% /alert %}}
 
コレクション作成時には、あらかじめ定義されたアーティファクトタイプの一覧から選択できます。利用可能なアーティファクトタイプは、そのコレクションが属するレジストリによって異なります。

コレクションにアーティファクトをリンクしたり、新たなコレクションを作成する前に、[そのコレクションが受け付けるアーティファクトタイプを確認しましょう]({{< relref "#check-the-types-of-artifact-that-a-collection-accepts" >}})。

### コレクションが受け付けるアーティファクトタイプの確認方法

コレクションにリンクする前に、そのコレクションが受け付けるアーティファクトタイプを調査しましょう。調査方法は、W&B Python SDK を使ったプログラムによる方法と、W&B App でのインタラクティブな方法があります。

{{% alert %}}
コレクションが受け付けていないアーティファクトタイプのアーティファクトをリンクしようとすると、エラーメッセージが表示されます。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="W&B App" %}}
レジストリカード（ホームページ上またはレジストリの設定ページ）で、受け付けているアーティファクトタイプを確認できます。

どちらの方法でも、まず W&B Registry App にアクセスしてください。

Registry App のホームページでは、登録されているレジストリカードをスクロールしていくと、各カードにそのレジストリが受け付けているアーティファクトタイプがグレーの楕円で表示されています。

{{< img src="/images/registry/artifact_types_model_card.png" alt="Artifact types selection" >}}

たとえば、上の画像ではいくつかのレジストリカードが Registry App ホームページに表示されています。**Model** レジストリカードを見ると、**model** と **model-new** の 2 種類のアーティファクトタイプが受け付けられていることが分かります。

レジストリの設定ページで受け付けているアーティファクトタイプを表示するには：

1. 設定を確認したいレジストリカードをクリックします。
2. 右上の歯車アイコンをクリックします。
3. **Accepted artifact types** フィールドまでスクロールしてください。   
  {{% /tab %}}
  {{% tab header="Python SDK (Beta)" %}}
W&B Python SDK を利用して、そのレジストリが受け付けるアーティファクトタイプをプログラムから取得できます。

```python
import wandb

registry_name = "<registry_name>"
artifact_types = wandb.Api().project(name=f"wandb-registry-{registry_name}").artifact_types()
print(artifact_type.name for artifact_type in artifact_types)
```

{{% alert %}}
このコードスニペットでは run の初期化を行いません。これは、W&B API へ問い合わせるだけなので、run を作成する必要がないためです（experimentやartifactのトラッキングが不要な場合）。
{{% /alert %}}  
  {{% /tab %}}
{{< /tabpane >}}

コレクションがどのタイプのアーティファクトを受け付けているか分かったら、[コレクションを作成しましょう]({{< relref "#create-a-collection" >}})。

## コレクションの作成

レジストリ内でコレクションをインタラクティブ、またはプログラムで作成できます。一度コレクションを作成すると、その後に受け付けるアーティファクトタイプを変更することはできません。

### プログラムでコレクションを作成する

`wandb.init.link_artifact()` メソッドを使って、アーティファクトをコレクションにリンクします。`target_path` フィールドには、以下の形式でコレクションとレジストリのパスを指定します：

```python
f"wandb-registry-{registry_name}/{collection_name}"
```

`registry_name` はレジストリ名、`collection_name` はコレクション名です。レジストリ名の前に `wandb-registry-` のプレフィックスを必ず付与してください。

{{% alert %}}
まだ存在しないコレクション名を指定してアーティファクトをリンクしようとすると、W&B は自動的にコレクションを作成します。既存のコレクション名を指定した場合は、既存のコレクションにアーティファクトがリンクされます。
{{% /alert %}}

以下のコードスニペットは、プログラムでコレクションを作成する例です。`<>` で囲まれた値はご自身の内容に置き換えてください。

```python
import wandb

# run を初期化
run = wandb.init(entity = "<team_entity>", project = "<project>")

# アーティファクトオブジェクトを作成
artifact = wandb.Artifact(
  name = "<artifact_name>",
  type = "<artifact_type>"
  )

registry_name = "<registry_name>"
collection_name = "<collection_name>"
target_path = f"wandb-registry-{registry_name}/{collection_name}"

# アーティファクトをコレクションにリンク
run.link_artifact(artifact = artifact, target_path = target_path)

run.finish()
```

### インタラクティブにコレクションを作成する

W&B Registry App UI を使って、レジストリ内でコレクションを作成する手順は以下のとおりです：

1. W&B App UI で **Registry** App に移動します。
2. レジストリを選択します。
3. 右上の **Create collection** ボタンをクリックします。
4. **Name** フィールドにコレクションの名前を入力します。
5. **Type** ドロップダウンからタイプを選択します。レジストリがカスタムアーティファクトタイプを許可していれば、コレクションが受け付ける 1 つまたは複数のアーティファクトタイプを入力できます。
6. 必要に応じて **Description** フィールドにコレクションの説明を追加します。
7. 必要に応じて **Tags** フィールドに 1 つ以上のタグを追加します。
8. **Link version** をクリックします。
9. **Project** のドロップダウンからアーティファクトが保存されているプロジェクトを選択します。
10. **Artifact** コレクションのドロップダウンからアーティファクトを選択します。
11. **Version** のドロップダウンから、コレクションにリンクしたいアーティファクトのバージョンを選びます。
12. **Create collection** ボタンをクリックします。

{{< img src="/images/registry/create_collection.gif" alt="Create a new collection" >}}