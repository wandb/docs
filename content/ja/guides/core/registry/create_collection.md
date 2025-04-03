---
title: Create a collection
menu:
  default:
    identifier: ja-guides-core-registry-create_collection
    parent: registry
weight: 4
---

*コレクション* とは、レジストリ内のリンクされた artifact バージョンのセットです。各コレクションは、明確なタスクまたはユースケースを表します。

たとえば、コア Dataset レジストリ内には複数のコレクションが存在する場合があります。各コレクションには、MNIST、CIFAR-10、ImageNet など、異なるデータセットが含まれています。

別の例として、「chatbot」というレジストリがあり、モデル artifact のコレクション、データセット artifact の別のコレクション、ファインチューンされたモデル artifact の別のコレクションが含まれている場合があります。

レジストリとそのコレクションをどのように編成するかは、あなた次第です。

{{% alert %}}
W&B Model Registry に精通している場合は、登録済みモデルをご存知かもしれません。Model Registry の登録済みモデルは、W&B Registry ではコレクションと呼ばれるようになりました。
{{% /alert %}}

## コレクションタイプ

各コレクションは、1 つ、かつ 1 つのみの artifact の *タイプ* を受け入れます。指定するタイプは、あなたと組織の他のメンバーがそのコレクションにリンクできる artifact の種類を制限します。

{{% alert %}}
artifact のタイプは、Python などのプログラミング言語のデータ型に似ていると考えることができます。このアナロジーでは、コレクションは文字列、整数、または浮動小数点数を格納できますが、これらのデータ型の組み合わせは格納できません。
{{% /alert %}}

たとえば、「データセット」 artifact タイプを受け入れるコレクションを作成するとします。これは、タイプ「データセット」を持つ将来の artifact バージョンのみをこのコレクションにリンクできることを意味します。同様に、モデル artifact タイプのみを受け入れるコレクションには、タイプ「model」の artifact のみをリンクできます。

{{% alert %}}
artifact オブジェクトを作成するときに、artifact のタイプを指定します。`wandb.Artifact()` の `type` フィールドに注意してください。

```python
import wandb

# Initialize a run
run = wandb.init(
  entity = "<team_entity>",
  project = "<project>"
  )

# Create an artifact object
artifact = wandb.Artifact(
    name="<artifact_name>", 
    type="<artifact_type>"
    )
```
{{% /alert %}}
 

コレクションを作成するときは、定義済みの artifact タイプのリストから選択できます。利用可能な artifact タイプは、コレクションが属するレジストリによって異なります。

artifact をコレクションにリンクする、または新しいコレクションを作成する前に、[コレクションが受け入れる artifact のタイプを調べてください]({{< relref path="#check-the-types-of-artifact-that-a-collection-accepts" lang="ja" >}})。

### コレクションが受け入れる artifact のタイプを確認する

コレクションにリンクする前に、コレクションが受け入れる artifact タイプを調べてください。コレクションが受け入れる artifact タイプは、W&B Python SDK を使用してプログラムで、または W&B App を使用してインタラクティブに調べることができます。

{{% alert %}}
その artifact タイプを受け入れないコレクションに artifact をリンクしようとすると、エラーメッセージが表示されます。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="W&B App" %}}
受け入れられる artifact タイプは、ホームページのレジストリカード、またはレジストリの設定ページにあります。

どちらの方法でも、まず W&B Registry App に移動します。

Registry App のホームページ内で、そのレジストリのレジストリカードまでスクロールすると、受け入れられる artifact タイプを表示できます。レジストリカード内の灰色の水平方向の楕円は、そのレジストリが受け入れる artifact タイプを示しています。

{{< img src="/images/registry/artifact_types_model_card.png" alt="" >}}

たとえば、上記の画像は、Registry App ホームページの複数のレジストリカードを示しています。**Model** レジストリカード内には、**model** と **model-new** の 2 つの artifact タイプが表示されています。

レジストリの設定ページ内で受け入れられる artifact タイプを表示するには：

1. 設定を表示するレジストリカードをクリックします。
2. 右上隅にある歯車アイコンをクリックします。
3. **受け入れられる artifact タイプ** フィールドまでスクロールします。   
  {{% /tab %}}
  {{% tab header="Python SDK (Beta)" %}}
W&B Python SDK を使用して、レジストリが受け入れる artifact タイプをプログラムで表示します。

```python
import wandb

registry_name = "<registry_name>"
artifact_types = wandb.Api().project(name=f"wandb-registry-{registry_name}").artifact_types()
print(artifact_type.name for artifact_type in artifact_types)
```

{{% alert %}}
上記のコードスニペットでは run を初期化しないことに注意してください。これは、W&B API をクエリするだけで、experiment、artifact などを追跡しない場合は、run を作成する必要がないためです。
{{% /alert %}}  
  {{% /tab %}}
{{< /tabpane >}}

コレクションが受け入れる artifact のタイプがわかったら、[コレクションを作成]({{< relref path="#create-a-collection" lang="ja" >}})できます。

## コレクションを作成する

レジストリ内にインタラクティブまたはプログラムでコレクションを作成します。コレクションの作成後に、コレクションが受け入れる artifact のタイプを変更することはできません。

### プログラムでコレクションを作成する

`wandb.init.link_artifact()` メソッドを使用して、artifact をコレクションにリンクします。`target_path` フィールドに、コレクションとレジストリの両方を次の形式のパスとして指定します。

```python
f"wandb-registry-{registry_name}/{collection_name}"
```

ここで、`registry_name` はレジストリの名前、`collection_name` はコレクションの名前です。プレフィックス `wandb-registry-` をレジストリ名に必ず追加してください。

{{% alert %}}
存在しないコレクションに artifact をリンクしようとすると、W&B は自動的にコレクションを作成します。存在するコレクションを指定すると、W&B は artifact を既存のコレクションにリンクします。
{{% /alert %}}

上記のコードスニペットは、プログラムでコレクションを作成する方法を示しています。`<>` で囲まれた他の値を必ず独自の値に置き換えてください。

```python
import wandb

# Initialize a run
run = wandb.init(entity = "<team_entity>", project = "<project>")

# Create an artifact object
artifact = wandb.Artifact(
  name = "<artifact_name>",
  type = "<artifact_type>"
  )

registry_name = "<registry_name>"
collection_name = "<collection_name>"
target_path = f"wandb-registry-{registry_name}/{collection_name}"

# Link the artifact to a collection
run.link_artifact(artifact = artifact, target_path = target_path)

run.finish()
```

### インタラクティブにコレクションを作成する

次の手順では、W&B Registry App UI を使用してレジストリ内にコレクションを作成する方法について説明します。

1. W&B App UI の **Registry** App に移動します。
2. レジストリを選択します。
3. 右上隅にある **コレクションを作成** ボタンをクリックします。
4. **名前** フィールドにコレクションの名前を入力します。
5. **タイプ** ドロップダウンからタイプを選択します。または、レジストリがカスタム artifact タイプを有効にする場合は、このコレクションが受け入れる 1 つまたは複数の artifact タイプを入力します。
6. 必要に応じて、**説明** フィールドにコレクションの説明を入力します。
7. 必要に応じて、**タグ** フィールドに 1 つまたは複数のタグを追加します。
8. **バージョンのリンク** をクリックします。
9. **プロジェクト** ドロップダウンから、artifact が保存されているプロジェクトを選択します。
10. **Artifact** コレクションドロップダウンから、artifact を選択します。
11. **バージョン** ドロップダウンから、コレクションにリンクする artifact バージョンを選択します。
12. **コレクションを作成** ボタンをクリックします。

{{< img src="/images/registry/create_collection.gif" alt="" >}}
