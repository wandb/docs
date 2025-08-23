---
title: Collectionを作成する
menu:
  default:
    identifier: ja-guides-core-registry-create_collection
    parent: registry
weight: 4
---

*collection* とは、Registry内で関連付けられた複数の artifact バージョンの集合です。各collectionは、個別のタスクやユースケースを表現します。

たとえば、コア Dataset Registry内には複数のcollectionを作成できます。各collectionには、MNIST、CIFAR-10、ImageNet など、異なるデータセットが含まれます。

別の例として、「chatbot」というRegistryがあるとします。このRegistryにはモデル artifact 用のcollection、データセット artifact 用のcollection、ファインチューン済みモデル artifact 用のcollectionなどを持たせることができます。

Registryやそのcollectionをどのように整理するかは自由です。

{{% alert %}}
W&B Model Registry に慣れている方は「registered models」という言葉をご存じかもしれません。Model Registry での registered models は、W&B Registry ではcollectionと呼ばれるようになりました。
{{% /alert %}}

## collectionの種類

各collectionが受け入れられる artifact の *タイプ* は 1 つだけです。指定したタイプによって、そのcollectionにどのような artifact を自分やチームのメンバーが追加できるかが制限されます。

{{% alert %}}
artifact のタイプは、Python などのプログラミング言語のデータ型のようなものと考えると分かりやすいです。たとえばcollectionが文字列や整数、浮動小数点数のいずれかを保存できても、これらを混ぜて保存することはできません。
{{% /alert %}}

例えば、「dataset」タイプの artifact を受け入れるcollectionを作成した場合、そのcollectionには今後「dataset」タイプの artifact だけを追加できます。同様に、「model」タイプのcollectionには「model」タイプの artifact しか追加できません。

{{% alert %}}
artifact を作成する際に type（タイプ）を指定します。`wandb.Artifact()` の `type` フィールドに注目してください。

```python
import wandb

# run を初期化
run = wandb.init(
  entity = "<team_entity>",
  project = "<project>"
  )

# artifact オブジェクトを作成
artifact = wandb.Artifact(
    name="<artifact_name>", 
    type="<artifact_type>"
    )
```
{{% /alert %}}
 
collectionを作成する際には、あらかじめ定義されている artifact タイプの中から選択できます。どの artifact タイプが利用できるかは、そのcollectionが属しているRegistryによって異なります。

artifact をcollectionにリンクしたり新たなcollectionを作成する前に、[そのcollectionが受け入れる artifact のタイプを確認してください]({{< relref path="#check-the-types-of-artifact-that-a-collection-accepts" lang="ja" >}})。

### collectionが受け入れる artifact タイプを確認する

collectionにリンクする前に、そのcollectionが受け入れる artifact タイプを確認しましょう。artifact タイプは W&B Python SDK を使ってプログラム的に、もしくは W&B App でインタラクティブに確認できます。

{{% alert %}}
collectionが受け入れない artifact タイプをリンクしようとすると、エラーメッセージが表示されます。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="W&B App" %}}
受け入れ可能な artifact タイプは、ホームページのRegistryカードやRegistryの設定ページで確認できます。

まずは、W&B Registry App のホームページにアクセスしてください。

Registry App のホームページでは、任意のRegistryカードまでスクロールすると、そのRegistryが受け入れる artifact タイプが灰色の横長の楕円で表示されています。

{{< img src="/images/registry/artifact_types_model_card.png" alt="Artifact types selection" >}}

例えば、上の画像では複数のRegistryカードが表示されています。**Model** Registryカードには、**model** と **model-new** という2つの artifact タイプが表示されています。

Registryの設定ページで artifact タイプを確認するには：

1. 設定を確認したいRegistryカードをクリックします。
2. 右上の歯車アイコンをクリックします。
3. **Accepted artifact types** フィールドまでスクロールします。   
  {{% /tab %}}
  {{% tab header="Python SDK (Beta)" %}}
W&B Python SDK を使えば、プログラムからRegistryが受け入れる artifact タイプを確認できます。

```python
import wandb

registry_name = "<registry_name>"
artifact_types = wandb.Api().project(name=f"wandb-registry-{registry_name}").artifact_types()
print(artifact_type.name for artifact_type in artifact_types)
```

{{% alert %}}
このコードスニペットでは run の初期化は不要です。W&B の API にクエリを送るだけの場合、run の作成や実験のトラッキング、artifact の作成などは必要ありません。
{{% /alert %}}  
  {{% /tab %}}
{{< /tabpane >}}

collectionがどのタイプの artifact を受け入れるか分かったら、[collectionを作成]({{< relref path="#create-a-collection" lang="ja" >}})できます。

## collectionを作成する

collectionは、インタラクティブまたはプログラム的にRegistry内へ作成できます。一度作成すると、そのcollectionが受け入れられる artifact タイプは変更できません。

### プログラムでcollectionを作成する

`wandb.init.link_artifact()` メソッドを使って artifact をcollectionにリンクします。`target_path` フィールドには、Registry名とcollection名を組み合わせた次の形式で指定します：

```python
f"wandb-registry-{registry_name}/{collection_name}"
```

ここで `registry_name` はRegistry名、`collection_name` はcollection名です。Registry名の前には必ず `wandb-registry-` プレフィックスを付けてください。

{{% alert %}}
存在しないcollectionへ artifact をリンクしようとすると、W&B が自動的にcollectionを作成します。既存のcollectionを指定した場合は、そのcollectionに artifact がリンクされます。
{{% /alert %}}

以下のコードスニペットは、プログラム的にcollectionを作成する手順例です。`<>` で囲まれている部分はご自身の値に置き換えてください。

```python
import wandb

# run を初期化
run = wandb.init(entity = "<team_entity>", project = "<project>")

# artifact オブジェクトを作成
artifact = wandb.Artifact(
  name = "<artifact_name>",
  type = "<artifact_type>"
  )

registry_name = "<registry_name>"
collection_name = "<collection_name>"
target_path = f"wandb-registry-{registry_name}/{collection_name}"

# artifact をcollectionにリンク
run.link_artifact(artifact = artifact, target_path = target_path)

run.finish()
```

### インタラクティブにcollectionを作成する

W&B Registry App UI を使ってRegistry内にcollectionを作成する手順は以下の通りです：

1. W&B App UI の **Registry** App に移動します。
2. Registryを選択します。
3. 画面右上の **Create collection** ボタンをクリックします。
4. **Name** フィールドにcollection名を入力します。
5. **Type** ドロップダウンからタイプを選びます。もしくは、Registryでカスタム artifact タイプが許可されている場合は、受け入れる artifact タイプを設定してください。
6. 任意で、**Description** フィールドにcollectionの説明を追加します。
7. 任意で、**Tags** フィールドにtagを追加します。
8. **Link version** をクリックします。
9. **Project** ドロップダウンから artifact が保存されているプロジェクトを選択します。
10. **Artifact collection** ドロップダウンから目的の artifact を選択します。
11. **Version** ドロップダウンからcollectionにリンクしたい artifact バージョンを選びます。
12. **Create collection** ボタンをクリックします。

{{< img src="/images/registry/create_collection.gif" alt="Create a new collection" >}}