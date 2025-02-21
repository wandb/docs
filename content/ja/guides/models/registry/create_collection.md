---
title: Create a collection
menu:
  default:
    identifier: ja-guides-models-registry-create_collection
    parent: registry
weight: 4
---

*コレクション* とは、レジストリ内のリンクされた Artifact バージョンのセットです。各コレクションは、明確なタスクまたは ユースケース を表します。

たとえば、コア Dataset レジストリ内に複数のコレクションが存在する場合があります。各コレクションには、MNIST、CIFAR-10、ImageNet などの異なる データセット が含まれています。

別の例として、「chatbot」というレジストリがあり、モデル Artifact のコレクション、データセット Artifact の別のコレクション、ファインチューン されたモデル Artifact の別のコレクションが含まれている場合があります。

レジストリとそのコレクションをどのように整理するかは、ユーザー次第です。

{{% alert %}}
W&B モデルレジストリ に精通している場合、登録済みモデルについてご存知かもしれません。モデルレジストリ の登録済みモデルは、W&B レジストリ のコレクションと呼ばれるようになりました。
{{% /alert %}}

## コレクション タイプ

各コレクションは、Artifact の1つの *タイプ* のみを受け入れます。指定するタイプによって、あなたと組織の他のメンバーがそのコレクションにリンクできる Artifact の種類が制限されます。

{{% alert %}}
Artifact タイプは、Python などのプログラミング言語のデータ型と似ていると考えることができます。このアナロジーでは、コレクションは文字列、整数、または浮動小数点数を保存できますが、これらのデータ型の組み合わせは保存できません。
{{% /alert %}}

たとえば、「データセット」Artifact タイプを受け入れるコレクションを作成するとします。これは、「データセット」タイプの将来の Artifact バージョンのみをこのコレクションにリンクできることを意味します。同様に、モデル Artifact タイプのみを受け入れるコレクションには、「モデル」タイプの Artifact のみをリンクできます。

{{% alert %}}
Artifact のタイプは、その Artifact オブジェクトを作成するときに指定します。`wandb.Artifact()` の `type` フィールドに注目してください。

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
 

コレクションを作成するときに、事前定義された Artifact タイプ のリストから選択できます。利用可能な Artifact タイプ は、コレクションが属するレジストリによって異なります。

Artifact をコレクションにリンクする、または新しいコレクションを作成する前に、[コレクションが受け入れる Artifact のタイプを調べてください]({{< relref path="#check-the-types-of-artifact-that-a-collection-accepts" lang="ja" >}})。

### コレクションが受け入れる Artifact のタイプを確認する

コレクションにリンクする前に、コレクションが受け入れる Artifact タイプ を調べてください。コレクションが受け入れる Artifact タイプ は、W&B Python SDK でプログラムで、または W&B アプリ でインタラクティブに調べることができます。

{{% alert %}}
その Artifact タイプ を受け入れないコレクションに Artifact をリンクしようとすると、エラーメッセージが表示されます。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="W&B App" %}}
受け入れられる Artifact タイプ は、ホームページのレジストリカード、またはレジストリ の 設定 ページ内にあります。

どちらの方法でも、まず W&B レジストリ アプリ に移動します。

レジストリ アプリ のホームページ内で、そのレジストリ のレジストリ カードまでスクロールすると、受け入れられる Artifact タイプ を表示できます。レジストリ カード内の灰色の水平方向の楕円は、そのレジストリ が受け入れる Artifact タイプ を示しています。

{{< img src="/images/registry/artifact_types_model_card.png" alt="" >}}

たとえば、上記の画像は、レジストリ アプリ のホームページにある複数のレジストリ カードを示しています。**モデル** レジストリ カード内には、**model** と **model-new** の2つの Artifact タイプ が表示されます。

レジストリ の 設定 ページ内で受け入れられる Artifact タイプ を表示するには:

1. 設定 を表示するレジストリ カードをクリックします。
2. 右上隅にある歯車アイコンをクリックします。
3. **受け入れられる Artifact タイプ** フィールドまでスクロールします。
  {{% /tab %}}
  {{% tab header="Python SDK (Beta)" %}}
W&B Python SDK を使用して、レジストリ が受け入れる Artifact タイプ をプログラムで表示します。

```python
import wandb

registry_name = "<registry_name>"
artifact_types = wandb.Api().project(name=f"wandb-registry-{registry_name}").artifact_types()
print(artifact_type.name for artifact_type in artifact_types)
```

{{% alert %}}
上記の コードスニペット では run を初期化しないことに注意してください。これは、実験 、Artifact などを追跡するのではなく、W&B API のみをクエリする場合は、run を作成する必要がないためです。
{{% /alert %}}  
  {{% /tab %}}
{{< /tabpane >}}

コレクションが受け入れる Artifact のタイプがわかったら、[コレクションを作成]({{< relref path="#create-a-collection" lang="ja" >}}) できます。

## コレクションを作成する

レジストリ 内にインタラクティブに、またはプログラムでコレクションを作成します。コレクションの作成後に、コレクションが受け入れる Artifact のタイプを変更することはできません。

### プログラムでコレクションを作成する

`wandb.init.link_artifact()` メソッドを使用して、Artifact をコレクションにリンクします。`target_path` フィールドにコレクションとレジストリ の両方を指定します。このパスは次の形式になります。

```python
f"wandb-registry-{registry_name}/{collection_name}"
```

ここで、`registry_name` はレジストリ の名前、`collection_name` はコレクションの名前です。プレフィックス `wandb-registry-` をレジストリ 名に必ず付加してください。

{{% alert %}}
存在しないコレクションに Artifact をリンクしようとすると、W&B は自動的にコレクションを作成します。存在するコレクションを指定した場合、W&B は Artifact を既存のコレクションにリンクします。
{{% /alert %}}

次の コードスニペット は、プログラムでコレクションを作成する方法を示しています。`<>` で囲まれた他の 値 を必ず独自の値に置き換えてください。

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

次の手順では、W&B レジストリ アプリ UI を使用してレジストリ 内にコレクションを作成する方法について説明します。

1. W&B アプリ UI で **レジストリ** アプリ に移動します。
2. レジストリ を選択します。
3. 右上隅にある **コレクションを作成** ボタンをクリックします。
4. **名前** フィールドにコレクションの名前を入力します。
5. **タイプ** ドロップダウンからタイプを選択します。または、レジストリ がカスタム Artifact タイプ を有効にする場合は、このコレクションが受け入れる1つ以上の Artifact タイプ を指定します。
6. 必要に応じて、**説明** フィールドにコレクションの説明を入力します。
7. 必要に応じて、**タグ** フィールドに1つ以上のタグを追加します。
8. **バージョンのリンク** をクリックします。
9. **プロジェクト** ドロップダウンから、Artifact が保存されている プロジェクト を選択します。
10. **Artifact** コレクション ドロップダウンから、Artifact を選択します。
11. **バージョン** ドロップダウンから、コレクションにリンクする Artifact バージョンを選択します。
12. **コレクションを作成** ボタンをクリックします。

{{< img src="/images/registry/create_collection.gif" alt="" >}}
