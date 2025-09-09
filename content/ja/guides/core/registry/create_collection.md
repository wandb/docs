---
title: コレクションを作成する
menu:
  default:
    identifier: ja-guides-core-registry-create_collection
    parent: registry
weight: 4
---

*コレクション* とは、レジストリ 内のリンクされた Artifacts の複数のバージョンで構成されるセットです。各 コレクション は、個別のタスクまたはユースケースを表します。
たとえば、主要な Datasets レジストリ 内には、複数の コレクション が存在する場合があります。各 コレクション には、MNIST、CIFAR-10、ImageNet など、異なる Datasets が含まれます。
別の例として、「チャットボット」という名前の レジストリ があり、そこには Model Artifacts の コレクション、Dataset Artifacts の別の コレクション、ファインチューン された Model Artifacts の別の コレクション が含まれる場合があります。
レジストリ とその コレクション をどのように整理するかは、ユーザー次第です。
{{% alert %}}
W&B Model Registry に精通している方なら、Registered Models のことをご存知かもしれません。Model Registry の Registered Models は、W&B Registry では コレクション と呼ばれるようになりました。
{{% /alert %}}
## コレクション のタイプ
各 コレクション は、1 種類の Artifacts の *タイプ* のみを許可します。指定する タイプ によって、あなたや組織の他のメンバーがその コレクション にリンクできる Artifacts の種類が制限されます。
{{% alert %}}
Artifacts の タイプ は、Python などのプログラミング言語における データ型 と同様に考えることができます。この例えでは、コレクション は文字列、整数、または浮動小数点数を格納できますが、これらの データ型 を混在させることはできません。
{{% /alert %}}
たとえば、「dataset」 の Artifact の タイプ を受け入れる コレクション を作成したとします。これは、将来の「dataset」 タイプ を持つ Artifact の バージョン のみを、この コレクション にリンクできることを意味します。同様に、「model」 タイプの Artifact のみを、model タイプのみを受け入れる コレクション にリンクできます。
{{% alert %}}
Artifact オブジェクト を作成するときに、Artifact の タイプ を指定します。`wandb.Artifact()` の `type` フィールドに注目してください。
```python
import wandb

# run を初期化する
run = wandb.init(
  entity = "<team_entity>",
  project = "<project>"
  )

# Artifact オブジェクト を作成する
artifact = wandb.Artifact(
    name="<artifact_name>",
    type="<artifact_type>"
    )
```
{{% /alert %}}
コレクション を作成するときは、事前定義された Artifacts の タイプ のリストから選択できます。利用可能な Artifacts の タイプ は、コレクション が属する レジストリ によって異なります。
Artifacts を コレクション にリンクするか、新しい コレクション を作成する前に、[コレクション が受け入れる Artifacts の タイプ を調査してください]({{< relref path="#check-the-types-of-artifact-that-a-collection-accepts" lang="ja" >}})。
### コレクション が受け入れる Artifacts の タイプ を確認する
コレクション にリンクする前に、コレクション が受け入れる Artifacts の タイプ を確認します。コレクション が受け入れる Artifacts の タイプ は、W&B Python SDK を使用してプログラムで、または W&B App を使用して対話的に確認できます。
{{% alert %}}
その Artifact の タイプ を受け入れない コレクション に Artifact をリンクしようとすると、エラーメッセージが表示されます。
{{% /alert %}}
{{< tabpane text=true >}}
  {{% tab header="W&B App" %}}
受け入れられる Artifacts の タイプ は、ホームページ の レジストリ カード または レジストリ の 設定 ページ で確認できます。
両方の方法で、まず W&B Registry App に移動します。
Registry App の ホームページ 内で、目的の レジストリ の レジストリ カード までスクロールすると、受け入れられる Artifacts の タイプ を表示できます。レジストリ カード 内の灰色の水平な楕円には、その レジストリ が受け入れる Artifacts の タイプ がリストされています。
{{< img src="/images/registry/artifact_types_model_card.png" alt="Artifacts の タイプ の選択" >}}
たとえば、上記の画像は Registry App の ホームページ にある複数の レジストリ カード を示しています。**Model** レジストリ カード 内では、**model** と **model-new** という 2 つの Artifacts の タイプ を確認できます。
レジストリ の 設定 ページ内で受け入れられる Artifacts の タイプ を表示するには:
1. 設定 を表示したい レジストリ カード をクリックします。
2. 右上隅にある歯車アイコンをクリックします。
3. **受け入れられる Artifacts の タイプ** フィールドまでスクロールします。
  {{% /tab %}}
  {{% tab header="Python SDK (ベータ版)" %}}
W&B Python SDK を使用して、レジストリ が受け入れる Artifacts の タイプ をプログラムで表示します。
```python
import wandb

registry_name = "<registry_name>"
artifact_types = wandb.Api().project(name=f"wandb-registry-{registry_name}").artifact_types()
print(artifact_type.name for artifact_type in artifact_types)
```
{{% alert %}}
上記の コードスニペット では run を初期化しないことに注意してください。これは、W&B API にクエリを実行しているだけであり、Experiments や Artifacts などを追跡していない場合は、run を作成する必要がないためです。
{{% /alert %}}
  {{% /tab %}}
{{< /tabpane >}}
コレクション が受け入れる Artifacts の タイプ がわかったら、[コレクション を作成できます]({{< relref path="#create-a-collection" lang="ja" >}})。
## コレクション を作成する
レジストリ 内に コレクション を対話的に、またはプログラムで作成します。作成後に、コレクション が受け入れる Artifacts の タイプ を変更することはできません。
### コレクション をプログラムで作成する
`wandb.init.link_artifact()` メソッド を使用して、Artifact を コレクション にリンクします。`target_path` フィールド に コレクション と レジストリ の両方を次の形式のパスとして指定します。
```python
f"wandb-registry-{registry_name}/{collection_name}"
```
ここで `registry_name` は レジストリ の名前であり、`collection_name` は コレクション の名前です。レジストリ名にプレフィックス `wandb-registry-` を必ず追加してください。
{{% alert %}}
存在しない コレクション に Artifact をリンクしようとすると、W&B は自動的に コレクション を作成します。既存の コレクション を指定した場合、W&B は Artifact を既存の コレクション にリンクします。
{{% /alert %}}
以下の コードスニペット は、コレクション をプログラムで作成する方法を示しています。`< >` で囲まれた他の 値 を、ご自身のものに置き換えてください。
```python
import wandb

# run を初期化する
run = wandb.init(entity = "<team_entity>", project = "<project>")

# Artifact オブジェクト を作成する
artifact = wandb.Artifact(
  name = "<artifact_name>",
  type = "<artifact_type>"
  )

registry_name = "<registry_name>"
collection_name = "<collection_name>"
target_path = f"wandb-registry-{registry_name}/{collection_name}"

# Artifact を コレクション にリンクする
run.link_artifact(artifact = artifact, target_path = target_path)

run.finish()
```
### コレクション を対話的に作成する
W&B Registry App UI を使用して レジストリ 内に コレクション を作成する手順は次のとおりです。
1. W&B App UI の **Registry** App に移動します。
2. レジストリ を選択します。
3. 右上隅にある **コレクション を作成** ボタンをクリックします。
4. **名前** フィールドに コレクション の名前を入力します。
5. **タイプ** ドロップダウンから タイプ を選択します。または、レジストリ でカスタム Artifacts の タイプ が有効になっている場合は、この コレクション が受け入れる 1 つ以上の Artifacts の タイプ を指定します。
6. オプションで、**説明** フィールドに コレクション の説明を入力します。
7. オプションで、**タグ** フィールドに 1 つ以上のタグを追加します。
8. **バージョン をリンク** をクリックします。
9. **Project** ドロップダウンから、Artifact が保存されている Project を選択します。
10. **Artifact** ドロップダウンから、Artifact を選択します。
11. **バージョン** ドロップダウンから、コレクション にリンクしたい Artifact の バージョン を選択します。
12. **コレクション を作成** ボタンをクリックします。
{{< img src="/images/registry/create_collection.gif" alt="新しい コレクション を作成" >}}