---
title: Manage artifact data retention
description: 生存時間ポリシー (TTL)
menu:
  default:
    identifier: ja-guides-core-artifacts-manage-data-ttl
    parent: manage-data
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/kas-artifacts-ttl-colab/colabs/wandb-artifacts/WandB_Artifacts_Time_to_live_TTL_Walkthrough.ipynb" >}}

W&B Artifact のタイム・トゥ・リブ (TTL) ポリシーを使用して、アーティファクトが W&B から削除されるタイミングをスケジュールします。アーティファクトを削除すると、W&B はそのアーティファクトを*ソフト削除*としてマークします。つまり、アーティファクトは削除のためにマークされますが、ファイルはストレージからすぐには削除されません。W&B がアーティファクトを削除する方法の詳細については、[Delete artifacts]({{< relref path="./delete-artifacts.md" lang="ja" >}}) ページをご覧ください。

W&B アプリでの Artifacts TTL を使用してデータ保持を管理する方法については、[こちら](https://www.youtube.com/watch?v=hQ9J6BoVmnc) のビデオチュートリアルをご覧ください。

{{% alert %}}
W&B は、Model Registry にリンクされたモデルアーティファクトに対して TTL ポリシーを設定するオプションを無効にします。これは、プロダクションワークフローで使用されているリンクされたモデルが誤って期限切れにならないようにするためです。
{{% /alert %}}
{{% alert %}}
* チームの管理者のみが[チームの設定]({{< relref path="/guides/models/app/settings-page/team-settings.md" lang="ja" >}})を表示でき、TTL 設定の編集は (1) TTL ポリシーを設定または編集できる人を許可する、(2) チームのデフォルト TTL を設定する、といったことができます。  
* W&B アプリ UI 内のアーティファクトの詳細で TTL ポリシーを設定または編集するオプションが見当たらない場合、またはプログラムで TTL を設定してもアーティファクトの TTL プロパティが正常に変更されない場合は、チームの管理者が権限を付与していない可能性があります。
{{% /alert %}}

## 自動生成されたアーティファクト
ユーザー生成のアーティファクトのみが TTL ポリシーを使用できます。W&B によって自動生成されたアーティファクトには TTL ポリシーを設定できません。

自動生成されたアーティファクトを示すアーティファクトタイプは次のとおりです：
- `run_table`
- `code`
- `job`
- 次で始まる任意のアーティファクトタイプ: `wandb-*`

アーティファクトのタイプは [W&B プラットフォーム]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ja" >}}) またはプログラムで確認できます：

```python
import wandb

run = wandb.init(project="<my-project-name>")
artifact = run.use_artifact(artifact_or_name="<my-artifact-name>")
print(artifact.type)
```

`<>` で囲まれた値は、自身のものに置き換えてください。

## TTL ポリシーを編集および設定できる人を定義する
チーム内で TTL ポリシーを設定および編集できる人を定義します。TTL の権限をチームの管理者にのみ付与するか、チームの管理者とメンバーの両方に TTL の権限を付与します。 

{{% alert %}}
TTL ポリシーを設定または編集できる人を定義できるのはチームの管理者のみです。
{{% /alert %}}

1. チームのプロファイルページに移動します。
2. **Settings** タブを選択します。
3. **Artifacts time-to-live (TTL) セクション**に移動します。
4. **TTL permissions ドロップダウン**から TTL ポリシーを設定および編集できる人を選択します。
5. **Review and save settings** をクリックします。
6. 変更を確認し、**Save settings** を選択します。

{{< img src="/images/artifacts/define_who_sets_ttl.gif" alt="" >}}

## TTL ポリシーを作成する
アーティファクトを作成する際、または作成後に、そのアーティファクトに対して TTL ポリシーを設定します。

以下のコードスニペットのすべてにおいて、`<>` で囲まれた内容を自身の情報に置き換えてコードスニペットを利用してください。

### アーティファクトを作成する際に TTL ポリシーを設定する
W&B Python SDK を使用してアーティファクトを作成するときに TTL ポリシーを定義します。通常、TTL ポリシーは日単位で定義されます。    

{{% alert %}}
アーティファクトを作成する際に TTL ポリシーを定義するのは、通常の [アーティファクトを作成する]({{< relref path="../construct-an-artifact.md" lang="ja" >}}) 方法に似ています。ただし、アーティファクトの `ttl` 属性に時間差を渡す点を除きます。
{{% /alert %}}

手順は以下の通りです：

1. [アーティファクトを作成する]({{< relref path="../construct-an-artifact.md" lang="ja" >}})。
2. ファイル、ディレクトリ、または参照のようなコンテンツをアーティファクトに[追加]({{< relref path="../construct-an-artifact.md#add-files-to-an-artifact" lang="ja" >}})します。
3. Python の標準ライブラリの一部である [`datetime.timedelta`](https://docs.python.org/3/library/datetime.html) データ型で TTL の時間制限を定義します。
4. [アーティファクトをログする]({{< relref path="../construct-an-artifact.md#3-save-your-artifact-to-the-wb-server" lang="ja" >}})。

以下のコードスニペットは、アーティファクトを作成し、TTL ポリシーを設定する方法を示しています。

```python
import wandb
from datetime import timedelta

run = wandb.init(project="<my-project-name>", entity="<my-entity>")
artifact = wandb.Artifact(name="<artifact-name>", type="<type>")
artifact.add_file("<my_file>")

artifact.ttl = timedelta(days=30)  # TTL ポリシーを設定する
run.log_artifact(artifact)
```

前述のコードスニペットは、アーティファクトに対して TTL ポリシーを 30 日に設定します。言い換えれば、W&B は 30 日後にそのアーティファクトを削除します。

### アーティファクトを作成した後に TTL ポリシーを設定または編集する
W&B アプリ UI または W&B Python SDK を使用して、既に存在するアーティファクトの TTL ポリシーを定義します。

{{% alert %}}
アーティファクトの TTL を変更すると、アーティファクトが期限切れになるまでの時間はアーティファクトの `createdAt` タイムスタンプを使用して計算され続けます。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
1. [アーティファクトをフェッチする]({{< relref path="../download-and-use-an-artifact.md" lang="ja" >}})。
2. アーティファクトの `ttl` 属性に時間差を渡します。
3. [`save`]({{< relref path="/ref/python/run.md#save" lang="ja" >}}) メソッドでアーティファクトを更新します。

以下のコードスニペットは、アーティファクトの TTL ポリシーを設定する方法を示しています：
```python
import wandb
from datetime import timedelta

artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = timedelta(days=365 * 2)  # 2 年後に削除
artifact.save()
```

前述のコード例は、TTL ポリシーを 2 年に設定しています。
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. W&B アプリ UI 内のプロジェクトに移動します。
2. 左側のパネルのアーティファクトアイコンを選択します。
3. アーティファクトのリストから、アーティファクトタイプを展開します。
4. 編集したい TTL ポリシーのアーティファクトバージョンを選択します。
5. **Version** タブをクリックします。
6. ドロップダウンから **Edit TTL policy** を選択します。
7. 表示されるモーダルで、TTL ポリシードロップダウンから **Custom** を選択します。
8. **TTL duration** フィールドで、日単位で TTL ポリシーを設定します。
9. **Update TTL** ボタンを選択して変更を保存します。

{{< img src="/images/artifacts/edit_ttl_ui.gif" alt="" >}}  
  {{% /tab %}}
{{< /tabpane >}}

### チームのデフォルト TTL ポリシーを設定する

{{% alert %}}
デフォルトの TTL ポリシーを設定できるのはチームの管理者のみです。
{{% /alert %}}

チームのデフォルト TTL ポリシーを設定します。デフォルトの TTL ポリシーは、作成日を基にしたすべての既存および将来のアーティファクトに適用されます。既存のバージョンレベル TTL ポリシーを持つアーティファクトは、チームのデフォルト TTL の影響を受けません。

1. チームのプロファイルページに移動します。
2. **Settings** タブを選択します。
3. **Artifacts time-to-live (TTL) セクション**に移動します。
4. **Set team's default TTL policy** をクリックします。
5. **Duration** フィールドで、日単位で TTL ポリシーを設定します。
6. **Review and save settings** をクリックします。
7. 変更を確認し、**Save settings** を選択します。

{{< img src="/images/artifacts/set_default_ttl.gif" alt="" >}}

### Run の外で TTL ポリシーを設定する

パブリック API を使用して、Run をフェッチせずにアーティファクトを取得し、TTL ポリシーを設定します。TTL ポリシーは通常、日単位で定義されます。

以下のコードサンプルは、パブリック API を使用してアーティファクトをフェッチし、TTL ポリシーを設定する方法を示しています。

```python 
api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

artifact.ttl = timedelta(days=365)  # 1 年後に削除

artifact.save()
```

## TTL ポリシーを無効にする
特定のアーティファクトバージョンに対して W&B Python SDK または W&B アプリ UI を使用して TTL ポリシーを無効にします。

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
1. [アーティファクトをフェッチする]({{< relref path="../download-and-use-an-artifact.md" lang="ja" >}})。
2. アーティファクトの `ttl` 属性を `None` に設定します。
3. [`save`]({{< relref path="/ref/python/run.md#save" lang="ja" >}}) メソッドでアーティファクトを更新します。

以下のコードスニペットは、アーティファクトの TTL ポリシーをオフにする方法を示しています：

```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = None
artifact.save()
```  
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. W&B アプリ UI 内のプロジェクトに移動します。
2. 左側のパネルのアーティファクトアイコンを選択します。
3. アーティファクトのリストから、アーティファクトタイプを展開します。
4. 編集したい TTL ポリシーのアーティファクトバージョンを選択します。
5. Version タブをクリックします。
6. **Link to registry** ボタンの横にあるミートボール UI アイコンをクリックします。
7. ドロップダウンから **Edit TTL policy** を選択します。
8. 表示されるモーダルで、TTL ポリシードロップダウンから **Deactivate** を選択します。
9. **Update TTL** ボタンを選択して変更を保存します。

{{< img src="/images/artifacts/remove_ttl_polilcy.gif" alt="" >}}  
  {{% /tab %}}
{{< /tabpane >}}

## TTL ポリシーを表示する
Python SDK または W&B アプリ UI を使用してアーティファクトの TTL ポリシーを表示します。

{{< tabpane text=true >}}
  {{% tab  header="Python SDK" %}}
print 文を使用してアーティファクトの TTL ポリシーを表示します。以下の例は、アーティファクトを取得してその TTL ポリシーを表示する方法を示しています：

```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
print(artifact.ttl)
```  
  {{% /tab %}}
  {{% tab  header="W&B App" %}}
W&B アプリ UI を使用してアーティファクトの TTL ポリシーを表示します。

1. W&B アプリの [https://wandb.ai](https://wandb.ai) に移動します。
2. W&B プロジェクトに進みます。
3. プロジェクト内で、左側のサイドバーにある Artifacts タブを選択します。
4. コレクションをクリックします。

コレクションビュー内では、選択したコレクション内のすべてのアーティファクトを見ることができます。`Time to Live` カラム内で、アーティファクトに割り当てられた TTL ポリシーが表示されます。

{{< img src="/images/artifacts/ttl_collection_panel_ui.png" alt="" >}}  
  {{% /tab %}}
{{< /tabpane >}}