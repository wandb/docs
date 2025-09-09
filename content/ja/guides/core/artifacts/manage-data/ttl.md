---
title: Artifacts のデータ保持を管理する
description: 保持期間ポリシー (TTL)
menu:
  default:
    identifier: ja-guides-core-artifacts-manage-data-ttl
    parent: manage-data
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/kas-artifacts-ttl-colab/colabs/wandb-artifacts/WandB_Artifacts_Time_to_live_TTL_Walkthrough.ipynb" >}}
W&B の Artifact time-to-live (TTL) ポリシーを使用して、W&B からアーティファクトが削除される時期をスケジュールします。アーティファクトを削除すると、W&B はそのアーティファクトを *ソフト削除* としてマークします。つまり、アーティファクトは削除対象としてマークされますが、ファイルはストレージからすぐに削除されません。W&B がアーティファクトを削除する方法の詳細については、[アーティファクトの削除]({{< relref path="./delete-artifacts.md" lang="ja" >}})ページを参照してください。
W&B App で Artifacts TTL を使用してデータの保持を管理する方法については、[Artifacts TTL によるデータ保持の管理](https://www.youtube.com/watch?v=hQ9J6BoVmnc)のビデオチュートリアルをご覧ください。
{{% alert %}}
W&B は、Model Registry にリンクされている モデル アーティファクト に対する TTL ポリシー設定オプションを無効にします。これは、リンクされたモデルがプロダクションのワークフローで使用されている場合に、誤って期限切れになることを防ぐためです。
{{% /alert %}}
{{% alert %}}
* チームの admin のみが、[チームの設定]({{< relref path="/guides/models/app/settings-page/team-settings.md" lang="ja" >}})を表示し、(1) TTL ポリシーを設定または編集できるユーザーを許可する、(2) チームのデフォルト TTL を設定するなど、チームレベルの TTL 設定にアクセスできます。
* W&B App UI のアーティファクトの詳細で TTL ポリシーを設定または編集するオプションが表示されない場合、またはプログラムで TTL を設定してもアーティファクトの TTL プロパティが正常に変更されない場合、チームの admin がそのための権限を付与していません。
{{% /alert %}}

## 自動生成されたアーティファクト
ユーザーが生成したアーティファクトのみが TTL ポリシーを使用できます。W&B によって自動生成されたアーティファクトには、TTL ポリシーを設定できません。
次の Artifact タイプは、自動生成されたアーティファクトを示します。
- `run_table`
- `code`
- `job`
- `wandb-*` で始まる任意の Artifact タイプ
[W&B platform]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ja" >}}) またはプログラムで Artifact のタイプを確認できます.
```python
import wandb

run = wandb.init(project="<my-project-name>")
artifact = run.use_artifact(artifact_or_name="<my-artifact-name>")
print(artifact.type)
```
`<` `>` で囲まれた値を独自の値に置き換えてください。

## TTL ポリシーを編集および設定できるユーザーを定義する
チーム内で TTL ポリシーを設定および編集できるユーザーを定義します。TTL 権限をチームの admin のみに付与することも、チームの admin とチームメンバーの両方に付与することもできます。
{{% alert %}}
チームの admin のみが、TTL ポリシーを設定または編集できるユーザーを定義できます。
{{% /alert %}}
1. チームのプロフィールページに移動します。
2. **Settings** タブを選択します。
3. **Artifacts time-to-live (TTL) セクション**に移動します。
4. **TTL permissions ドロップダウン**から、TTL ポリシーを設定および編集できるユーザーを選択します。
5. **Review and save settings** をクリックします。
6. 変更を確認し、**Save settings** を選択します。
{{< img src="/images/artifacts/define_who_sets_ttl.gif" alt="Setting TTL permissions" >}}

## TTL ポリシーの作成
アーティファクトの TTL ポリシーは、アーティファクトの作成時、またはアーティファクトの作成後に遡及的に設定できます。
以下のすべてのコードスニペットでは、`<` `>` で囲まれた内容を自分の情報に置き換えて、コードスニペットを使用してください。

### アーティファクトの作成時に TTL ポリシーを設定する
アーティファクトを作成する際に、W&B Python SDK を使用して TTL ポリシーを定義します。TTL ポリシーは通常、日数で定義されます。
{{% alert %}}
アーティファクトを作成する際に TTL ポリシーを定義することは、通常 [アーティファクトを作成する]({{< relref path="../construct-an-artifact.md" lang="ja" >}})方法と似ています。ただし、アーティファクトの `ttl` 属性に時間デルタを渡す点が異なります。
{{% /alert %}}
手順は以下のとおりです。
1. [アーティファクトを作成]({{< relref path="../construct-an-artifact.md" lang="ja" >}})します。
2. ファイル、ディレクトリー、または参照などの[コンテンツをアーティファクトに追加]({{< relref path="../construct-an-artifact.md#add-files-to-an-artifact" lang="ja" >}})します。
3. Python の標準ライブラリの一部である[`datetime.timedelta`](https://docs.python.org/3/library/datetime.html)データ型を使用して TTL 時間制限を定義します。
4. [アーティファクトをログ]({{< relref path="../construct-an-artifact.md#3-save-your-artifact-to-the-wb-server" lang="ja" >}})に記録します。
以下のコードスニペットは、アーティファクトを作成し、TTL ポリシーを設定する方法を示しています。
```python
import wandb
from datetime import timedelta

run = wandb.init(project="<my-project-name>", entity="<my-entity>")
artifact = wandb.Artifact(name="<artifact-name>", type="<type>")
artifact.add_file("<my_file>")

artifact.ttl = timedelta(days=30)  # TTLポリシーを設定
run.log_artifact(artifact)
```
上記のコードスニペットは、アーティファクトの TTL ポリシーを 30 日に設定しています。つまり、W&B は 30 日後にアーティファクトを削除します。

### アーティファクト作成後に TTL ポリシーを設定または編集する
W&B App UI または W&B Python SDK を使用して、既存のアーティファクトの TTL ポリシーを定義します。
{{% alert %}}
アーティファクトの TTL を変更しても、アーティファクトが期限切れになるまでの時間は、アーティファクトの `createdAt` タイムスタンプを使用して計算されます。
{{% /alert %}}
{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
1. [アーティファクトを取得]({{< relref path="../download-and-use-an-artifact.md" lang="ja" >}})します。
2. アーティファクトの `ttl` 属性に時間デルタを渡します。
3. [`save`]({{< relref path="/ref/python/sdk/classes/run.md#save" lang="ja" >}})メソッドでアーティファクトを更新します。
以下のコードスニペットは、アーティファクトの TTL ポリシーを設定する方法を示しています。
```python
import wandb
from datetime import timedelta

artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = timedelta(days=365 * 2)  # 2年後に削除
artifact.save()
```
上記のコード例では、TTL ポリシーを 2 年に設定しています。
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. W&B App UI で W&B の Project に移動します。
2. 左側の パネル で Artifact アイコンを選択します。
3. アーティファクトのリストから、編集したいアーティファクトのタイプを展開します。
4. TTL ポリシーを編集したいアーティファクトのバージョンを選択します。
5. **Version** タブをクリックします。
6. ドロップダウンから **Edit TTL policy** を選択します。
7. 表示されるモーダル内で、TTL ポリシーのドロップダウンから **Custom** を選択します。
8. **TTL duration** フィールド内で、TTL ポリシーを日数単位で設定します。
9. **Update TTL** ボタンを選択して変更を保存します。
{{< img src="/images/artifacts/edit_ttl_ui.gif" alt="Editing TTL policy" >}}
  {{% /tab %}}
{{< /tabpane >}}

### チームのデフォルト TTL ポリシーを設定する
{{% alert %}}
チームの admin のみが、チームのデフォルト TTL ポリシーを設定できます。
{{% /alert %}}
チームのデフォルト TTL ポリシーを設定します。デフォルト TTL ポリシーは、既存および将来のすべてのアーティファクトに、それぞれの作成日に基づいて適用されます。既存のバージョンレベルの TTL ポリシーを持つアーティファクトは、チームのデフォルト TTL の影響を受けません。
1. チームのプロフィールページに移動します。
2. **Settings** タブを選択します。
3. **Artifacts time-to-live (TTL) セクション**に移動します。
4. **Set team's default TTL policy** をクリックします。
5. **Duration** フィールド内で、TTL ポリシーを日数単位で設定します。
6. **Review and save settings** をクリックします。
7. 変更を確認し、**Save settings** を選択します。
{{< img src="/images/artifacts/set_default_ttl.gif" alt="Setting default TTL policy" >}}

### run の外部で TTL ポリシーを設定する
パブリック API を使用して、run をフェッチせずにアーティファクトを取得し、TTL ポリシーを設定します。TTL ポリシーは通常、日数で定義されます。
以下のコードサンプルは、パブリック API を使用してアーティファクトをフェッチし、TTL ポリシーを設定する方法を示しています。
```python
api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

artifact.ttl = timedelta(days=365)  # 1年後に削除

artifact.save()
```

## TTL ポリシーを非アクティブ化する
W&B Python SDK または W&B App UI を使用して、特定のアーティファクトバージョンに対する TTL ポリシーを非アクティブ化します。
{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
1. [アーティファクトを取得]({{< relref path="../download-and-use-an-artifact.md" lang="ja" >}})します。
2. アーティファクトの `ttl` 属性を `None` に設定します。
3. [`save`]({{< relref path="/ref/python/sdk/classes/run.md#save" lang="ja" >}})メソッドでアーティファクトを更新します。
以下のコードスニペットは、アーティファクトの TTL ポリシーをオフにする方法を示しています。
```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = None
artifact.save()
```
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. W&B App UI で W&B の Project に移動します。
2. 左側の パネル で Artifact アイコンを選択します。
3. アーティファクトのリストから、編集したいアーティファクトのタイプを展開します。
4. TTL ポリシーを編集したいアーティファクトのバージョンを選択します。
5. Version タブをクリックします。
6. **Link to registry** ボタンの隣にあるミートボール UI アイコンをクリックします。
7. ドロップダウンから **Edit TTL policy** を選択します。
8. 表示されるモーダル内で、TTL ポリシーのドロップダウンから **Deactivate** を選択します。
9. **Update TTL** ボタンを選択して変更を保存します。
{{< img src="/images/artifacts/remove_ttl_polilcy.gif" alt="Removing TTL policy" >}}
  {{% /tab %}}
{{< /tabpane >}}

## TTL ポリシーを表示する
Python SDK または W&B App UI を使用して、アーティファクトの TTL ポリシーを表示します。
{{< tabpane text=true >}}
  {{% tab  header="Python SDK" %}}
print 文を使用してアーティファクトの TTL ポリシーを表示します。次の例は、アーティファクトを取得してその TTL ポリシーを表示する方法を示しています。
```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
print(artifact.ttl)
```
  {{% /tab %}}
  {{% tab  header="W&B App" %}}
W&B App UI でアーティファクトの TTL ポリシーを表示します。
1. [W&B App](https://wandb.ai) に移動します。
2. W&B の Project に移動します。
3. Project 内で、左側のサイドバーにある Artifacts タブを選択します。
4. コレクションをクリックします。
コレクションビューでは、選択したコレクション内のすべてのアーティファクトを確認できます。`Time to Live` 列には、そのアーティファクトに割り当てられた TTL ポリシーが表示されます。
{{< img src="/images/artifacts/ttl_collection_panel_ui.png" alt="TTL collection view" >}}
  {{% /tab %}}
{{< /tabpane >}}