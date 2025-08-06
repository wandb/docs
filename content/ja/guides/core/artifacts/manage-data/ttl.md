---
title: アーティファクト データ保持を管理する
description: Time to live ポリシー（TTL）
menu:
  default:
    identifier: ttl
    parent: manage-data
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/kas-artifacts-ttl-colab/colabs/wandb-artifacts/WandB_Artifacts_Time_to_live_TTL_Walkthrough.ipynb" >}}

W&B の Artifacts の削除タイミングをスケジュールするには、W&B Artifacts の time-to-live（TTL）ポリシーを設定します。Artifact を削除すると、W&B はその Artifact を *ソフトデリート* としてマークします。つまり、その artifact は削除予定としてマークされますが、ファイル自体はすぐにストレージから削除されるわけではありません。W&B が artifact をどのように削除するかの詳細は、[Delete artifacts]({{< relref "./delete-artifacts.md" >}}) ページをご覧ください。

[Managing data retention with Artifacts TTL](https://www.youtube.com/watch?v=hQ9J6BoVmnc) のビデオチュートリアルもご覧いただくと、W&B アプリで Artifacts TTL を使ったデータ保持管理方法が学べます。

{{% alert %}}
W&B では、モデルアーティファクトが Model Registry にリンクされている場合、TTL ポリシーを設定するオプションが無効化されます。これは、プロダクション ワークフローで使用されているリンク済みモデルが誤って期限切れにならないようにするためです。
{{% /alert %}}
{{% alert %}}
* チームの [settings]({{< relref "/guides/models/app/settings-page/team-settings.md" >}}) や、TTL ポリシーの設定・編集権限の管理、チームデフォルト TTL の設定などのチーム単位の TTL 設定は、管理者のみが閲覧・変更できます。  
* W&B App UI で artifact 詳細画面に TTL 設定・編集のオプションが表示されない、またはプログラム上で TTL を設定しても反映されない場合、管理者から適切な権限が付与されていない可能性があります。
{{% /alert %}}

## 自動生成 Artifacts
TTL ポリシーはユーザーが作成した artifact のみ利用可能です。W&B により自動生成された artifact には TTL ポリシーを設定できません。

以下の artifact タイプは自動生成された artifact であることを示します：
- `run_table`
- `code`
- `job`
- `wandb-*` で始まる artifact タイプ全て

artifact のタイプは [W&B プラットフォーム]({{< relref "/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" >}}) でも、プログラムからも確認できます。

```python
import wandb

run = wandb.init(project="<my-project-name>")
artifact = run.use_artifact(artifact_or_name="<my-artifact-name>")
print(artifact.type)
```

`<>` で囲まれている部分はご自身の情報に置き換えてください。

## TTL ポリシーの編集・設定を許可するユーザーを定義する
チーム内で誰が TTL ポリシーを設定・編集できるかを決められます。TTL 設定権限をチーム管理者だけにすることも、管理者だけでなくチームメンバーにも付与することも可能です。

{{% alert %}}
TTL ポリシーの設定・編集権限は管理者だけが設定できます。
{{% /alert %}}

1. チームのプロフィールページにアクセスします。
2. **Settings** タブを選択します。
3. **Artifacts time-to-live (TTL) セクション**に進みます。
4. **TTL permissions ドロップダウン**から、誰が TTL ポリシーを設定・編集できるか選択します。  
5. **Review and save settings** をクリックします。
6. 内容を確認し **Save settings** を選択します。

{{< img src="/images/artifacts/define_who_sets_ttl.gif" alt="Setting TTL permissions" >}}

## TTL ポリシーの作成
Artifact 作成時、あるいは作成後に TTL ポリシーを設定できます。

以下のコードスニペットでは、`<>` で囲まれている部分はご自身の情報に置き換えてご利用ください。

### Artifact 作成時に TTL ポリシーを設定する
W&B Python SDK を使って artifact 作成時に TTL ポリシーを設定することができます。TTL は通常「日単位」で指定します。

{{% alert %}}
Artifact 作成時に TTL ポリシーを設定する方法は、[artifact の作成]({{< relref "../construct-an-artifact.md" >}})方法とほぼ同じです。異なるのは、artifact の `ttl` 属性にタイムデルタを指定する点だけです。
{{% /alert %}}

手順は以下の通りです。

1. [Artifact を作成]({{< relref "../construct-an-artifact.md" >}})します。
2. ファイルやディレクトリ、リファレンスなど、[artifact にコンテンツを追加]({{< relref "../construct-an-artifact.md#add-files-to-an-artifact" >}})します。
3. Python 標準ライブラリの [`datetime.timedelta`](https://docs.python.org/3/library/datetime.html) で TTL の期間を定義します。
4. [artifact をログ]({{< relref "../construct-an-artifact.md#3-save-your-artifact-to-the-wb-server" >}})します。

以下のコードスニペットは、artifact 作成時に TTL ポリシーを設定する例です。

```python
import wandb
from datetime import timedelta

run = wandb.init(project="<my-project-name>", entity="<my-entity>")
artifact = wandb.Artifact(name="<artifact-name>", type="<type>")
artifact.add_file("<my_file>")

artifact.ttl = timedelta(days=30)  # TTLポリシーを30日に設定
run.log_artifact(artifact)
```

上記コード例では、artifact の TTL ポリシーを 30 日に設定しています。つまり、30 日後に W&B がその artifact を削除します。

### Artifact 作成後に TTL ポリシーを設定・編集する
作成済みの artifact に対しても W&B App UI または W&B Python SDK から TTL ポリシーを設定・編集できます。

{{% alert %}}
Artifact の TTL を変更しても、有効期限の計算は artifact の `createdAt` タイムスタンプからの経過時間を元に行われます。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
1. [artifact を取得]({{< relref "../download-and-use-an-artifact.md" >}})します。
2. artifact の `ttl` 属性にタイムデルタを渡します。
3. [`save`]({{< relref "/ref/python/sdk/classes/run.md#save" >}}) メソッドで artifact を更新します。

以下は TTL ポリシーを設定するコード例です:
```python
import wandb
from datetime import timedelta

artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = timedelta(days=365 * 2)  # 2年後に削除
artifact.save()
```

上記のコード例では TTL ポリシーが2年に設定されます。
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. W&B App UI で W&B プロジェクトに移動します。
2. 左パネルの artifact アイコンを選択します。
3. artifact 一覧から、対象の artifact タイプを展開します。
4. TTL ポリシーを編集したい artifact の version を選択します。
5. **Version** タブをクリックします。
6. ドロップダウンから **Edit TTL policy** を選択します。
7. 表示されたモーダルで、TTL ポリシードロップダウンから **Custom** を選択します。
8. **TTL duration** フィールドで日数単位の TTL を入力します。
9. **Update TTL** ボタンで変更を保存します。

{{< img src="/images/artifacts/edit_ttl_ui.gif" alt="Editing TTL policy" >}}  
  {{% /tab %}}
{{< /tabpane >}}



### チームのデフォルト TTL ポリシーを設定する

{{% alert %}}
チームのデフォルト TTL ポリシーは管理者のみが設定できます。
{{% /alert %}}

チーム全体でデフォルトの TTL ポリシーを設定できます。デフォルト TTL ポリシーは、作成済みおよび今後作成される artifact へ、作成日を基準に適用されます。既にバージョン単位で TTL ポリシーが設定されている artifact には影響しません。

1. チームのプロフィールページにアクセスします。
2. **Settings** タブを選択します。
3. **Artifacts time-to-live (TTL) セクション**に進みます。
4. **Set team's default TTL policy** をクリックします。
5. **Duration** フィールドで日数単位の TTL を設定します。
6. **Review and save settings** をクリックします。
7. 内容を確認し **Save settings** を選択します。

{{< img src="/images/artifacts/set_default_ttl.gif" alt="Setting default TTL policy" >}}

### run 外で TTL ポリシーを設定する

Public API を使えば、run を明示的に取得しなくても artifact を取得し、TTL ポリシーを設定できます。TTL ポリシーは通常「日単位」で指定します。

以下のサンプルは Public API で artifact を取得し、TTL ポリシーを設定する例です。

```python 
api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

artifact.ttl = timedelta(days=365)  # 1年後に削除

artifact.save()
```

## TTL ポリシーの無効化
W&B Python SDK または W&B App UI を使って、特定の artifact バージョンの TTL ポリシーを無効にできます。



{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
1. [artifact を取得]({{< relref "../download-and-use-an-artifact.md" >}})します。
2. artifact の `ttl` 属性を `None` に設定します。
3. [`save`]({{< relref "/ref/python/sdk/classes/run.md#save" >}}) メソッドで artifact を更新します。

以下は TTL ポリシーを無効にするコード例です:
```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = None
artifact.save()
```  
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. W&B App UI で W&B プロジェクトに移動します。
2. 左パネルの artifact アイコンを選択します。
3. artifact 一覧から、対象の artifact タイプを展開します。
4. TTL ポリシーを編集したい artifact の version を選択します。
5. Version タブをクリックします。
6. **Link to registry** ボタンの横にある「三点リーダ（⋮）」UI アイコンをクリックします。
7. ドロップダウンから **Edit TTL policy** を選択します。
8. モーダルで TTL ポリシードロップダウンから **Deactivate** を選択します。
9. **Update TTL** ボタンで変更を保存します。

{{< img src="/images/artifacts/remove_ttl_polilcy.gif" alt="Removing TTL policy" >}}  
  {{% /tab %}}
{{< /tabpane >}}




## TTL ポリシーの確認
Python SDK または W&B App UI で artifact の TTL ポリシーを確認できます。

{{< tabpane text=true >}}
  {{% tab  header="Python SDK" %}}
print 文を使って artifact の TTL ポリシーを確認できます。下記は artifact を取得して TTL を表示する例です。

```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
print(artifact.ttl)
```  
  {{% /tab %}}
  {{% tab  header="W&B App" %}}
W&B App UI から artifact の TTL ポリシーを確認できます。

1. [W&B App](https://wandb.ai) にアクセスします。
2. ご自身の W&B Project へ移動します。
3. プロジェクト内で左サイドバーの Artifacts タブを選択します。
4. コレクションをクリックします。

コレクションビューには、選択したコレクション内の全ての artifact が表示されます。`Time to Live` 列に各 artifact に設定されている TTL ポリシーが表示されます。

{{< img src="/images/artifacts/ttl_collection_panel_ui.png" alt="TTL collection view" >}}  
  {{% /tab %}}
{{< /tabpane >}}