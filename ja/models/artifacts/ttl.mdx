---
title: アーティファクトデータ保持の管理
description: 存続期間 (TTL) ポリシー
menu:
  default:
    identifier: ja-guides-core-artifacts-manage-data-ttl
    parent: manage-data
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/kas-artifacts-ttl-colab/colabs/wandb-artifacts/WandB_Artifacts_Time_to_live_TTL_Walkthrough.ipynb" >}}

W&B の Artifacts のタイム・トゥ・リブ（TTL）ポリシーを使用して、Artifacts が W&B から削除されるスケジュールを設定します。Artifact を削除すると、W&B はそのアーティファクトを *ソフト削除* としてマークします。つまり、アーティファクトは削除対象としてマークされますが、ファイルはすぐにストレージから削除されるわけではありません。W&B がアーティファクトを削除する方法の詳細については、[アーティファクトを削除](./delete-artifacts.md)ページを参照してください。

W&B アプリで Artifacts TTL を使って データ保持を管理する方法を学ぶには、[この](https://www.youtube.com/watch?v=hQ9J6BoVmnc) ビデオチュートリアルをご覧ください。

{{% alert %}}
W&B は、モデルレジストリにリンクされたモデルアーティファクトの TTL ポリシーを設定するオプションを非アクティブ化します。これは、生産ワークフローで使用されるリンクされたモデルが誤って期限切れにならないようにするためです。
{{% /alert %}}
{{% alert %}}
* チームの設定]({{< relref path="/guides/models/app/settings-page/teams.md" lang="ja" >}})と、(1) TTL ポリシーを設定または編集できる人を許可するか、(2) チームのデフォルト TTL を設定するかなどのチームレベルの TTL 設定は、チーム管理者のみが表示およびアクセスできます。
* W&B アプリ UI のアーティファクトの詳細に TTL ポリシーを設定または編集するオプションが表示されない場合、またはプログラムで TTL を設定してもアーティファクトの TTL プロパティが正常に変更されない場合は、チーム管理者が権限を付与していません。
{{% /alert %}}

## 自動生成された Artifacts
ユーザー生成のアーティファクトのみが TTL ポリシーを使用できます。W&B によって自動生成されたアーティファクトには TTL ポリシーを設定することはできません。

自動生成されたアーティファクトを示すアーティファクトタイプは次のとおりです：
- `run_table`
- `code`
- `job`
- `wandb-*` で始まる種類のアーティファクト

アーティファクトの種類は、[W&B プラットフォーム]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ja" >}})またはプログラムで確認できます：

```python
import wandb

run = wandb.init(project="<my-project-name>")
artifact = run.use_artifact(artifact_or_name="<my-artifact-name>")
print(artifact.type)
```

含まれる `<>` で囲まれた値をあなたのものに置き換えてください。

## TTL ポリシーを編集および設定できる人を定義する
チーム内で TTL ポリシーを設定および編集できる人を定義します。TTL 許可をチーム管理者のみに与えることもできますし、チーム管理者とチームメンバーの両方に TTL 許可を与えることもできます。

{{% alert %}}
TTL ポリシーを設定または編集できる人を定義できるのはチーム管理者だけです。
{{% /alert %}}

1. チームのプロフィールページに移動します。
2. **設定** タブを選択します。
3. **Artifacts のタイム・トゥ・リブ (TTL) セクション**に移動します。
4. **TTL 許可のドロップダウン**から、TTL ポリシーを設定および編集できる人を選択します。  
5. **設定をレビューして保存**をクリックします。
6. 変更を確認し、**設定を保存**を選択します。

{{< img src="/images/artifacts/define_who_sets_ttl.gif" alt="" >}}

## TTL ポリシーを作成する
アーティファクトを作成するとき、または作成後に TTL ポリシーを設定します。

以下のコードスニペットすべてにおいて、 `<>` で包まれた内容をあなたの情報に置き換えてコードスニペットを使用してください。

### アーティファクト作成時に TTL ポリシーを設定する
W&B Python SDK を使用してアーティファクトを作成する際に TTL ポリシーを定義します。TTL ポリシーは通常日数で定義されます。    

{{% alert %}}
アーティファクト作成時に TTL ポリシーを定義することは、通常の[アーティファクトを作成](../construct-an-artifact.md)する方法に似ています。例外は、アーティファクトの `ttl` 属性に時間差を渡す点です。
{{% /alert %}}

手順は次のとおりです：

1. [アーティファクトを作成](../construct-an-artifact.md)します。
2. ファイル、ディレクトリ、または参照など、アーティファクトにコンテンツを[追加](../construct-an-artifact.md#add-files-to-an-artifact)します。
3. Python の標準ライブラリの一部である [`datetime.timedelta`](https://docs.python.org/3/library/datetime.html) データ型で TTL の期限を定義します。
4. [アーティファクトをログ](../construct-an-artifact.md#3-save-your-artifact-to-the-wb-server)します。

以下のコードスニペットはアーティファクトを作成し、TTL ポリシーを設定する方法を示しています。

```python
import wandb
from datetime import timedelta

run = wandb.init(project="<my-project-name>", entity="<my-entity>")
artifact = wandb.Artifact(name="<artifact-name>", type="<type>")
artifact.add_file("<my_file>")

artifact.ttl = timedelta(days=30)  # TTL ポリシーを設定
run.log_artifact(artifact)
```

上記のコードスニペットは、アーティファクトの TTL ポリシーを 30 日間に設定します。つまり、W&B は 30 日後にアーティファクトを削除します。

### アーティファクト作成後に TTL ポリシーを設定または編集する
存在するアーティファクトに対して W&B アプリの UI または W&B Python SDK を使用して TTL ポリシーを定義します。

{{% alert %}}
アーティファクトの TTL を変更する場合、アーティファクトの期限切れまでの時間は、アーティファクトの作成時刻 (`createdAt` タイムスタンプ) を基に計算されます。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
1. [あなたのアーティファクトを取得]({{< relref path="../download-and-use-an-artifact.md" lang="ja" >}})します。
2. アーティファクトの `ttl` 属性に時間差を渡します。
3. [`save`]({{< relref path="/ref/python/run.md#save" lang="ja" >}}) メソッドでアーティファクトを更新します。

以下のコードスニペットは、アーティファクトに TTL ポリシーを設定する方法を示しています：
```python
import wandb
from datetime import timedelta

artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = timedelta(days=365 * 2)  # 2年後に削除
artifact.save()
```

上記のコード例では、TTL ポリシーを2年間に設定します。
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. W&B アプリ UI の W&B プロジェクトに移動します。
2. 左のパネルでアーティファクトアイコンを選択します。
3. アーティファクトの一覧からアーティファクトタイプを展開します。
4. TTL ポリシーを編集したいアーティファクトバージョンを選択します。
5. **バージョン** タブをクリックします。
6. ドロップダウンから **TTL ポリシー編集** を選択します。
7. 表示されるモーダル内で、TTL ポリシードロップダウンから **カスタム** を選択します。
8. **TTL 期間**フィールドで、日数単位で TTL ポリシーを設定します。
9. **TTL 更新** ボタンを選択して変更を保存します。

{{< img src="/images/artifacts/edit_ttl_ui.gif" alt="" >}}  
  {{% /tab %}}
{{< /tabpane >}}

### チームのデフォルト TTL ポリシーを設定する

{{% alert %}}
チームのデフォルト TTL ポリシーを設定できるのはチーム管理者だけです。
{{% /alert %}}

チームのデフォルト TTL ポリシーを設定します。デフォルトの TTL ポリシーは、既存と今後のアーティファクトすべてに、その作成日を基に適用されます。バージョンレベルで既に TTL ポリシーが存在するアーティファクトは、チームのデフォルト TTL に影響を受けません。

1. チームのプロフィールページに移動します。
2. **設定** タブを選択します。
3. **Artifacts のタイム・トゥ・リブ (TTL) セクション**に移動します。
4. **チームのデフォルト TTL ポリシー設定** をクリックします。
5. **期間**フィールドにおいて、日数単位で TTL ポリシーを設定します。
6. **設定をレビューして保存** をクリックします。
7. 変更を確認し、**設定を保存** を選択します。

{{< img src="/images/artifacts/set_default_ttl.gif" alt="" >}}

### run 外で TTL ポリシーを設定する

公開 API を使って run を取得せずにアーティファクトを取得し、TTL ポリシーを設定します。TTL ポリシーは通常日数単位で定義されます。

以下のコードサンプルは、公開 API を使用してアーティファクトを取得し、TTL ポリシーを設定する方法を示しています。

```python 
api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

artifact.ttl = timedelta(days=365)  # 1年後削除

artifact.save()
```

## TTL ポリシーを非アクティブにする
W&B Python SDK または W&B アプリ UI を使用して、特定のアーティファクトバージョンの TTL ポリシーを非アクティブにします。

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
1. [あなたのアーティファクトを取得]({{< relref path="../download-and-use-an-artifact.md" lang="ja" >}})します。
2. アーティファクトの `ttl` 属性を `None` に設定します。
3. [`save`]({{< relref path="/ref/python/run.md#save" lang="ja" >}}) メソッドでアーティファクトを更新します。

以下のコードスニペットは、アーティファクトに対する TTL ポリシーをオフにする方法を示しています：
```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = None
artifact.save()
```  
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. W&B アプリ UI の W&B プロジェクトに移動します。
2. 左パネルでアーティファクトアイコンを選択します。
3. アーティファクトのリストからアーティファクトタイプを展開します。
4. TTL ポリシーを編集したいアーティファクトバージョンを選択します。
5. バージョンタブをクリックします。
6. **リンク先レジストリ** ボタンの隣にある肉球 UI アイコンをクリックします。
7. ドロップダウンから **TTL ポリシー編集** を選択します。
8. 表示されるモーダル内で、TTL ポリシードロップダウンから **非アクティブ** を選択します。
9. 変更を保存するために **TTL 更新** ボタンを選択します。

{{< img src="/images/artifacts/remove_ttl_polilcy.gif" alt="" >}}  
  {{% /tab %}}
{{< /tabpane >}}

## TTL ポリシーを確認する
W&B Python SDK または W&B アプリ UI を使用して、アーティファクトの TTL ポリシーを確認します。

{{< tabpane text=true >}}
  {{% tab  header="Python SDK" %}}
print 文を使用してアーティファクトの TTL ポリシーを表示します。以下の例では、アーティファクトを取得してその TTL ポリシーを表示する方法を示しています：

```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
print(artifact.ttl)
```  
  {{% /tab %}}
  {{% tab  header="W&B App" %}}
W&B アプリ UI を使用してアーティファクトの TTL ポリシーを表示します。

1. W&B アプリの [https://wandb.ai](https://wandb.ai) に移動します。
2. あなたの W&B プロジェクトに移動します。
3. プロジェクト内で、左のサイドバーの Artifacts タブを選択します。
4. コレクションをクリックします。

選択されたコレクション内のすべてのアーティファクトが表示されます。`Time to Live` 列にそのアーティファクトに割り当てられた TTL ポリシーが表示されます。

{{< img src="/images/artifacts/ttl_collection_panel_ui.png" alt="" >}}  
  {{% /tab %}}
{{< /tabpane >}}