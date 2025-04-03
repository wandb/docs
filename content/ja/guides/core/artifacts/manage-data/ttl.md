---
title: Manage artifact data retention
description: Time to live ポリシー (TTL)
menu:
  default:
    identifier: ja-guides-core-artifacts-manage-data-ttl
    parent: manage-data
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/kas-artifacts-ttl-colab/colabs/wandb-artifacts/WandB_Artifacts_Time_to_live_TTL_Walkthrough.ipynb" >}}

W&B Artifact time-to-live（TTL）ポリシーを使用して、Artifacts が W&B から削除されるタイミングをスケジュールします。アーティファクトを削除すると、W&B はそのアーティファクトを _ソフト削除_ としてマークします。つまり、アーティファクトは削除対象としてマークされますが、ファイルはストレージからすぐに削除されるわけではありません。W&B が Artifacts を削除する方法の詳細については、[Artifacts の削除]({{< relref path="./delete-artifacts.md" lang="ja" >}}) ページを参照してください。

W&B アプリで Artifacts TTL を使用してデータ保持を管理する方法については、[こちらの](https://www.youtube.com/watch?v=hQ9J6BoVmnc)ビデオチュートリアルをご覧ください。

{{% alert %}}
W&B は、モデルレジストリにリンクされたモデル Artifacts の TTL ポリシーを設定するオプションを無効にします。これは、リンクされたモデルがプロダクションワークフローで使用されている場合に誤って期限切れにならないようにするためです。
{{% /alert %}}
{{% alert %}}
* チーム管理者のみが [チームの設定]({{< relref path="/guides/models/app/settings-page/team-settings.md" lang="ja" >}})を表示し、（1）TTL ポリシーを設定または編集できるユーザーを許可するか、（2）チームのデフォルト TTL を設定するなど、チームレベルの TTL 設定にアクセスできます。
* W&B アプリ UI のアーティファクトの詳細で TTL ポリシーを設定または編集するオプションが表示されない場合、またはプログラムで TTL を設定してもアーティファクトの TTL プロパティが正常に変更されない場合は、チーム管理者がそのための権限を付与していません。
{{% /alert %}}

## 自動生成された Artifacts
ユーザーが生成した Artifacts のみ、TTL ポリシーを使用できます。W&B によって自動生成された Artifacts は、TTL ポリシーを設定できません。

次の Artifact の種類は、自動生成された Artifact を示します。
- `run_table`
- `code`
- `job`
- `wandb-*` で始まる任意の Artifact の種類

[W&B プラットフォーム]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ja" >}})で、またはプログラムで Artifact の種類を確認できます。

```python
import wandb

run = wandb.init(project="<my-project-name>")
artifact = run.use_artifact(artifact_or_name="<my-artifact-name>")
print(artifact.type)
```

`<>` で囲まれた値を独自の値に置き換えます。

## TTL ポリシーを編集および設定できるユーザーを定義する
チーム内で TTL ポリシーを設定および編集できるユーザーを定義します。TTL 権限をチーム管理者のみに付与するか、チーム管理者とチームメンバーの両方に TTL 権限を付与できます。

{{% alert %}}
チーム管理者のみが、TTL ポリシーを設定または編集できるユーザーを定義できます。
{{% /alert %}}

1. チームのプロファイルページに移動します。
2. [**設定**] タブを選択します。
3. [**Artifacts time-to-live (TTL) section**] に移動します。
4. [**TTL permissions dropdown**] で、TTL ポリシーを設定および編集できるユーザーを選択します。
5. [**Review and save settings**] をクリックします。
6. 変更を確認し、[**Save settings**] を選択します。

{{< img src="/images/artifacts/define_who_sets_ttl.gif" alt="" >}}

## TTL ポリシーを作成する
アーティファクトを作成するとき、またはアーティファクトの作成後に遡及的に、アーティファクトの TTL ポリシーを設定します。

以下のすべてのコードスニペットについて、`<>` で囲まれたコンテンツを自分の情報に置き換えて、コードスニペットを使用します。

### アーティファクトを作成するときに TTL ポリシーを設定する
W&B Python SDK を使用して、アーティファクトを作成するときに TTL ポリシーを定義します。TTL ポリシーは通常、日数で定義されます。

{{% alert %}}
アーティファクトを作成するときに TTL ポリシーを定義することは、通常[アーティファクトを作成]({{< relref path="../construct-an-artifact.md" lang="ja" >}})する方法と似ています。ただし、アーティファクトの `ttl` 属性に時間デルタを渡す点が異なります。
{{% /alert %}}

手順は次のとおりです。

1. [アーティファクトを作成]({{< relref path="../construct-an-artifact.md" lang="ja" >}})します。
2. ファイル、ディレクトリー、参照など、[アーティファクトにコンテンツを追加]({{< relref path="../construct-an-artifact.md#add-files-to-an-artifact" lang="ja" >}})します。
3. Python の標準ライブラリの一部である [`datetime.timedelta`](https://docs.python.org/3/library/datetime.html) データ型を使用して、TTL 制限時間を定義します。
4. [アーティファクトをログに記録]({{< relref path="../construct-an-artifact.md#3-save-your-artifact-to-the-wb-server" lang="ja" >}})します。

次のコードスニペットは、アーティファクトを作成し、TTL ポリシーを設定する方法を示しています。

```python
import wandb
from datetime import timedelta

run = wandb.init(project="<my-project-name>", entity="<my-entity>")
artifact = wandb.Artifact(name="<artifact-name>", type="<type>")
artifact.add_file("<my_file>")

artifact.ttl = timedelta(days=30)  # TTL ポリシーを設定
run.log_artifact(artifact)
```

上記のコードスニペットは、アーティファクトの TTL ポリシーを 30 日に設定します。つまり、W&B は 30 日後にアーティファクトを削除します。

### アーティファクトを作成した後で TTL ポリシーを設定または編集する
W&B アプリ UI または W&B Python SDK を使用して、既に存在するアーティファクトの TTL ポリシーを定義します。

{{% alert %}}
アーティファクトの TTL を変更すると、アーティファクトが期限切れになるまでの時間は、アーティファクトの `createdAt` タイムスタンプを使用して計算されます。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
1. [アーティファクトを取得]({{< relref path="../download-and-use-an-artifact.md" lang="ja" >}})します。
2. 時間デルタをアーティファクトの `ttl` 属性に渡します。
3. [`save`]({{< relref path="/ref/python/run.md#save" lang="ja" >}})メソッドを使用してアーティファクトを更新します。


次のコードスニペットは、アーティファクトの TTL ポリシーを設定する方法を示しています。
```python
import wandb
from datetime import timedelta

artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = timedelta(days=365 * 2)  # 2 年後に削除
artifact.save()
```

上記のコード例では、TTL ポリシーを 2 年に設定しています。
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. W&B アプリ UI で W&B プロジェクトに移動します。
2. 左側のパネルでアーティファクトアイコンを選択します。
3. アーティファクトのリストから、TTL ポリシーを編集するアーティファクトの種類を展開します。
4. TTL ポリシーを編集するアーティファクトバージョンを選択します。
5. [**バージョン**] タブをクリックします。
6. ドロップダウンから [**TTL ポリシーの編集**] を選択します。
7. 表示されるモーダル内で、TTL ポリシードロップダウンから [**カスタム**] を選択します。
8. [**TTL 期間**] フィールドに、TTL ポリシーを日数単位で設定します。
9. [**TTL の更新**] ボタンを選択して、変更を保存します。

{{< img src="/images/artifacts/edit_ttl_ui.gif" alt="" >}}
  {{% /tab %}}
{{< /tabpane >}}

### チームのデフォルト TTL ポリシーを設定する

{{% alert %}}
チーム管理者のみが、チームのデフォルト TTL ポリシーを設定できます。
{{% /alert %}}

チームのデフォルト TTL ポリシーを設定します。デフォルト TTL ポリシーは、それぞれの作成日に基づいて、既存および将来のすべてのアーティファクトに適用されます。既存のバージョンレベルの TTL ポリシーを持つアーティファクトは、チームのデフォルト TTL の影響を受けません。

1. チームのプロファイルページに移動します。
2. [**設定**] タブを選択します。
3. [**Artifacts time-to-live (TTL) section**] に移動します。
4. [**チームのデフォルト TTL ポリシーを設定**] をクリックします。
5. [**期間**] フィールドに、TTL ポリシーを日数単位で設定します。
6. [**Review and save settings**] をクリックします。
7. 変更を確認し、[**Save settings**] を選択します。

{{< img src="/images/artifacts/set_default_ttl.gif" alt="" >}}

### run の外部で TTL ポリシーを設定する

パブリック API を使用して、run を取得せずにアーティファクトを取得し、TTL ポリシーを設定します。TTL ポリシーは通常、日数で定義されます。

次のコードサンプルは、パブリック API を使用してアーティファクトを取得し、TTL ポリシーを設定する方法を示しています。

```python
api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

artifact.ttl = timedelta(days=365)  # 1 年後に削除

artifact.save()
```

## TTL ポリシーを無効化する
W&B Python SDK または W&B アプリ UI を使用して、特定のアーティファクトバージョンの TTL ポリシーを無効化します。

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
1. [アーティファクトを取得]({{< relref path="../download-and-use-an-artifact.md" lang="ja" >}})します。
2. アーティファクトの `ttl` 属性を `None` に設定します。
3. [`save`]({{< relref path="/ref/python/run.md#save" lang="ja" >}})メソッドを使用してアーティファクトを更新します。


次のコードスニペットは、アーティファクトの TTL ポリシーをオフにする方法を示しています。
```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = None
artifact.save()
```
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. W&B アプリ UI で W&B プロジェクトに移動します。
2. 左側のパネルでアーティファクトアイコンを選択します。
3. アーティファクトのリストから、TTL ポリシーを編集するアーティファクトの種類を展開します。
4. TTL ポリシーを編集するアーティファクトバージョンを選択します。
5. [バージョン] タブをクリックします。
6. [**レジストリへのリンク**] ボタンの横にあるミートボール UI アイコンをクリックします。
7. ドロップダウンから [**TTL ポリシーの編集**] を選択します。
8. 表示されるモーダル内で、TTL ポリシードロップダウンから [**非アクティブ化**] を選択します。
9. [**TTL の更新**] ボタンを選択して、変更を保存します。

{{< img src="/images/artifacts/remove_ttl_polilcy.gif" alt="" >}}
  {{% /tab %}}
{{< /tabpane >}}

## TTL ポリシーを表示する
Python SDK または W&B アプリ UI でアーティファクトの TTL ポリシーを表示します。

{{< tabpane text=true >}}
  {{% tab  header="Python SDK" %}}
print ステートメントを使用して、アーティファクトの TTL ポリシーを表示します。次の例は、アーティファクトを取得し、その TTL ポリシーを表示する方法を示しています。

```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
print(artifact.ttl)
```
  {{% /tab %}}
  {{% tab  header="W&B App" %}}
W&B アプリ UI でアーティファクトの TTL ポリシーを表示します。

1. [https://wandb.ai](https://wandb.ai) で W&B アプリに移動します。
2. W&B プロジェクトに移動します。
3. プロジェクト内で、左側のサイドバーにある [Artifacts] タブを選択します。
4. コレクションをクリックします。

コレクションビュー内では、選択したコレクション内のすべてのアーティファクトを表示できます。[`Time to Live`] 列には、そのアーティファクトに割り当てられた TTL ポリシーが表示されます。

{{< img src="/images/artifacts/ttl_collection_panel_ui.png" alt="" >}}
  {{% /tab %}}
{{< /tabpane >}}
