---
title: アーティファクトデータの保持期間を管理する
description: タイムトゥリブ（TTL）ポリシー
menu:
  default:
    identifier: ja-guides-core-artifacts-manage-data-ttl
    parent: manage-data
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/kas-artifacts-ttl-colab/colabs/wandb-artifacts/WandB_Artifacts_Time_to_live_TTL_Walkthrough.ipynb" >}}

W&B の Artifacts タイムトゥリブ（TTL）ポリシーを使って、Artifacts の削除タイミングをスケジューリングできます。Artifacts を削除すると、W&B はその Artifact を*ソフトデリート*としてマークします。つまり、Artifact は削除対象になりますが、ファイルはすぐにはストレージから消去されません。W&B の Artifact 削除の仕組みについては、[Artifacts の削除]({{< relref path="./delete-artifacts.md" lang="ja" >}}) ページをご覧ください。

[Managing data retention with Artifacts TTL](https://www.youtube.com/watch?v=hQ9J6BoVmnc) のビデオチュートリアルで、W&B アプリで Artifacts TTL を使ったデータ保持管理方法が学べます。

{{% alert %}}
モデルレジストリにリンクされているモデル Artifact については、TTL ポリシーを設定するオプションが無効化されています。これは、プロダクションワークフローで使用中のモデルが誤って期限切れにならないようにするためです。
{{% /alert %}}
{{% alert %}}
* チームレベルの TTL 設定（(1) TTL ポリシーの設定や編集を許可する対象、(2) チームのデフォルト TTL 設定など）を閲覧・変更できるのは、チーム管理者のみです。詳細は[チームの設定]({{< relref path="/guides/models/app/settings-page/team-settings.md" lang="ja" >}}) を参照してください。
* W&B アプリ UI で Artifact 詳細に TTL ポリシーの設定・編集オプションが表示されない、またはプログラムから TTL を設定しても反映されない場合、チーム管理者によって権限が付与されていない可能性があります。
{{% /alert %}}

## 自動生成された Artifacts
TTL ポリシーはユーザーが生成した Artifact のみ利用できます。W&B によって自動生成された Artifact には TTL ポリシーは設定できません。

次の Artifact タイプは自動生成 Artifact を示します：
- `run_table`
- `code`
- `job`
- `wandb-*` で始まるすべての Artifact タイプ

Artifact のタイプは[W&B プラットフォーム]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ja" >}}) で確認するか、プログラムから取得できます：

```python
import wandb

run = wandb.init(project="<my-project-name>")
artifact = run.use_artifact(artifact_or_name="<my-artifact-name>")
print(artifact.type)
```

`<>` で囲まれた値はご自身の情報に置き換えてください。

## TTL ポリシーの設定・編集権限の指定
チーム内で誰が TTL ポリシーを設定・編集できるかを定義します。TTL 権限をチーム管理者のみに絞ることも、チーム管理者およびチームメンバーの両方に付与することも可能です。

{{% alert %}}
TTL ポリシーの設定・編集権限を指定できるのはチーム管理者のみです。
{{% /alert %}}

1. チームのプロフィールページへ移動します。
2. **Settings** タブを選択します。
3. **Artifacts time-to-live (TTL) セクション**に移動します。
4. **TTL permissions ドロップダウン**で、TTL ポリシーの設定・編集を許可する対象を選びます。
5. **Review and save settings** をクリックします。
6. 変更を確認し、**Save settings** を選択します。

{{< img src="/images/artifacts/define_who_sets_ttl.gif" alt="TTL 権限の設定" >}}

## TTL ポリシーの作成
Artifact 作成時、または作成後に TTL ポリシーを設定できます。

以下のすべてのコードスニペットでは、`<>` で囲まれた内容をご自身の情報に置き換えてご利用ください。

### Artifact 作成時に TTL ポリシーを設定する
Artifact 作成時に TTL ポリシーを設定するには、W&B Python SDK を使用します。TTL ポリシーは通常、日数で指定します。

{{% alert %}}
Artifact 作成時に TTL ポリシーを設定する方法は、[Artifact の作成]({{< relref path="../construct-an-artifact.md" lang="ja" >}}) と基本的に同じですが、`ttl` 属性に日数を渡す点のみ異なります。
{{% /alert %}}

手順は以下の通りです：

1. [Artifact を作成]({{< relref path="../construct-an-artifact.md" lang="ja" >}})
2. [Artifact にファイルやディレクトリ、参照などを追加]({{< relref path="../construct-an-artifact.md#add-files-to-an-artifact" lang="ja" >}})
3. Python 標準ライブラリの [`datetime.timedelta`](https://docs.python.org/3/library/datetime.html) を使って TTL の期間を設定
4. [Artifact をログする]({{< relref path="../construct-an-artifact.md#3-save-your-artifact-to-the-wb-server" lang="ja" >}})

以下のコードスニペットは、Artifact 作成時に TTL ポリシーを設定する例です。

```python
import wandb
from datetime import timedelta

run = wandb.init(project="<my-project-name>", entity="<my-entity>")
artifact = wandb.Artifact(name="<artifact-name>", type="<type>")
artifact.add_file("<my_file>")

artifact.ttl = timedelta(days=30)  # TTL ポリシーを設定（30日）
run.log_artifact(artifact)
```

上記コード例では、Artifact の TTL ポリシーを 30 日に設定しています。つまり、W&B は 30 日後にこの Artifact を削除します。

### Artifact 作成後に TTL ポリシーを設定・編集する
すでに作成された Artifact には、W&B App UI か W&B Python SDK を使って TTL ポリシーを定義できます。

{{% alert %}}
Artifact の TTL を変更しても、Artifact の有効期限は `createdAt` タイムスタンプから計算されます。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
1. [Artifact を取得]({{< relref path="../download-and-use-an-artifact.md" lang="ja" >}})
2. `ttl` 属性に期間を渡します
3. [`save`]({{< relref path="/ref/python/sdk/classes/run.md#save" lang="ja" >}}) メソッドで反映させます

以下は TTL ポリシーを設定する例です：
```python
import wandb
from datetime import timedelta

artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = timedelta(days=365 * 2)  # 2年後に削除
artifact.save()
```

この例では TTL ポリシーを 2 年に設定しています。
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. W&B App UI でプロジェクトにアクセスします
2. 左パネルから Artifact アイコンを選択します
3. Artifact の一覧から、対象の Artifact タイプを展開します
4. TTL ポリシーを編集したい Artifact バージョンを選択します
5. **Version** タブをクリック
6. ドロップダウンから **Edit TTL policy** を選択
7. 表示されたモーダルで TTL ポリシーのドロップダウンから **Custom** を選択
8. **TTL duration** フィールドで日数を指定
9. **Update TTL** ボタンで変更を保存

{{< img src="/images/artifacts/edit_ttl_ui.gif" alt="TTL ポリシーの編集" >}}
  {{% /tab %}}
{{< /tabpane >}}

### チームのデフォルト TTL ポリシー設定

{{% alert %}}
チームのデフォルト TTL ポリシーはチーム管理者のみが設定できます。
{{% /alert %}}

チームのデフォルト TTL ポリシーは、作成日を基準にすべての既存および今後作成される Artifact に適用されます。すでにバージョン単位で TTL が設定されている Artifact には影響しません。

1. チームのプロフィールページへ移動します。
2. **Settings** タブを選択します。
3. **Artifacts time-to-live (TTL) セクション**に移動します。
4. **Set team's default TTL policy** をクリック
5. **Duration** フィールドで日数単位の TTL ポリシーを設定
6. **Review and save settings** をクリック
7. 内容確認の上、**Save settings** を選択

{{< img src="/images/artifacts/set_default_ttl.gif" alt="デフォルトTTLポリシーの設定" >}}

### run 外での TTL ポリシー設定

Public API を使って run を取得せずに Artifact を取得し、TTL ポリシーを設定できます。TTL は通常日数で指定します。

以下の例は、Public API で Artifact を取得し TTL ポリシーを設定する方法です。

```python 
api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

artifact.ttl = timedelta(days=365)  # 1年後に削除

artifact.save()
```

## TTL ポリシーの無効化
特定の Artifact バージョンの TTL ポリシーを、W&B Python SDK または App UI から無効化できます。

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
1. [Artifact を取得]({{< relref path="../download-and-use-an-artifact.md" lang="ja" >}})
2. `ttl` 属性を `None` に設定します
3. [`save`]({{< relref path="/ref/python/sdk/classes/run.md#save" lang="ja" >}}) メソッドで更新します

TTL ポリシーを無効化する例です：
```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = None
artifact.save()
```  
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. W&B App UI でプロジェクトにアクセスします
2. 左パネルから Artifact アイコンを選択
3. Artifact の一覧から、対象の Artifact タイプを展開します
4. TTL ポリシーを編集したい Artifact バージョンを選択
5. Version タブをクリック
6. **Link to registry** ボタン横のメニュー（肉球UI）をクリック
7. ドロップダウンから **Edit TTL policy** を選択
8. 表示されたモーダルで TTL ポリシードロップダウンから **Deactivate** を選択
9. **Update TTL** ボタンで変更を保存

{{< img src="/images/artifacts/remove_ttl_polilcy.gif" alt="TTL ポリシーの無効化" >}}  
  {{% /tab %}}
{{< /tabpane >}}

## TTL ポリシーの確認
Artifacts の TTL ポリシーは、Python SDK または W&B App UI で確認できます。

{{< tabpane text=true >}}
  {{% tab  header="Python SDK" %}}
print 文で Artifact の TTL ポリシーを確認できます。以下は Artifact を取得して TTL ポリシーを表示する例です：

```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
print(artifact.ttl)
```  
  {{% /tab %}}
  {{% tab  header="W&B App" %}}
W&B App UI で Artifact の TTL ポリシーを確認できます。

1. [W&B App](https://wandb.ai) にアクセス
2. W&B プロジェクトを開く
3. プロジェクト内の左サイドバーで Artifacts タブを選択
4. コレクションをクリック

コレクションビューでは、選択したコレクション内のすべての Artifact が表示されます。`Time to Live` 列で各 Artifact の TTL ポリシーが確認できます。

{{< img src="/images/artifacts/ttl_collection_panel_ui.png" alt="TTL コレクションビュー" >}}  
  {{% /tab %}}
{{< /tabpane >}}