---
description: 生存期間ポリシー（TTL）
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# Manage data retention with Artifact TTL policy

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/kas-artifacts-ttl-colab/colabs/wandb-artifacts/WandB_Artifacts_Time_to_live_TTL_Walkthrough.ipynb"/>

W&BのArtifactの有効期間（TTL）ポリシーを使用して、アーティファクトがW&Bから削除されるスケジュールを設定します。アーティファクトを削除すると、W&Bはそのアーティファクトを*ソフトデリート*としてマークします。つまり、アーティファクトは削除対象としてマークされますが、ファイルがストレージからすぐに削除されるわけではありません。W&Bがアーティファクトを削除する方法の詳細については、[Delete artifacts](./delete-artifacts.md)ページを参照してください。

W&BアプリでArtifacts TTLを使用してデータ保持を管理する方法については、この[ビデオチュートリアル](https://www.youtube.com/watch?v=hQ9J6BoVmnc)をご覧ください。

:::note
W&BはモデルレジストリにリンクされたモデルアーティファクトのTTLポリシー設定オプションを無効にします。これは、プロダクションワークフローで使用されている場合、リンクされたモデルが誤って期限切れにならないようにするためです。
:::
:::info
* チーム管理者のみが[チームの設定](../app/settings-page/team-settings.md)を表示し、TTLの設定をチームレベルでアクセスできます。例えば、(1)TTLポリシーを設定または編集できる人を許可するか、(2) チームのデフォルトTTLを設定するかなど。
* W&BアプリのUIでTTLポリシーを設定または編集するオプションがアーティファクトの詳細に表示されない場合、またはプログラムでTTLを設定してもアーティファクトのTTLプロパティが正常に変更されない場合は、チーム管理者が権限を付与していない可能性があります。
:::

## TTLポリシーを編集および設定できる人を定義する
チーム内でTTLポリシーを設定および編集できる人を定義します。TTLの権限をチーム管理者のみに付与するか、チーム管理者とチームメンバーの両方にTTLの権限を付与するかを選択できます。

:::info
TTLポリシーを設定または編集できる人を定義できるのはチーム管理者のみです。
:::

1. チームのプロフィールページに移動します。
2. **Settings**タブを選択します。
3. **Artifacts time-to-live (TTL)セクション**に移動します。
4. **TTL permissions**ドロップダウンから、TTLポリシーを設定および編集できる人を選択します。
5. **Review and save settings**をクリックします。
6. 変更を確認し、**Save settings**を選択します。

![](/images/artifacts/define_who_sets_ttl.gif)

## TTLポリシーを作成する
アーティファクトを作成するときに、または作成後に遡及してTTLポリシーを設定します。

以下のコードスニペットでは、`<>`で囲まれた内容を置き換えて使用してください。

### アーティファクトを作成するときにTTLポリシーを設定する
W&B Python SDKを使用してアーティファクトを作成するときにTTLポリシーを定義します。TTLポリシーは通常、日数で定義されます。

:::tip
アーティファクトを作成するときにTTLポリシーを定義する方法は、通常の[アーティファクトを作成する](./construct-an-artifact.md)方法と似ています。ただし、タイムデルタをアーティファクトの`ttl`属性に渡す点が異なります。
:::

手順は次のとおりです:

1. [アーティファクトを作成する](./construct-an-artifact.md)。
2. ファイル、ディレクトリ、または参照などを[アーティファクトに追加する](./construct-an-artifact.md#add-files-to-an-artifact)。
3. Pythonの標準ライブラリの一部である[`datetime.timedelta`](https://docs.python.org/3/library/datetime.html)データ型でTTL時間制限を定義する。
4. [アーティファクトをログする](./construct-an-artifact.md#3-save-your-artifact-to-the-wb-server)。

以下のコードスニペットは、アーティファクトを作成し、TTLポリシーを設定する方法を示しています。

```python
import wandb
from datetime import timedelta

run = wandb.init(project="<my-project-name>", entity="<my-entity>")
artifact = wandb.Artifact(name="<artifact-name>", type="<type>")
artifact.add_file("<my_file>")

artifact.ttl = timedelta(days=30)  # TTLポリシーを設定
run.log_artifact(artifact)
```

前述のコードスニペットでは、アーティファクトのTTLポリシーを30日に設定しています。つまり、W&Bは30日後にアーティファクトを削除します。

### アーティファクトを作成した後にTTLポリシーを設定または編集する
既存のアーティファクトのTTLポリシーを定義するには、W&BアプリUIまたはW&B Python SDKを使用します。

:::note
アーティファクトのTTLを変更する場合、そのアーティファクトが削除されるまでの時間は依然としてアーティファクトの`createdAt`タイムスタンプを基に計算されます。
:::

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python SDK', value: 'python'},
    {label: 'W&B App', value: 'app'},
  ]}>
  <TabItem value="python">

1. [アーティファクトを取得する](./download-and-use-an-artifact.md)。
2. タイムデルタをアーティファクトの`ttl`属性に渡す。
3. [`save`](../../ref/python/run.md#save)メソッドでアーティファクトを更新する。

以下のコードスニペットは、アーティファクトにTTLポリシーを設定する方法を示しています:
```python
import wandb
from datetime import timedelta

artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = timedelta(days=365 * 2)  # 2年後に削除
artifact.save()
```

前述のコード例では、TTLポリシーを2年に設定しています。

  </TabItem>
  <TabItem value="app">

1. W&BアプリUIの自分のプロジェクトに移動します。
2. 左のパネルでアーティファクトアイコンを選択します。
3. アーティファクトのリストから、タイプを展開します。
4. TTLポリシーを編集したいアーティファクトバージョンを選択します。
5. **Version**タブをクリックします。
6. ドロップダウンから**Edit TTL policy**を選択します。
7. 表示されるモーダル内で、TTLポリシードロップダウンから**Custom**を選択します。
8. **TTL duration**フィールド内で、日単位でTTLポリシーを設定します。
9. 変更を保存するために**Update TTL**ボタンを選択します。

![](/images/artifacts/edit_ttl_ui.gif)

  </TabItem>
</Tabs>

## チームのデフォルトTTLポリシーを設定する

:::info
チームのデフォルトTTLポリシーを設定できるのはチーム管理者のみです。
:::

チームのデフォルトTTLポリシーを設定します。デフォルトのTTLポリシーは、既存および将来の全てのアーティファクトに、それぞれの作成日に基づいて適用されます。既存のバージョンレベルのTTLポリシーを持つアーティファクトはチームのデフォルトTTLの影響を受けません。

1. チームのプロフィールページに移動します。
2. **Settings**タブを選択します。
3. **Artifacts time-to-live (TTL)セクション**に移動します。
4. **Set team's default TTL policy**をクリックします。
5. **Duration**フィールド内で、日単位でTTLポリシーを設定します。
6. **Review and save settings**をクリックします。
7. 変更を確認し、**Save settings**を選択します。

![](/images/artifacts/set_default_ttl.gif)

## TTLポリシーを無効にする
特定のアーティファクトバージョンのTTLポリシーを無効にするには、W&B Python SDKまたはW&BアプリUIを使用します。

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python SDK', value: 'python'},
    {label: 'W&B App', value: 'app'},
  ]}>
  <TabItem value="python">

1. [アーティファクトを取得する](./download-and-use-an-artifact.md)。
2. アーティファクトの`ttl`属性を`None`に設定します。
3. [`save`](../../ref/python/run.md#save)メソッドでアーティファクトを更新します。

以下のコードスニペットは、アーティファクトのTTLポリシーを無効にする方法を示しています:
```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = None
artifact.save()
```

  </TabItem>
  <TabItem value="app">

1. W&BアプリUIの自分のプロジェクトに移動します。
2. 左のパネルでアーティファクトアイコンを選択します。
3. アーティファクトのリストから、タイプを展開します。
4. TTLポリシーを編集したいアーティファクトバージョンを選択します。
5. **Version**タブをクリックします。
6. **Link to registry**ボタンの横にあるミートボールUIアイコンをクリックします。
7. ドロップダウンから**Edit TTL policy**を選択します。
8. 表示されるモーダル内で、TTLポリシードロップダウンから**Deactivate**を選択します。
9. 変更を保存するために**Update TTL**ボタンを選択します。

![](/images/artifacts/remove_ttl_polilcy.gif)

  </TabItem>
</Tabs>

## TTLポリシーを表示する
Python SDKまたはW&BアプリのUIを使用して、アーティファクトのTTLポリシーを表示します。

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python SDK', value: 'python'},
    {label: 'W&B App', value: 'app'},
  ]}>
  <TabItem value="python">

print文を使用してアーティファクトのTTLポリシーを表示します。以下の例では、アーティファクトを取得してそのTTLポリシーを表示する方法を示しています:

```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
print(artifact.ttl)
```

  </TabItem>
  <TabItem value="app">

W&BアプリUIでアーティファクトのTTLポリシーを表示します。

1. [https://wandb.ai](https://wandb.ai)のW&Bアプリにアクセスします。
2. 自分のW&Bプロジェクトに移動します。
3. プロジェクト内で、左のサイドバーのArtifactsタブを選択します。
4. コレクションをクリックします。

コレクションビュー内では、選択されたコレクション内の全てのアーティファクトを見ることができます。`Time to Live`列内で、そのアーティファクトに割り当てられたTTLポリシーを見ることができます。

![](/images/artifacts/ttl_collection_panel_ui.png)

  </TabItem>
</Tabs>