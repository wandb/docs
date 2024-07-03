---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# アーティファクトバージョンをレジストリにリンクする

プログラムまたは対話形式でアーティファクトバージョンをレジストリにリンクします。

:::info
アーティファクトをレジストリにリンクすると、そのアーティファクトがそのレジストリに「公開」されます。そのレジストリにアクセスできるユーザーは、アーティファクトをコレクションにリンクすると、リンクされたアーティファクトバージョンにアクセスできます。

言い換えると、アーティファクトをレジストリコレクションにリンクすることで、そのアーティファクトバージョンはプライベートなプロジェクトレベルのスコープから、共有組織レベルのスコープに移行します。
:::

ユースケースに基づいて、以下のタブに記載されている手順に従ってアーティファクトバージョンをリンクしてください。

<Tabs
  defaultValue="python_sdk"
  values={[
    {label: 'Python SDK', value: 'python_sdk'},
    {label: 'Registry App', value: 'registry_ui'},
    {label: 'Artifact browser', value: 'artifacts_ui'},
  ]}>
  <TabItem value="python_sdk">

[`link_artifact`](../../ref/python/run.md#link_artifact) メソッドを使用して、プログラムでアーティファクトをレジストリにリンクします。アーティファクトをリンクするときは、`target_path` パラメータにアーティファクトバージョンをリンクしたいパスを指定します。ターゲットパスは `"{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"` の形式を取ります。

`<>` で囲まれた値を自分の情報に置き換えてください：

```python
import wandb

ARTIFACT_NAME = "<ARTIFACT-TO-LINK>"
ARTIFACT_TYPE = "ARTIFACT-TYPE"
ENTITY_NAME = "<TEAM-ARTIFACT-BELONGS-IN>"
PROJECT_NAME = "<PROJECT-ARTIFACT-TO-LINK-BELONGS-IN>"

ORG_ENTITY_NAME = "<YOUR ORG NAME>"
REGISTRY_NAME = "<REGISTRY-TO-LINK-TO>"
COLLECTION_NAME = "<REGISTRY-COLLECTION-TO-LINK-TO>"

run = wandb.init(entity=ENTITY_NAME, project=PROJECT_NAME)
artifact = wandb.Artifact(name=ARTIFACT_NAME, type=ARTIFACT_TYPE)
run.link_artifact(
    artifact=artifact,
    target_path=f"{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
)
run.finish()
```

**Models** レジストリまたは **Dataset** レジストリにアーティファクトバージョンをリンクしたい場合は、アーティファクトタイプに `"model"` または `"dataset"` を設定してください。

  </TabItem>
  <TabItem value="registry_ui">

1. Registry App に移動します。
![](/images/registry/navigate_to_registry_app.png)
2. アーティファクトバージョンをリンクしたいコレクションの名前の横にマウスをホバーします。
3. **詳細を見る** の横にあるミートボールメニューアイコン（三つの水平線）を選択します。
4. プルダウンメニューから **新しいバージョンをリンク** を選択します。
5. 表示されるサイドバーから、**Team** プルダウンメニューでチームの名前を選択します。
5. **Project** プルダウンメニューから、アーティファクトを含むプロジェクトの名前を選択します。
6. **Artifact** プルダウンメニューから、アーティファクトの名前を選択します。
7. **Version** プルダウンメニューから、コレクションにリンクしたいアーティファクトバージョンを選択します。

  </TabItem>
  <TabItem value="artifacts_ui">

1. W&B App 上のプロジェクトのアーティファクトブラウザーに移動します: `https://wandb.ai/<entity>/<project>/artifacts`
2. 左サイドバーの Artifacts アイコンを選択します。
3. レジストリにリンクしたいアーティファクトバージョンをクリックします。
4. **バージョン概要** セクション内で、**レジストリにリンク** ボタンをクリックします。
5. 画面右側に表示されるモーダルから、**登録モデルを選択** メニューのドロップダウンでアーティファクトを選択します。
6. **次のステップ** をクリックします。
7. (オプション) **エイリアス** ドロップダウンからエイリアスを選択します。
8. **レジストリにリンク** をクリックします。

  </TabItem>
</Tabs>

:::tip リンクされたバージョン vs ソースバージョン
* ソースバージョン: チームのプロジェクト内で [run](../runs/intro.md) にログされたアーティファクトバージョン。
* リンクされたバージョン: レジストリに公開されたアーティファクトバージョン。これはソースアーティファクトへのポインタであり、まったく同じアーティファクトバージョンがレジストリのスコープで利用可能になります。
:::