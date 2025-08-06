---
title: アーティファクトのバージョンをレジストリにリンクする
menu:
  default:
    identifier: link_version
    parent: registry
weight: 5
---

アーティファクトのバージョンをコレクションにリンクすることで、組織内の他のメンバーも利用できるようになります。

アーティファクトをレジストリにリンクすると、そのアーティファクトがそのレジストリに「公開」されます。そのレジストリにアクセス権があるユーザーは、コレクション内でリンクされたアーティファクトのバージョンにアクセスできます。

言い換えると、アーティファクトをレジストリコレクションにリンクすることで、アーティファクトのバージョンがプロジェクト内限定のプライベートな範囲から、組織全体で共有される範囲に移ることになります。

{{% alert %}}
「type」という用語はアーティファクトオブジェクトのタイプを指します。アーティファクトオブジェクト（[`wandb.Artifact`]({{< relref "/ref/python/sdk/classes/artifact.md" >}})）を作成する際、またはアーティファクトをログする（[`wandb.init.log_artifact`]({{< relref "/ref/python/sdk/classes/run.md#log_artifact" >}})）際に、`type` パラメータでタイプを指定します。
{{% /alert %}}

## アーティファクトをコレクションにリンクする

アーティファクトのバージョンをコレクションにインタラクティブまたはプログラムでリンクできます。

{{% alert %}}
アーティファクトをレジストリへリンクする前に、そのコレクションが許可しているアーティファクトタイプを確認してください。コレクションタイプの詳細については [Create a collection]({{< relref "./create_collection.md" >}}) 内の「Collection types」をご確認ください。
{{% /alert %}}

ユースケースに応じて、以下のタブに記載されている手順に従ってアーティファクトバージョンをリンクしてください。

{{% alert %}}
アーティファクトバージョンがメトリクスをログしている場合（例：`run.log_artifact()` を使用）、そのバージョンの詳細ページからメトリクスを確認したり、アーティファクトのページから異なるバージョン間でメトリクスを比較できます。詳しくは [View linked artifacts in a registry]({{< relref "#view-linked-artifacts-in-a-registry" >}}) をご覧ください。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
{{% alert %}}
[リンクする手順を動画で確認](https://www.youtube.com/watch?v=2i_n1ExgO0A)（約 8 分）。
{{% /alert %}}

[`wandb.init.Run.link_artifact()`]({{< relref "/ref/python/sdk/classes/run.md#link_artifact" >}}) を使って、プログラムからアーティファクトバージョンをコレクションにリンクできます。

{{% alert %}}
アーティファクトをコレクションにリンクする前に、そのコレクションが属しているレジストリが既に作成されていることを確認してください。レジストリの存在を確認するには、W&B App の Registry アプリでレジストリ名を検索してください。
{{% /alert %}}

`target_path` パラメータを使用して、リンクしたいコレクションとレジストリを指定します。target path は "wandb-registry" というプリフィックス、レジストリ名、コレクション名をスラッシュ区切りでつなげたものです：

```text
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}
```

以下のコードスニペットをコピー＆ペーストし、既存のレジストリ内のコレクションにアーティファクトバージョンをリンクしてください。山括弧（`<>`）で囲まれた値はご自身の値に置き換えてください。

```python
import wandb

# run を初期化
run = wandb.init(
  entity = "<team_entity>",
  project = "<project_name>"
)

# アーティファクトオブジェクトを作成
# type パラメータはアーティファクトオブジェクトおよびコレクションタイプを指定
artifact = wandb.Artifact(name = "<name>", type = "<type>")

# ファイルをアーティファクトオブジェクトに追加
# ローカル環境上のファイルパスを指定
artifact.add_file(local_path = "<local_path_to_artifact>")

# リンクしたいコレクションとレジストリを指定
REGISTRY_NAME = "<registry_name>"  
COLLECTION_NAME = "<collection_name>"
target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

# アーティファクトをコレクションにリンク
run.link_artifact(artifact = artifact, target_path = target_path)
```
{{% alert %}}
アーティファクトバージョンを Model registry や Dataset registry にリンクしたい場合、アーティファクトの type をそれぞれ `"model"` または `"dataset"` にしてください。
{{% /alert %}}

  {{% /tab %}}
  {{% tab header="Registry App" %}}
1. Registry App にアクセスしてください。
    {{< img src="/images/registry/navigate_to_registry_app.png" alt="Registry App navigation" >}}
2. アーティファクトバージョンをリンクしたいコレクション名の横にマウスポインタを移動します。
3. **View details** の横にある 3 点リーダのアイコンを選択します。
4. ドロップダウンから **Link new version** を選びます。
5. 表示されるサイドバーから **Team** のドロップダウンよりチーム名を選択します。
5. **Project** ドロップダウンから、ご自身のアーティファクトを含むプロジェクト名を選択します。
6. **Artifact** ドロップダウンからアーティファクト名を選択します。
7. **Version** ドロップダウンから、コレクションにリンクしたいアーティファクトバージョンを選択します。

  
  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B App 上のプロジェクトの artifact browser (`https://wandb.ai/<entity>/<project>/artifacts`) へアクセスします。
2. 左サイドバーの Artifacts アイコンを選択。
3. リンクしたいアーティファクトバージョンをクリックします。
4. **Version overview** セクション内の **Link to registry** ボタンをクリック。
5. 右側に表示されるモーダルの **Select a register model** メニューからアーティファクトを選択。
6. **Next step** をクリック。
7. （任意）**Aliases** ドロップダウンからエイリアスを選択します。
8. **Link to registry** をクリック。

  
  {{% /tab %}}
{{< /tabpane >}}

リンクされたアーティファクトのメタデータ、バージョン情報、利用状況、リネージ等を Registry App で確認できます。

## レジストリでリンクされたアーティファクトを確認

Registry App 上で、リンクされたアーティファクトのメタデータ、リネージ、利用状況等を確認できます。

1. Registry App にアクセスします。
2. アーティファクトをリンクしたレジストリ名を選択します。
3. コレクション名を選択します。
4. コレクション内のアーティファクトがメトリクスをログしている場合、**Show metrics** をクリックすることでバージョン間のメトリクス比較が可能です。
4. アーティファクトバージョンの一覧から、アクセスしたいバージョンを選択します。バージョン番号は `v0` から始まり、順に付与されます。
5. アーティファクトバージョンの詳細をみるには、そのバージョンをクリックします。ページ内のタブからは、そのバージョンのメタデータ（ログされたメトリクス含む）、リネージ、利用状況などを確認できます。

**Version** タブ内の **Full Name** フィールドに注目してください。リンク済みアーティファクトの full name は、レジストリ名、コレクション名、アーティファクトバージョンのエイリアスまたはインデックスから構成されます。

```text title="Full name of a linked artifact"
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{INTEGER}
```

リンク済みアーティファクトをプログラムから利用するには、この full name が必要です。

## トラブルシューティング

アーティファクトをリンクできない場合によくある確認事項をまとめます。

### 個人アカウントでアーティファクトをログしている

個人エンティティで W&B にログしたアーティファクトは、レジストリにリンクできません。組織内のチームエンティティを利用してアーティファクトをログするようにしてください。組織のチームでログしたアーティファクトのみ、組織のレジストリへリンク可能です。

{{% alert title="" %}}
アーティファクトをレジストリにリンクしたい場合、必ずチームエンティティでログしてください。
{{% /alert %}}


#### チームエンティティを探す

W&B ではチーム名が team entity になります。例えばチーム名が **team-awesome** の場合、team entity は `team-awesome` です。

チーム名を確認するには：

1. チームの W&B プロフィールページにアクセスします。
2. サイトの URL をコピーします。形式は `https://wandb.ai/<team>` で、ここで `<team>` がチームの名前＝team entity です。

#### チームエンティティからログする
1. [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) を使って run を初期化する際、entity としてチーム名を指定します。 entity パラメータを省略すると、デフォルトの entity（自分の個人 entity だったり）が使われる場合があります。

  ```python 
  import wandb   

  run = wandb.init(
    entity='<team_entity>', 
    project='<project_name>'
    )
  ```

2. run にアーティファクトをログします。run.log_artifact もしくは Artifact オブジェクトを作成しファイルを追加する方法も使えます：

    ```python
    artifact = wandb.Artifact(name="<artifact_name>", type="<type>")
    ```
    アーティファクトのログ方法は [Construct artifacts]({{< relref "/guides/core/artifacts/construct-an-artifact.md" >}}) をご覧ください。
3. 個人エンティティでアーティファクトがログされた場合、組織内の entity で再度ログする必要があります。

### W&B App UI でレジストリのパスを確認する

UI でレジストリのパスを確認する方法は 2 通りあります。空のコレクションを作成して詳細を表示するか、コレクションのホームページに自動生成されるコードをコピーして利用します。

#### 自動生成コードをコピー＆ペースト

1. https://wandb.ai/registry/ の Registry app に移動します。
2. リンクしたいレジストリをクリックします。
3. ページ上部に自動生成されたコードブロックが表示されます。
4. これをコードにコピーし、path の最後の部分をご自身のコレクション名に変更してください。

{{< img src="/images/registry/get_autogenerated_code.gif" alt="Auto-generated code snippet" >}}

#### 空のコレクションを作成する

1. https://wandb.ai/registry/ の Registry app にアクセスします。
2. リンクしたいレジストリをクリック。
4. 空のコレクションをクリック（なければ新規作成）。
5. 表示されるコードスニペット内の、`.link_artifact()` の `target_path` フィールドを確認します。
6. （必要であれば）コレクションを削除してください。

{{< img src="/images/registry/check_empty_collection.gif" alt="Create an empty collection" >}}

例えば、手順を完了後に次のような `target_path` を含むコードブロックが見つかります：

```python
target_path = 
      "smle-registries-bug-bash/wandb-registry-Golden Datasets/raw_images"
```

各部分を分解すると、アーティファクトをプログラムからリンクするために必要な path の作成方法がわかります。

```python
ORG_ENTITY_NAME = "smle-registries-bug-bash"
REGISTRY_NAME = "Golden Datasets"
COLLECTION_NAME = "raw_images"
```

{{% alert %}}
一時的なコレクションからコピーしたコレクション名を、リンクしたい正しいコレクション名に必ず置き換えてください。
{{% /alert %}}