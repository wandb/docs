---
title: アーティファクトバージョンをRegistryにリンクする
menu:
  default:
    identifier: ja-guides-core-registry-link_version
    parent: registry
weight: 5
---

アーティファクトのバージョンをコレクションにリンクすることで、組織内の他のメンバーも利用できるようにします。

アーティファクトをRegistryにリンクすると、そのアーティファクトがそのRegistryに「公開」されます。そのRegistryへアクセスできる任意のユーザーは、コレクションにあるリンク済みアーティファクトのバージョンにアクセス可能です。

言い換えれば、アーティファクトをRegistryのコレクションにリンクすることで、そのアーティファクトバージョンがプライベートなプロジェクトレベルの範囲から、組織全体で共有できる範囲へと拡張されます。

{{% alert %}}
「type」という用語はアーティファクトオブジェクトのタイプを指します。アーティファクトオブジェクト（[`wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}})）を作成したり、アーティファクトをログする時（[`wandb.init.log_artifact`]({{< relref path="/ref/python/sdk/classes/run.md#log_artifact" lang="ja" >}})）、`type` パラメータでタイプを指定します。
{{% /alert %}}

## アーティファクトをコレクションにリンクする

アーティファクトのバージョンをコレクションに、インタラクティブに、またはプログラムでリンクできます。

{{% alert %}}
アーティファクトをRegistryにリンクする前に、そのコレクションで許可されているアーティファクトタイプを確認しましょう。コレクションのタイプについて詳しくは、[コレクションの作成]({{< relref path="./create_collection.md" lang="ja" >}}) 内の「コレクションタイプ」をご覧ください。
{{% /alert %}}

ユースケースに応じて、以下のタブから該当する方法を選んでアーティファクトバージョンをリンクしてください。

{{% alert %}}
アーティファクトバージョンでメトリクスをログした場合（例えば `run.log_artifact()` を使って）、そのバージョンの詳細ページでメトリクスを確認したり、アーティファクトページ上でバージョン間のメトリクスを比較できます。詳しくは[Registry内でリンク済みアーティファクトを表示]({{< relref path="#view-linked-artifacts-in-a-registry" lang="ja" >}})を参照してください。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
{{% alert %}}
[バージョンのリンク方法を解説した動画](https://www.youtube.com/watch?v=2i_n1ExgO0A)（8分）もぜひご覧ください。
{{% /alert %}}

[`wandb.init.Run.link_artifact()`]({{< relref path="/ref/python/sdk/classes/run.md#link_artifact" lang="ja" >}}) を使って、アーティファクトバージョンをコレクションにプログラムでリンクできます。

{{% alert %}}
アーティファクトをコレクションにリンクする前に、そのコレクションが属するRegistryがすでに存在していることを確認してください。Registryの存在を確かめるには、W&B App UI の Registry アプリでRegistry名を検索してください。
{{% /alert %}}

`target_path` パラメータを使って、アーティファクトをリンクしたいコレクションおよびRegistryを指定します。target path は "wandb-registry" のプレフィックスとRegistry名・コレクション名をスラッシュ区切りで組み合わせたものです。

```text
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}
```

下記コードスニペットをコピー＆ペーストし、既存のRegistry内のコレクションへアーティファクトバージョンをリンクしましょう。`<>` で囲まれた部分はご自身の値に置き換えてください。

```python
import wandb

# run を初期化
run = wandb.init(
  entity = "<team_entity>",
  project = "<project_name>"
)

# アーティファクトオブジェクトを生成
# typeパラメータでアーティファクトおよび
# コレクションタイプを指定します
artifact = wandb.Artifact(name = "<name>", type = "<type>")

# ファイルをアーティファクトに追加
# ローカルマシン上のファイルパスを指定
artifact.add_file(local_path = "<local_path_to_artifact>")

# アーティファクトをリンクするコレクション・Registryを指定
REGISTRY_NAME = "<registry_name>"  
COLLECTION_NAME = "<collection_name>"
target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

# アーティファクトをコレクションにリンク
run.link_artifact(artifact = artifact, target_path = target_path)
```
{{% alert %}}
Model Registryまたは Dataset Registryにアーティファクトバージョンをリンクしたい場合、アーティファクトタイプとしてそれぞれ `"model"` または `"dataset"` を指定してください。
{{% /alert %}}

  {{% /tab %}}
  {{% tab header="Registry App" %}}
1. Registry App にアクセスします。
    {{< img src="/images/registry/navigate_to_registry_app.png" alt="Registry App navigation" >}}
2. アーティファクトバージョンをリンクしたいコレクション名の横でマウスオーバーします。
3. **View details** 横の三点リーダーアイコン（三本線のメニュー）を選択します。
4. ドロップダウンから **Link new version** を選びます。
5. サイドバーで、**Team** ドロップダウンからチーム名を選択します。
5. **Project** ドロップダウンからアーティファクトを含むプロジェクト名を選択します。
6. **Artifact** ドロップダウンからアーティファクト名を選択します。
7. **Version** ドロップダウンからコレクションにリンクしたいアーティファクトバージョンを選びます。

  
  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B App のプロジェクト内アーティファクトブラウザ（`https://wandb.ai/<entity>/<project>/artifacts`）にアクセスします。
2. 左サイドバーの Artifacts アイコンを選択します。
3. Registryへリンクしたいアーティファクトバージョンをクリック。
4. **Version overview** セクション内で **Link to registry** ボタンをクリックします。
5. 画面右側に表示されるモーダルで、**Select a register model** メニューからアーティファクトを選択。
6. **Next step** をクリックします。
7. （任意）**Aliases** ドロップダウンでエイリアスを選択します。
8. **Link to registry** をクリックします。

  
  {{% /tab %}}
{{< /tabpane >}}





リンク済みアーティファクトのメタデータ・バージョン情報・使用状況・リネージなどの情報は Registry App で確認できます。

## Registry内でリンク済みアーティファクトを表示

リンクされたアーティファクトのメタデータ、リネージ、使用状況などの情報は Registry App で確認できます。

1. Registry App へアクセスします。
2. アーティファクトをリンクしたRegistry名を選択します。
3. コレクション名を選択します。
4. コレクション内アーティファクトがメトリクスを記録している場合は **Show metrics** をクリックしてバージョン間で比較できます。
4. アーティファクトバージョン一覧からアクセスしたいバージョンを選択します。バージョン番号は `v0` からのインクリメンタル割り当てです。
5. アーティファクトバージョン詳細を見たい場合は、そのバージョンをクリック。ページ内のタブから、そのバージョンのメタデータ（記録されたメトリクス含む）、リネージ、使用情報を確認できます。

**Version** タブにある **Full Name** 欄に注目してください。リンク済みアーティファクトのフルネームは、Registry、コレクション名、アーティファクトバージョンのエイリアスまたはインデックスから成ります。

```text title="リンク済みアーティファクトのフルネーム"
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{INTEGER}
```

アーティファクトバージョンへプログラム経由でアクセスする際は、このフルネームが必要です。

## トラブルシューティング

アーティファクトのリンクに失敗する際、以下の点を再確認しましょう。

### 個人アカウントからのアーティファクト記録

個人エンティティで記録したアーティファクトは、Registryにリンクできません。アーティファクトは必ず組織内のチームエンティティでログしましょう。組織チームで記録されたアーティファクトのみ、その組織のRegistryへリンク可能です。

{{% alert title="" %}}
アーティファクトをRegistryにリンクしたい場合は、必ずチームエンティティでログしてください。
{{% /alert %}}


#### チームエンティティの確認方法

W&B では、チーム名がそのままエンティティ名となります。たとえば、チーム名が **team-awesome** なら、そのチームエンティティも `team-awesome` です。

チーム名を確認するには:

1. チームの W&B プロフィールページへアクセス。
2. サイトの URL をコピー。形式は `https://wandb.ai/<team>` です。`<team>` がチーム名およびエンティティ名となります。

#### チームエンティティでログする方法
1. [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) で run を初期化する際、`entity` にチーム名を指定します。指定しない場合、デフォルトのエンティティ（ご自身の個人エンティティなど）が利用されるため、必ず明示しましょう。

  ```python 
  import wandb   

  run = wandb.init(
    entity='<team_entity>', 
    project='<project_name>'
    )
  ```

2. run.log_artifact を利用するか、Artifact オブジェクトを生成してファイルを追加することでアーティファクトをログします。

    ```python
    artifact = wandb.Artifact(name="<artifact_name>", type="<type>")
    ```
    アーティファクトのログ方法は[アーティファクトの構築]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ja" >}})を参考にしてください。
3. 個人エンティティでログした場合は、組織内エンティティへ再度ログし直す必要があります。

### W&B App UI でRegistryのパスを確認する

Registryのパスは、空のコレクションを作って詳細を確認するか、コレクションホームページの自動生成コードをコピーすることで確認できます。

#### 自動生成コードをコピー

1. Registry app（https://wandb.ai/registry/）にアクセスします。
2. アーティファクトをリンクしたいRegistryをクリックします。
3. ページ上部に自動生成されたコードブロックが表示されます。
4. これを自分のコードに貼り付け、最後のパス部分を自分のコレクション名に置き換えてください。

{{< img src="/images/registry/get_autogenerated_code.gif" alt="Auto-generated code snippet" >}}

#### 空のコレクションを作成

1. Registry app（https://wandb.ai/registry/）にアクセスします。
2. アーティファクトをリンクしたいRegistryを選択します。
4. 空のコレクションをクリック。存在しない場合は新しく作成します。
5. 表示されたコードスニペット内から `.link_artifact()` の `target_path` フィールドを確認します。
6. （任意）コレクションを削除します。

{{< img src="/images/registry/check_empty_collection.gif" alt="Create an empty collection" >}}

たとえば、上記手順を経て下記のようなコードブロックが表示された場合:

```python
target_path = 
      "smle-registries-bug-bash/wandb-registry-Golden Datasets/raw_images"
```

各要素を分解すると、アーティファクトをプログラムからリンクする際のパス設定方法が分かります。

```python
ORG_ENTITY_NAME = "smle-registries-bug-bash"
REGISTRY_NAME = "Golden Datasets"
COLLECTION_NAME = "raw_images"
```

{{% alert %}}
一時的なコレクション名ではなく、実際にリンクしたいコレクション名で置き換えるのを忘れずに！
{{% /alert %}}