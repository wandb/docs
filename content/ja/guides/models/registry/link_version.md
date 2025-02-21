---
title: Link an artifact version to a registry
menu:
  default:
    identifier: ja-guides-models-registry-link_version
    parent: registry
weight: 5
---

Artifact の バージョンをコレクションにリンクして、組織内の他のメンバーが利用できるようにします。

Artifact を レジストリにリンクすると、その Artifact がそのレジストリに「公開」されます。そのレジストリへのアクセス権を持つ ユーザー は、コレクション内のリンクされた Artifact バージョンにアクセスできます。

言い換えれば、Artifact を レジストリ コレクションにリンクすると、その Artifact バージョンがプライベートなプロジェクトレベルのスコープから、共有の組織レベルのスコープになります。

{{% alert %}}
「タイプ」という用語は、Artifact オブジェクトのタイプを指します。Artifact オブジェクト ([`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ja" >}})) を作成するか、Artifact ([`wandb.init.log_artifact`]({{< relref path="/ref/python/run.md#log_artifact" lang="ja" >}})) を ログ に記録するときに、`type` パラメータのタイプを指定します。
{{% /alert %}}

## Artifact を コレクションにリンクする

Artifact バージョンをインタラクティブまたはプログラムでコレクションにリンクします。

{{% alert %}}
Artifact を レジストリにリンクする前に、そのコレクションが許可する Artifact のタイプを確認してください。コレクションのタイプについて詳しくは、[コレクションを作成する]({{< relref path="./create_collection.md" lang="ja" >}}) の「コレクションのタイプ」をご覧ください。
{{% /alert %}}

ユースケース に基づいて、以下のタブに記載されている手順に従って Artifact バージョンをリンクします。

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
[`wandb.init.Run.link_artifact()`]({{< relref path="/ref/python/run.md#link_artifact" lang="ja" >}}) を使用して、プログラムで Artifact バージョンをコレクションにリンクします。

{{% alert %}}
Artifact を コレクションにリンクする前に、コレクションが属するレジストリが既に存在することを確認してください。レジストリが存在することを確認するには、W&B App UI の Registry アプリに移動し、レジストリの名前を検索します。
{{% /alert %}}

`target_path` パラメータを使用して、Artifact バージョンをリンクするコレクションとレジストリを指定します。ターゲット パスは、プレフィックス「wandb-registry」、レジストリの名前、およびフォワード スラッシュで区切られたコレクションの名前で構成されます。

```text
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}
```

以下の コードスニペット をコピーして貼り付け、既存のレジストリ内のコレクションに Artifact バージョンをリンクします。`<>` で囲まれた 値 を独自の値に置き換えます。

```python
import wandb

# Run を初期化します
run = wandb.init(
  entity = "<team_entity>",
  project = "<project_name>"
)

# Artifact オブジェクトを作成します
# type パラメータは、
# Artifact オブジェクトのタイプとコレクション タイプの両方を指定します
artifact = wandb.Artifact(name = "<name>", type = "<type>")

# ファイルを Artifact オブジェクトに追加します。
# ローカル マシンのファイルへの パス を指定します。
artifact.add_file(local_path = "<local_path_to_artifact>")

# Artifact をリンクするコレクションとレジストリを指定します
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

# Artifact を コレクションにリンクします
run.link_artifact(artifact = artifact, target_path = target_path)
```
{{% alert %}}
Artifact バージョンを モデルレジストリ または データセット レジストリにリンクする場合は、Artifact タイプをそれぞれ `"model"` または `"dataset"` に設定します。
{{% /alert %}}

  {{% /tab %}}
  {{% tab header="Registry App" %}}
1. Registry App に移動します。
{{< img src="/images/registry/navigate_to_registry_app.png" alt="" >}}
2. Artifact バージョンをリンクするコレクションの名前の横にマウスを置きます。
3. [**詳細を表示**] の横にあるミートボール メニュー アイコン (3 つの水平ドット) を選択します。
4. ドロップダウンから、[**新しい バージョン をリンク**] を選択します。
5. 表示されるサイドバーで、[**Team**] ドロップダウンからチームの名前を選択します。
5. [**Project**] ドロップダウンから、Artifact を含むプロジェクトの名前を選択します。
6. [**Artifact**] ドロップダウンから、Artifact の名前を選択します。
7. [**Version**] ドロップダウンから、コレクションにリンクする Artifact バージョンを選択します。

  
  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B App のプロジェクトの Artifact ブラウザ (`https://wandb.ai/<entity>/<project>/artifacts`) に移動します。
2. 左側のサイドバーにある Artifacts アイコンを選択します。
3. レジストリにリンクする Artifact バージョンをクリックします。
4. [**Version overview**] セクション内で、[**レジストリにリンク**] ボタンをクリックします。
5. 画面の右側に表示されるモーダルで、[**レジストリ モデル を選択**] メニュー ドロップダウンから Artifact を選択します。
6. [**次のステップ**] をクリックします。
7. (オプション) [**エイリアス**] ドロップダウンから エイリアス を選択します。
8. [**レジストリにリンク**] をクリックします。

  
  {{% /tab %}}
{{< /tabpane >}}

リンクされた Artifact の メタデータ 、バージョン データ、使用状況、リネージ 情報を Registry App で表示します。

## レジストリでリンクされた Artifact を表示する

Registry App で、メタデータ、リネージ 、使用状況情報など、リンクされた Artifact に関する情報を表示します。

1. Registry App に移動します。
2. Artifact を リンクしたレジストリの名前を選択します。
3. コレクションの名前を選択します。
4. Artifact バージョンのリストから、アクセスするバージョンを選択します。バージョン番号は、`v0` から始まる各リンクされた Artifact バージョンに段階的に割り当てられます。

Artifact バージョンを選択すると、そのバージョンの メタデータ 、リネージ 、および 使用状況情報を表示できます。

[**Version**] タブ内の [**フルネーム**] フィールドに注意してください。リンクされた Artifact のフルネームは、レジストリ、コレクション名、および Artifact バージョンの エイリアス またはインデックスで構成されます。

```text title="リンクされた Artifact のフルネーム"
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{INTEGER}
```

プログラムで Artifact バージョンにアクセスするには、リンクされた Artifact のフルネームが必要です。

## トラブルシューティング

Artifact を リンクできない場合は、以下に一般的な確認事項をいくつか示します。

### 個人的な アカウント から Artifact を ログ に記録する

個人の エンティティ を使用して W&B に ログ 記録された Artifact は、レジストリにリンクできません。組織内のチーム エンティティ を使用して Artifact を ログ 記録していることを確認してください。組織のチーム内で ログ 記録された Artifact のみ、組織のレジストリにリンクできます。

{{% alert title="" %}}
Artifact を レジストリにリンクする場合は、チーム エンティティ を使用して Artifact を ログ に記録していることを確認してください。
{{% /alert %}}

#### チーム エンティティ を見つける

W&B は、チームの名前をチームの エンティティ として使用します。たとえば、チームの名前が **team-awesome** の場合、チーム エンティティ は `team-awesome` になります。

チームの名前は、次の方法で確認できます。

1. チームの W&B プロファイル ページに移動します。
2. サイトの URL をコピーします。URL の形式は `https://wandb.ai/<team>` です。ここで、`<team>` はチームの名前とチームの エンティティ の両方です。

#### チーム エンティティ から ログ に記録する
1. [`wandb.init()`]({{< relref path="/ref/python/init" lang="ja" >}}) で Run を初期化するときに、チームを エンティティ として指定します。Run を初期化するときに `entity` を指定しない場合、Run はデフォルトの エンティティ を使用します。デフォルトの エンティティ は、チーム エンティティ である場合とそうでない場合があります。
  ```python
  import wandb

  run = wandb.init(
    entity='<team_entity>',
    project='<project_name>'
    )
  ```
2. run.log_artifact を使用するか、Artifact オブジェクトを作成し、次にファイルを追加して、Artifact を Run に ログ 記録します。

    ```python
    artifact = wandb.Artifact(name="<artifact_name>", type="<type>")
    ```
    Artifact の ログ 記録方法について詳しくは、[Artifact を構築する]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ja" >}}) をご覧ください。
3. Artifact が個人の エンティティ に ログ 記録されている場合は、組織内の エンティティ に再度 ログ 記録する必要があります。

### W&B App UI でレジストリの パス を確認する

UI でレジストリの パス を確認する方法は 2 つあります。空のコレクションを作成してコレクションの詳細を表示するか、コレクションのホームページで自動生成された コード をコピーして貼り付けます。

#### 自動生成された コード をコピーして貼り付ける

1. Registry アプリ (https://wandb.ai/registry/) に移動します。
2. Artifact を リンクするレジストリをクリックします。
3. ページの上部に、自動生成された コード ブロックが表示されます。
4. これを コード にコピーして貼り付け、パス の最後の部分をコレクションの名前に置き換えてください。

{{< img src="/images/registry/get_autogenerated_code.gif" alt="" >}}

#### 空のコレクションを作成する

1. Registry アプリ (https://wandb.ai/registry/) に移動します。
2. Artifact を リンクするレジストリをクリックします。
4. 空のコレクションをクリックします。空のコレクションが存在しない場合は、新しいコレクションを作成します。
5. 表示される コードスニペット 内で、`.link_artifact()` 内の `target_path` フィールドを特定します。
6. (オプション) コレクションを削除します。

{{< img src="/images/registry/check_empty_collection.gif" alt="" >}}

たとえば、概説されている手順を完了した後、`target_path` パラメータを含む コード ブロックが見つかったとします。

```python
target_path =
      "smle-registries-bug-bash/wandb-registry-Golden Datasets/raw_images"
```

これをコンポーネントに分解すると、Artifact を プログラム でリンクするための パス を作成するために何を使用する必要があるかがわかります。

```python
ORG_ENTITY_NAME = "smle-registries-bug-bash"
REGISTRY_NAME = "Golden Datasets"
COLLECTION_NAME = "raw_images"
```

{{% alert %}}
一時コレクションのコレクションの名前を、Artifact を リンクするコレクションの名前に置き換えてください。
{{% /alert %}}
```
