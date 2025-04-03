---
title: Link an artifact version to a registry
menu:
  default:
    identifier: ja-guides-core-registry-link_version
    parent: registry
weight: 5
---

Artifact の バージョンをコレクションにリンクして、組織内の他のメンバーが利用できるようにします。

Artifact をレジストリにリンクすると、その Artifact がそのレジストリに「公開」されます。そのレジストリへのアクセス権を持つ ユーザー は、コレクション内のリンクされた Artifact の バージョン にアクセスできます。

言い換えれば、Artifact をレジストリ コレクションにリンクすると、その Artifact の バージョン がプライベートなプロジェクトレベルのスコープから、共有の組織レベルのスコープに移行します。

{{% alert %}}
「type」という用語は、Artifact オブジェクト の type を指します。Artifact オブジェクト ([`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ja" >}})) を作成するか、Artifact ([`wandb.init.log_artifact`]({{< relref path="/ref/python/run.md#log_artifact" lang="ja" >}})) を ログ に記録する場合、`type` パラメータ の type を指定します。
{{% /alert %}}

## Artifact をコレクションにリンクする

Artifact の バージョン を、インタラクティブまたはプログラムでコレクションにリンクします。

{{% alert %}}
Artifact をレジストリにリンクする前に、そのコレクションが許可する Artifact の type を確認してください。コレクションの type について詳しくは、[コレクションを作成する]({{< relref path="./create_collection.md" lang="ja" >}}) の「コレクション の type」をご覧ください。
{{% /alert %}}

ユースケース に基づいて、以下のタブに記載されている手順に従って、Artifact の バージョン をリンクしてください。

{{% alert %}}
Artifact の バージョン がメトリクスを ログ に記録している場合 (たとえば、`run.log_artifact()` を使用するなど)、その バージョン の詳細ページからその バージョン のメトリクスを表示したり、Artifact のページから Artifact の バージョン 全体のメトリクスを比較したりできます。[レジストリ内のリンクされた Artifact を表示する]({{< relref path="#view-linked-artifacts-in-a-registry" lang="ja" >}}) を参照してください。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
{{% alert %}}
[バージョン のリンクを示す動画](https://www.youtube.com/watch?v=2i_n1ExgO0A) (8 分) をご覧ください。
{{% /alert %}}

[`wandb.init.Run.link_artifact()`]({{< relref path="/ref/python/run.md#link_artifact" lang="ja" >}}) を使用して、プログラムで Artifact の バージョン をコレクションにリンクします。

{{% alert %}}
Artifact をコレクションにリンクする前に、コレクションが属するレジストリがすでに存在することを確認してください。レジストリが存在することを確認するには、W&B App UI の Registry アプリ に移動し、レジストリの名前を検索します。
{{% /alert %}}

`target_path` パラメータ を使用して、Artifact の バージョン のリンク先となるコレクションとレジストリを指定します。ターゲット パス は、プレフィックス「wandb-registry」、レジストリの名前、およびフォワード スラッシュで区切られたコレクションの名前で構成されます。

```text
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}
```

以下の コードスニペット をコピーして貼り付け、既存のレジストリ内のコレクションに Artifact の バージョン をリンクします。「<>」で囲まれた 値 を自分の 値 に置き換えます。

```python
import wandb

# Run を初期化する
run = wandb.init(
  entity = "<team_entity>",
  project = "<project_name>"
)

# Artifact オブジェクトを作成する
# type パラメータ は、Artifact オブジェクト の type と
# コレクション の type の両方を指定します
artifact = wandb.Artifact(name = "<name>", type = "<type>")

# Artifact オブジェクト にファイルを追加します。
# ローカルマシン 上のファイルへの パス を指定します。
artifact.add_file(local_path = "<local_path_to_artifact>")

# Artifact のリンク先のコレクションとレジストリを指定します
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

# Artifact をコレクションにリンクする
run.link_artifact(artifact = artifact, target_path = target_path)
```
{{% alert %}}
Artifact の バージョン を Model Registry または Dataset registry にリンクする場合は、Artifact の type をそれぞれ `"model"` または `"dataset"` に設定します。
{{% /alert %}}

  {{% /tab %}}
  {{% tab header="Registry App" %}}
1. Registry App に移動します。
    {{< img src="/images/registry/navigate_to_registry_app.png" alt="" >}}
2. Artifact の バージョン をリンクするコレクションの名前の横に マウス を合わせます。
3. **詳細を表示** の横にある ミートボール メニュー アイコン (3 つの水平ドット) を選択します。
4. ドロップダウンから、**新しい バージョン をリンク** を選択します。
5. 表示されるサイドバーから、**Team** ドロップダウンから チーム の名前を選択します。
5. **Project** ドロップダウンから、Artifact を含むプロジェクトの名前を選択します。
6. **Artifact** ドロップダウンから、Artifact の名前を選択します。
7. **バージョン** ドロップダウンから、コレクションにリンクする Artifact の バージョン を選択します。

  
  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B App のプロジェクト の Artifact ブラウザ (`https://wandb.ai/<entity>/<project>/artifacts`) に移動します。
2. 左側のサイドバーにある Artifact アイコンを選択します。
3. レジストリにリンクする Artifact の バージョン をクリックします。
4. **バージョン の概要** セクション内で、**レジストリにリンク** ボタンをクリックします。
5. 画面の右側に表示されるモーダルから、**レジストリ モデル を選択** メニュー ドロップダウンから Artifact を選択します。
6. **次のステップ** をクリックします。
7. (オプション) **エイリアス** ドロップダウンから エイリアス を選択します。
8. **レジストリにリンク** をクリックします。

  
  {{% /tab %}}
{{< /tabpane >}}

Registry アプリ で、リンクされた Artifact の メタデータ、バージョン データ、使用状況、リネージ 情報を表示します。

## レジストリ内のリンクされた Artifact を表示する

Registry アプリ で、メタデータ、リネージ、使用状況情報など、リンクされた Artifact に関する情報を表示します。

1. Registry アプリ に移動します。
2. Artifact をリンクしたレジストリの名前を選択します。
3. コレクションの名前を選択します。
4. コレクションの Artifact がメトリクスを ログ に記録する場合は、**メトリクスの表示** をクリックして バージョン 全体のメトリクスを比較します。
5. Artifact の バージョン のリストから、アクセスする バージョン を選択します。バージョン 番号は、`v0` から始まるリンクされた各 Artifact の バージョン にインクリメントに割り当てられます。
6. Artifact の バージョン に関する詳細を表示するには、バージョン をクリックします。このページのタブから、その バージョン のメタデータ (ログ に記録されたメトリクスを含む)、リネージ、および使用状況情報を表示できます。

**バージョン** タブ内の **フルネーム** フィールドに注意してください。リンクされた Artifact のフルネームは、レジストリ、コレクション名、および Artifact の バージョン の エイリアス または インデックス で構成されます。

```text title="リンクされた Artifact のフルネーム"
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{INTEGER}
```

プログラムで Artifact の バージョン にアクセスするには、リンクされた Artifact のフルネームが必要です。

## トラブルシューティング

Artifact をリンクできない場合は、以下に示す一般的な確認事項を再確認してください。

### 個人の アカウント から Artifact を ログ に記録する

個人の Entity で W&B に ログ に記録された Artifact は、レジストリにリンクできません。組織内の チーム Entity を使用して Artifact を ログ に記録していることを確認してください。組織の チーム 内で ログ に記録された Artifact のみ、組織のレジストリにリンクできます。

{{% alert title="" %}}
Artifact をレジストリにリンクする場合は、チーム Entity で Artifact を ログ に記録していることを確認してください。
{{% /alert %}}

#### チーム Entity を検索する

W&B は、チーム の名前を チーム の Entity として使用します。たとえば、チーム の名前が **team-awesome** の場合、チーム Entity は `team-awesome` です。

次の方法で チーム の名前を確認できます。

1. チーム の W&B プロファイル ページに移動します。
2. サイト の URL をコピーします。これは `https://wandb.ai/<team>` の形式です。`<team>` は チーム の名前と チーム の Entity の両方です。

#### チーム Entity から ログ に記録する
1. [`wandb.init()`]({{< relref path="/ref/python/init" lang="ja" >}}) で run を初期化するときに、Entity として チーム を指定します。run を初期化するときに `entity` を指定しない場合、run はデフォルト の Entity を使用します。これは チーム Entity である場合とそうでない場合があります。
  ```python
  import wandb

  run = wandb.init(
    entity='<team_entity>',
    project='<project_name>'
    )
  ```
2. run.log_artifact を使用して、または Artifact オブジェクト を作成し、次にファイルを Artifact オブジェクト に追加して、Artifact を run に ログ に記録します。

    ```python
    artifact = wandb.Artifact(name="<artifact_name>", type="<type>")
    ```
    Artifact の ログ 方法について詳しくは、[Artifact を構築する]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ja" >}}) をご覧ください。
3. Artifact が個人の Entity に ログ に記録されている場合は、組織内の Entity に再度 ログ に記録する必要があります。

### W&B App UI でレジストリの パス を確認する

UI を使用してレジストリの パス を確認するには、空のコレクションを作成してコレクションの詳細を表示するか、コレクションの ホーム ページで自動生成された コード をコピーして貼り付けます。

#### 自動生成された コード をコピーして貼り付けます

1. https://wandb.ai/registry/ で Registry アプリ に移動します。
2. Artifact をリンクするレジストリをクリックします。
3. ページの上部に、自動生成された コード ブロックが表示されます。
4. これを コード にコピーして貼り付け、パス の最後の部分をコレクションの名前に置き換えてください。

{{< img src="/images/registry/get_autogenerated_code.gif" alt="" >}}

#### 空のコレクションを作成する

1. https://wandb.ai/registry/ で Registry アプリ に移動します。
2. Artifact をリンクするレジストリをクリックします。
4. 空のコレクションをクリックします。空のコレクションが存在しない場合は、新しいコレクションを作成します。
5. 表示される コードスニペット 内で、`.link_artifact()` 内の `target_path` フィールドを特定します。
6. (オプション) コレクションを削除します。

{{< img src="/images/registry/check_empty_collection.gif" alt="" >}}

たとえば、概説されている手順を完了した後、`target_path` パラメータ を使用して コード ブロックを見つけます。

```python
target_path =
      "smle-registries-bug-bash/wandb-registry-Golden Datasets/raw_images"
```

これをコンポーネント に分解すると、プログラムで Artifact をリンクするための パス を作成するために使用する必要があるものを確認できます。

```python
ORG_ENTITY_NAME = "smle-registries-bug-bash"
REGISTRY_NAME = "Golden Datasets"
COLLECTION_NAME = "raw_images"
```

{{% alert %}}
一時コレクションのコレクションの名前を、Artifact のリンク先のコレクションの名前に置き換えてください。
{{% /alert %}}
```
