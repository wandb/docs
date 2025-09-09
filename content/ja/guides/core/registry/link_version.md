---
title: Artifacts のバージョンをレジストリにリンクする
menu:
  default:
    identifier: ja-guides-core-registry-link_version
    parent: registry
weight: 5
---

Artifacts のバージョンをコレクションにリンクすることで、組織内の他のメンバーが利用できるようになります。

アーティファクト を registry にリンクすると、そのアーティファクト が registry に「公開」されます。その registry へのアクセス権を持つユーザーは誰でも、コレクション内のリンクされたアーティファクト のバージョンにアクセスできます。

言い換えれば、アーティファクト を registry コレクションにリンクすることで、そのアーティファクト のバージョンはプライベートなプロジェクト レベルのスコープから、共有の組織レベルのスコープへと移行します。

{{% alert %}}
「type」という用語は、アーティファクト オブジェクトの type を指します。アーティファクト オブジェクトを作成する際 ([`wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}})) や、アーティファクト をログする際 ([`wandb.init.log_artifact`]({{< relref path="/ref/python/sdk/classes/run.md#log_artifact" lang="ja" >}})) に、`type` パラメータに type を指定します。
{{% /alert %}}

## アーティファクト をコレクションにリンクする

アーティファクト のバージョンは、対話的またはプログラムでコレクションにリンクできます。

{{% alert %}}
アーティファクト を registry にリンクする前に、そのコレクションが許可するアーティファクト の type を確認してください。コレクションの type の詳細については、[コレクションの作成]({{< relref path="./create_collection.md" lang="ja" >}}) 内の「コレクションの type」を参照してください。
{{% /alert %}}

ユースケースに応じて、以下のタブで説明されている手順に従ってアーティファクト のバージョンをリンクしてください。

{{% alert %}}
アーティファクト のバージョンがメトリクスをログする場合 (`run.log_artifact()` を使用するなど)、そのバージョンの詳細ページからメトリクスを表示でき、アーティファクト のページからアーティファクト のバージョン間でメトリクスを比較できます。[registry でリンクされたアーティファクト を表示する]({{< relref path="#view-linked-artifacts-in-a-registry" lang="ja" >}}) を参照してください。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
{{% alert %}}
[バージョンをリンクするデモンストレーションビデオ](https://www.youtube.com/watch?v=2i_n1ExgO0A) (8分) をご覧ください。
{{% /alert %}}

[`wandb.init.Run.link_artifact()`]({{< relref path="/ref/python/sdk/classes/run.md#link_artifact" lang="ja" >}}) を使用して、アーティファクト のバージョンをコレクションにプログラムでリンクします。

{{% alert %}}
アーティファクト をコレクションにリンクする前に、そのコレクションが属する registry が既に存在することを確認してください。registry の存在を確認するには、W&B App UI の Registry App に移動し、registry の名前を検索してください。
{{% /alert %}}

`target_path` パラメータを使用して、アーティファクト のバージョンをリンクしたいコレクションと registry を指定します。target path は、プレフィックス「wandb-registry」、registry の名前、およびコレクションの名前をスラッシュで区切ったものです。

```text
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}
```

既存の registry 内のコレクションにアーティファクト のバージョンをリンクするには、以下のコードスニペットをコピーして貼り付けます。`< >` で囲まれた値はご自身の値に置き換えてください。

```python
import wandb

# run を初期化します
run = wandb.init(
  entity = "<team_entity>",
  project = "<project_name>"
)

# アーティファクト オブジェクトを作成します
# type パラメータは、
# アーティファクト オブジェクトの type とコレクションの type の両方を指定します。
artifact = wandb.Artifact(name = "<name>", type = "<type>")

# ファイルを アーティファクト オブジェクトに追加します。
# ローカルマシン上のファイルのパスを指定します。
artifact.add_file(local_path = "<local_path_to_artifact>")

# アーティファクト をリンクするコレクションと registry を指定します。
REGISTRY_NAME = "<registry_name>"  
COLLECTION_NAME = "<collection_name>"
target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

# アーティファクト をコレクションにリンクします
run.link_artifact(artifact = artifact, target_path = target_path)
```
{{% alert %}}
アーティファクト のバージョンをモデルレジストリ または データセット レジストリにリンクしたい場合は、アーティファクト の type をそれぞれ `"model"` または `"dataset"` に設定してください。
{{% /alert %}}

  {{% /tab %}}
  {{% tab header="Registry App" %}}
1. Registry App に移動します。
    {{< img src="/images/registry/navigate_to_registry_app.png" alt="Registry App のナビゲーション" >}}
2. アーティファクト のバージョンをリンクしたいコレクション名の横にマウスを合わせます。
3. 「**View details**」の横にあるミートボールメニューアイコン (3つの水平な点) を選択します。
4. ドロップダウンから「**Link new version**」を選択します。
5. 表示されるサイドバーから、「**Team**」ドロップダウンからチーム名を選択します。
6. 「**Project**」ドロップダウンから、アーティファクト を含むプロジェクト の名前を選択します。
7. 「**Artifact**」ドロップダウンから、アーティファクト の名前を選択します。
8. 「**Version**」ドロップダウンから、コレクションにリンクしたいアーティファクト のバージョンを選択します。

  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. W&B App で プロジェクト の Artifact browser に移動します: `https://wandb.ai/<entity>/<project>/artifacts`
2. 左サイドバーの Artifacts アイコンを選択します。
3. registry にリンクしたいアーティファクト のバージョンをクリックします。
4. 「**Version overview**」セクション内で、「**Link to registry**」ボタンをクリックします。
5. 画面右に表示されるモーダルから、「**Select a register model**」メニュードロップダウンからアーティファクト を選択します。
6. 「**Next step**」をクリックします。
7. (オプション) 「**Aliases**」ドロップダウンからエイリアスを選択します。
8. 「**Link to registry**」をクリックします。

  {{% /tab %}}
{{< /tabpane >}}

リンクされたアーティファクト のメタデータ、バージョンデータ、使用状況、リネージ情報などを Registry App で表示します。

## registry でリンクされたアーティファクト を表示する

Registry App で、メタデータ、リネージ、使用状況情報など、リンクされたアーティファクト に関する情報を表示します。

1. Registry App に移動します。
2. アーティファクト をリンクした registry の名前を選択します。
3. コレクションの名前を選択します。
4. コレクションのアーティファクト がメトリクスをログする場合、「**Show metrics**」をクリックしてバージョン間でメトリクスを比較します。
5. アーティファクト のバージョンリストから、アクセスしたいバージョンを選択します。バージョン番号は、リンクされた各アーティファクト のバージョンに `v0` から順に割り当てられます。
6. アーティファクト のバージョンの詳細を表示するには、そのバージョンをクリックします。このページのタブから、そのバージョンのメタデータ (ログされたメトリクスを含む)、リネージ、および使用状況情報を表示できます。

「**Version**」タブ内の「**Full Name**」フィールドに注目してください。リンクされたアーティファクト のフルネームは、registry、コレクション名、およびアーティファクト のバージョンのエイリアスまたはインデックスで構成されます。

```text title="リンクされたアーティファクト のフルネーム"
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{INTEGER}
```

リンクされたアーティファクト のフルネームは、アーティファクト のバージョンにプログラムでアクセスするために必要です。

## トラブルシューティング

アーティファクト をリンクできない場合に再確認すべき一般的な事項を以下に示します。

### 個人アカウントからのアーティファクト のログ

個人の entity で W&B にログされた Artifacts は registry にリンクできません。組織内のチーム entity を使用して Artifacts をログしていることを確認してください。組織のチーム内でログされた Artifacts のみが、組織の registry にリンクできます。

{{% alert title="" %}}
アーティファクト を registry にリンクしたい場合は、チーム entity でアーティファクト をログしていることを確認してください。
{{% /alert %}}

#### チームの entity を見つける

W&B は、チームの名前をチームの entity として使用します。たとえば、チーム名が **team-awesome** の場合、チームの entity は `team-awesome` です。

チームの名前は、以下の方法で確認できます。

1. チームの W&B プロフィールページに移動します。
2. サイトの URL をコピーします。`https://wandb.ai/<team>` の形式です。ここで `<team>` はチームの名前とチームの entity の両方です。

#### チーム entity からログする
1. [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) で run を初期化する際に、チームを entity として指定します。run を初期化する際に `entity` を指定しない場合、run はデフォルトの entity を使用しますが、これはチームの entity である場合とそうでない場合があります。

  ```python 
  import wandb   

  run = wandb.init(
    entity='<team_entity>', 
    project='<project_name>'
    )
  ```

2. run.log_artifact を使用するか、Artifact オブジェクトを作成してそれにファイルを追加することで、アーティファクト を run にログします。

    ```python
    artifact = wandb.Artifact(name="<artifact_name>", type="<type>")
    ```
    Artifacts をログするには、[アーティファクト の構築]({{< relref path="/guides/core/artifacts/construct-an-artifact.md" lang="ja" >}}) を参照してください。
3. アーティファクト が個人の entity にログされている場合、組織内の entity に再ログする必要があります。

### W&B App UI で registry のパスを確認する

UI で registry のパスを確認する方法は 2 つあります。空のコレクションを作成してコレクションの詳細を表示するか、コレクションのホームページで自動生成されたコードをコピー＆ペーストする方法です。

#### 自動生成されたコードをコピー＆ペーストする

1. Registry app (https://wandb.ai/registry/) に移動します。
2. アーティファクト をリンクしたい registry をクリックします。
3. ページの上部に、自動生成されたコードブロックが表示されます。
4. これをコードにコピー＆ペーストし、パスの最後の部分をコレクションの名前に置き換えてください。

{{< img src="/images/registry/get_autogenerated_code.gif" alt="自動生成されたコードスニペット" >}}

#### 空のコレクションを作成する

1. Registry app (https://wandb.ai/registry/) に移動します。
2. アーティファクト をリンクしたい registry をクリックします。
3. 空のコレクションをクリックします。空のコレクションが存在しない場合は、新しいコレクションを作成します。
4. 表示されるコードスニペット内で、`.link_artifact()` 内の `target_path` フィールドを特定します。
5. (オプション) コレクションを削除します。

{{< img src="/images/registry/check_empty_collection.gif" alt="空のコレクションを作成する" >}}

たとえば、上記の手順を完了した後、`target_path` パラメータを含むコードブロックが見つかります。

```python
target_path = 
      "smle-registries-bug-bash/wandb-registry-Golden Datasets/raw_images"
```

これを構成要素に分解すると、アーティファクト をプログラムでリンクするためのパスを作成するために必要なものがわかります。

```python
ORG_ENTITY_NAME = "smle-registries-bug-bash"
REGISTRY_NAME = "Golden Datasets"
COLLECTION_NAME = "raw_images"
```

{{% alert %}}
一時的なコレクションの名前を、アーティファクト をリンクしたいコレクションの名前に置き換えていることを確認してください。
{{% /alert %}}