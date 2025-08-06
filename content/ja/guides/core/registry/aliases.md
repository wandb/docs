---
title: エイリアスを使ってアーティファクト バージョンを参照する
weight: 5
---

特定の [artifact version]({{< relref "guides/core/artifacts/create-a-new-artifact-version" >}}) を 1 つ以上のエイリアスで参照することができます。[W&B は同じ名前でリンクされた各 artifact に自動的にエイリアスを割り当てます]({{< relref "aliases#default-aliases" >}})。また、[独自のカスタムエイリアスを 1 つまたは複数作成して]({{< relref "aliases#custom-aliases" >}})、特定の artifact version を参照することも可能です。

エイリアスは、レジストリ UI でそのエイリアス名の入った四角形として表示されます。[エイリアスが保護されている場合]({{< relref "aliases#protected-aliases" >}})、グレーの四角形とロックアイコンで表示されます。それ以外の場合はオレンジ色の四角形として表示されます。エイリアスはレジストリ間で共有されません。

{{% alert title="エイリアスとタグ、どちらを使うべきか" %}}
エイリアスは特定の artifact version を参照するために使用します。コレクション内の各エイリアスは一意です。特定のエイリアスは一度に 1 つの artifact version にしか割り当てられません。

タグは、artifact version やコレクションを共通のテーマで整理・グループ化するために使います。複数の artifact version やコレクションが同じタグを共有できます。
{{% /alert %}}

artifact version にエイリアスを追加すると、[Registry オートメーション]({{< relref  "/guides/core/automations/automation-events/#registry" >}}) を開始して、Slack チャンネルへの通知や webhook の発火が可能です。

## デフォルトのエイリアス

W&B は、同じ名前でリンクした artifact version それぞれに以下のエイリアスを自動で割り当てます:

* コレクション内で最も新しい artifact version には `latest` エイリアス。
* ユニークなバージョン番号。W&B は各 artifact version を（0 起点で）カウントし、そのカウント番号を利用してユニークなバージョン番号を割り当てます。

たとえば、`zoo_model` という artifact を 3 回リンクすると、エイリアスとして `v0`、`v1`、`v2` がそれぞれ生成されます。`v2` には `latest` エイリアスも付きます。

## カスタムエイリアス

ユースケースに応じて、特定の artifact version にカスタムエイリアスを 1 つまたは複数作成できます。例:

- `dataset_version_v0`、`dataset_version_v1`、`dataset_version_v2` などのエイリアスを使って、どのデータセットでモデルを学習させたか識別できるようにします。
- `best_model` エイリアスを付けて、パフォーマンスが最も良い artifact モデルのバージョンを管理することもできます。

[Member または Admin の registry 権限]({{< relref "guides/core/registry/configure_registry/#registry-roles" >}}) のあるユーザーは、そのレジストリ内でリンクされた artifact のカスタムエイリアスを追加・削除できます。必要に応じて、[保護されたエイリアス]({{< relref "aliases/#protected-aliases" >}}) を利用して、変更や削除から守りたい artifact version にラベル付けできます。

W&B Registry または Python SDK でカスタムエイリアスを作成できます。用途に応じて、下のタブから最適な方法を選択してください。

{{< tabpane text=true >}}
{{% tab header="W&B Registry" value="app" %}}

1. W&B Registry にアクセスします。
2. コレクション内の **View details** ボタンをクリックします。
3. **Versions** セクションで、特定の artifact version の **View** ボタンをクリックします。
4. **Aliases** 欄の横にある **+** ボタンをクリックして、エイリアスを 1 つまたは複数追加します。

{{% /tab %}}

{{% tab header="Python SDK" value="python" %}}
Python SDK で artifact version をコレクションにリンクする際、`alias` 引数として 1 つまたは複数のエイリアスのリストを指定できます。 もしエイリアスが存在しない場合、W&B が（[保護されていないエイリアス]({{< relref "#custom-aliases" >}}) として）自動で作成します。

以下のコードスニペットは、artifact version をコレクションにリンクし、Python SDK でエイリアスを追加する方法を示しています。`<>` 内はご自身の値に置き換えてください。

```python
import wandb

# run を初期化
run = wandb.init(entity = "<team_entity>", project = "<project_name>")

# artifact オブジェクトを作成
# type パラメータは artifact オブジェクトと
# コレクションタイプの両方を指定します
artifact = wandb.Artifact(name = "<name>", type = "<type>")

# artifact オブジェクトにファイルを追加
# ファイルのパスはローカルマシンのものを指定します
artifact.add_file(local_path = "<local_path_to_artifact>")

# artifact をリンクするコレクションとレジストリを指定
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

# artifact version をコレクションにリンク
# この artifact version にエイリアスを追加
run.link_artifact(
    artifact = artifact, 
    target_path = target_path, 
    aliases = ["<alias_1>", "<alias_2>"]
    )
```
{{% /tab %}}
{{< /tabpane >}}

### 保護されたエイリアス

[保護されたエイリアス]({{< relref "aliases/#protected-aliases" >}}) を利用して、編集や削除してはならない artifact version にラベルを付け、識別することができます。たとえば、組織の機械学習本番パイプラインで使用中の artifact version に `production` の保護されたエイリアスを付けて管理できます。

[Registry 管理者]({{< relref "/guides/core/registry/configure_registry/#registry-roles" >}}) や [サービスアカウント]({{< relref "/support/kb-articles/service_account_useful" >}})（Admin 権限あり）は、保護されたエイリアスの作成・追加・削除が可能です。Member や Viewer は、保護されたバージョンのリンク解除や、そのコレクションの削除はできません。詳細は [レジストリのアクセス設定]({{< relref "/guides/core/registry/configure_registry.md" >}}) を参照してください。

よく使われる保護されたエイリアスの例:

- **Production**: artifact version が本番利用の準備ができていることを示します。
- **Staging**: artifact version がテスト用に準備されていることを示します。

#### 保護されたエイリアスの作成

以下は W&B Registry UI で保護されたエイリアスを作成する手順です。

1. Registry App にアクセスします。
2. レジストリを選択します。
3. ページ右上のギアボタンをクリックしてレジストリの設定を表示します。
4. **Protected Aliases** セクションで **+** ボタンをクリックし、保護されたエイリアスを 1 つ以上追加します。

作成後、各保護されたエイリアスは **Protected Aliases** セクション内でロックアイコン付きのグレーの四角形として表示されます。

{{% alert %}}
カスタム（保護されていない）エイリアスと異なり、保護されたエイリアスの作成は W&B Registry UI 限定機能であり、Python SDK ではプログラム的には作成できません。artifact version に保護されたエイリアスを追加するには、W&B Registry UI か Python SDK のどちらかが利用できます。
{{% /alert %}}

以下はW&B Registry UIで artifact version に保護されたエイリアスを追加する手順です。

1. W&B Registry にアクセスします。
2. コレクション内の **View details** ボタンをクリックします。
3. **Versions** セクションで、特定の artifact version の **View** ボタンをクリックします。
4. **Aliases** 欄の横にある **+** ボタンをクリックして、保護されたエイリアスを 1 つまたは複数追加します。

保護されたエイリアスの作成後、管理者は Python SDK からプログラム的に artifact version への追加も可能です。[カスタムエイリアスの作成](#custom-aliases) セクションの W&B Registry および Python SDK のタブをご参照ください。

## 既存のエイリアスを探す

[W&B Registry のグローバル検索バー]({{< relref "/guides/core/registry/search_registry/#search-for-registry-items" >}}) で既存のエイリアスを検索できます。保護されたエイリアスを探す場合:

1. W&B Registry App にアクセスします。
2. ページ上部の検索バーで検索ワードを入力し、Enter キーで検索します。

指定したワードが既存のレジストリやコレクション名、artifact version タグ、コレクションタグ、エイリアスと一致した場合、検索結果が表示されます。

## 例

{{% alert %}}
次のコード例は [W&B Registry チュートリアル](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb) の続きです。利用にはあらかじめ [ノートブックの手順通り Zoo データセットを取得・加工] (https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb#scrollTo=87fecd29-8146-41e2-86fb-0bb4e3e3350a) しておく必要があります。Zoo データセットがあれば、artifact version を作成し、カスタムエイリアスを追加できます。
{{% /alert %}}

以下のコードスニペットは artifact version を作成し、カスタムエイリアスを追加する方法を示しています。例では [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/111/zoo) の Zoo データセットと `Zoo_Classifier_Models` レジストリの `Model` コレクションを利用しています。

```python
import wandb

# run を初期化
run = wandb.init(entity = "smle-reg-team-2", project = "zoo_experiment")

# artifact オブジェクトを作成
# type パラメータは artifact オブジェクトと
# コレクションタイプの両方を指定します
artifact = wandb.Artifact(name = "zoo_dataset", type = "dataset")

# artifact オブジェクトにファイルを追加
# ファイルのパスはローカルマシンのものを指定します
artifact.add_file(local_path="zoo_dataset.pt", name="zoo_dataset")
artifact.add_file(local_path="zoo_labels.pt", name="zoo_labels")

# artifact をリンクするコレクションとレジストリを指定
REGISTRY_NAME = "Model"
COLLECTION_NAME = "Zoo_Classifier_Models"
target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

# artifact version をコレクションにリンク
# この artifact version にエイリアスを追加
run.link_artifact(
    artifact = artifact,
    target_path = target_path,
    aliases = ["production-us", "production-eu"]
    )
```

1. まず、`wandb.Artifact()` で artifact オブジェクトを作成します。
2. 続いて、2 つのデータセット用 PyTorch テンソルを `wandb.Artifact.add_file()` で artifact オブジェクトに追加します。
3. 最後に、`link_artifact()` を使って artifact version を `Zoo_Classifier_Models` レジストリ内の `Model` コレクションにリンクします。また、`aliases` パラメータに `production-us` と `production-eu` を渡すことで、2 つのカスタムエイリアスも同時に追加します。