---
title: エイリアスを使ってアーティファクト バージョンを参照する
menu:
  default:
    identifier: ja-guides-core-registry-aliases
weight: 5
---

特定の [artifact バージョン]({{< relref path="guides/core/artifacts/create-a-new-artifact-version" lang="ja" >}}) を、1つまたは複数のエイリアスで参照できます。[W&B は自動的にエイリアスを割り当て]({{< relref path="aliases#default-aliases" lang="ja" >}}) 、同じ名前でリンクした各 artifact にエイリアスを付けます。また、[カスタムエイリアスを作成]({{< relref path="aliases#custom-aliases" lang="ja" >}}) して特定の artifact バージョンを参照することもできます。

エイリアスは Registry UI 上で、そのエイリアス名が記載された四角い枠として表示されます。[エイリアスが保護されている場合]({{< relref path="aliases#protected-aliases" lang="ja" >}})、そのエイリアスは錠前アイコン付きのグレーの枠として表示されます。それ以外の場合、オレンジ色の枠となります。エイリアスはRegistry間で共有されません。

{{% alert title="エイリアスとタグの使い分けについて" %}}
エイリアスは、特定の artifact バージョンを参照したい場合に使います。1つのコレクション内のエイリアスは一意であり、同時に1つの artifact バージョンにしか割り当てられません。

タグは、アーティファクトのバージョンやコレクションを共通のテーマで整理・グループ化したい場合に使用します。複数の artifact バージョンやコレクションが同じタグを共有することが可能です。
{{% /alert %}}

artifact バージョンにエイリアスを追加する際、[Registry オートメーション]({{< relref path="/guides/core/automations/automation-events/#registry" lang="ja" >}}) を開始して Slack チャンネルに通知したり、webhook を発火させることもできます。

## デフォルトエイリアス

W&B は、同じ名前で各 artifact バージョンに以下のエイリアスを自動付与します。

* `latest` エイリアスは、コレクションにリンクされた最新バージョンの artifact に付与されます。
* 一意なバージョン番号。W&B はリンクした artifact バージョンごとに回数（0インデックスで）をカウントし、そのカウント値をもとにバージョン番号（`v0`, `v1` など）を割り当てます。

例えば、`zoo_model` という artifact を3回リンクすると、`v0`、`v1`、`v2` という3つのエイリアスが生成され、`v2` には `latest` エイリアスも付きます。

## カスタムエイリアス

ユースケースに応じて、特定の artifact バージョンに対して1つまたは複数のカスタムエイリアスを作成できます。例：

- モデルがどのデータセットで学習されたかを識別するために、`dataset_version_v0`、`dataset_version_v1`、`dataset_version_v2` のようなエイリアスを使用できます。
- 最もパフォーマンスの良い artifact モデルのバージョンを追跡するために `best_model` エイリアスを使うのも良いでしょう。

あるRegistryで [Member または Admin のRegistryロール]({{< relref path="guides/core/registry/configure_registry/#registry-roles" lang="ja" >}}) を持つユーザーであれば、リンク済み artifact へのカスタムエイリアスを追加・削除できます。必要に応じて [保護されたエイリアス]({{< relref path="aliases/#protected-aliases" lang="ja" >}}) を設定することで、改変・削除から守りたい artifact バージョンを識別し、ラベル付けできます。

W&B Registry UI または Python SDK でカスタムエイリアスを作成できます。用途に合わせて、下記のタブから該当する手順を選んでください。

{{< tabpane text=true >}}
{{% tab header="W&B Registry" value="app" %}}

1. W&B Registry にアクセスします。
2. コレクション内の **View details** ボタンをクリックします。
3. **Versions** セクションで、対象の artifact バージョンの **View** ボタンをクリックします。
4. **Aliases** フィールド横の **+** ボタンをクリックし、1つ以上のエイリアスを追加します。

{{% /tab %}}

{{% tab header="Python SDK" value="python" %}}
Python SDK で artifact バージョンをコレクションにリンクする際、`alias` パラメータに 1つ以上のエイリアスのリストを引数として渡せます（[`link_artifact()`]({{< relref path="/ref/python/sdk/classes/run.md/#link_artifact" lang="ja" >}}) 参照）。指定したエイリアスが未作成なら、W&B は自動で（[保護されていないエイリアス]({{< relref path="#custom-aliases" lang="ja" >}})）作成します。

以下のコードスニペットは、Python SDK で artifact バージョンをコレクションにリンクし、エイリアスを追加する方法を示します。`<>` 内の値はご自身のものに置き換えてください。

```python
import wandb

# run を初期化
run = wandb.init(entity = "<team_entity>", project = "<project_name>")

# artifact オブジェクトを作成
# type パラメータは artifact オブジェクトとコレクションタイプを指定
artifact = wandb.Artifact(name = "<name>", type = "<type>")

# artifact オブジェクトをファイルに追加
# ローカルマシン上のファイルパスを指定
artifact.add_file(local_path = "<local_path_to_artifact>")

# この artifact をリンクするコレクションとRegistryを指定
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

# artifact バージョンをコレクションにリンク
# この artifact バージョンに1つ以上のエイリアスを追加
run.link_artifact(
    artifact = artifact, 
    target_path = target_path, 
    aliases = ["<alias_1>", "<alias_2>"]
    )
```
{{% /tab %}}
{{< /tabpane >}}

### 保護されたエイリアス
[保護されたエイリアス]({{< relref path="aliases/#protected-aliases" lang="ja" >}}) を使うことで、変更や削除を防ぎたい artifact バージョンをラベル付け・識別できます。たとえば、組織の機械学習本番パイプラインで利用中の artifact バージョンには `production` 保護エイリアスを指定することが推奨されます。

[Registry 管理者]({{< relref path="/guides/core/registry/configure_registry/#registry-roles" lang="ja" >}}) や [サービスアカウント]({{< relref path="/support/kb-articles/service_account_useful" lang="ja" >}})（Admin ロール）であれば、保護エイリアスの作成や artifact バージョンへの追加・削除が可能です。Member や Viewer は保護バージョンのリンク解除や、そのバージョンを含むコレクションの削除はできません。詳細は [Registryアクセスの設定]({{< relref path="/guides/core/registry/configure_registry.md" lang="ja" >}}) をご覧ください。

一般的な保護エイリアス例：

- **Production**: この artifact バージョンが本番利用可能であることを示します。
- **Staging**: この artifact バージョンがテスト環境で準備できていることを示します。

#### 保護エイリアスの作成方法

次の手順で W&B Registry UI から保護エイリアスの作成ができます。

1. Registry アプリにアクセスします。
2. 任意のRegistryを選択します。
3. ページ右上の歯車ボタンを押してRegistryの設定を開きます。
4. **Protected Aliases** セクションで **+** ボタンをクリックし、1つ以上の保護エイリアスを追加します。

作成後、各保護エイリアスは **Protected Aliases** セクションで、錠前アイコン付きのグレーの枠で表示されます。  

{{% alert %}}
保護されていないカスタムエイリアスとは異なり、保護エイリアス作成は W&B Registry UI のみ対応しています（Python SDK からのプログラムによる作成はできません）。artifact バージョンへの保護エイリアス追加は、W&B Registry UI または Python SDK のどちらでも可能です。
{{% /alert %}}

W&B Registry UI から artifact バージョンに保護エイリアスを追加する手順は以下の通りです。

1. W&B Registry にアクセスします。
2. コレクション内の **View details** ボタンをクリックします。
3. **Versions** セクションで、対象 artifact バージョンの **View** ボタンをクリックします。
4. **Aliases** フィールド横の **+** ボタンをクリックし、1つまたは複数の保護エイリアスを追加します。

保護エイリアスを作成後は、管理者権限ユーザーが Python SDK を使用してプログラム上 artifact バージョンに保護エイリアスを追加することもできます。具体的な追加方法は上記 [カスタムエイリアス作成](#custom-aliases) セクション内の W&B Registry および Python SDK タブを参照してください。

## 既存エイリアスの検索
[W&B Registry のグローバル検索バー]({{< relref path="/guides/core/registry/search_registry/#search-for-registry-items" lang="ja" >}}) を使うと、既存エイリアスも検索可能です。保護エイリアスを探す手順は以下の通りです。

1. W&B Registry App にアクセスします。
2. ページ最上部の検索バーに検索したい語句を入力し、Enter キーを押します。

入力した語句が既存のRegistry、コレクション名、artifact バージョンのタグ、コレクションタグ、またはエイリアスと一致した場合、その結果が検索バー下に表示されます。

## 例

{{% alert %}}
以下のコード例は [W&B Registry チュートリアル](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb) の続きです。以下のコードを利用するには、まず [Zoo データセットの取得と処理をノートブックの手順通りに実行](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb#scrollTo=87fecd29-8146-41e2-86fb-0bb4e3e3350a) してください。Zoo データセットが準備できたら、artifact バージョンの作成とカスタムエイリアスの追加が可能です。
{{% /alert %}}

以下のコードスニペットは、artifact バージョンの作成とカスタムエイリアスの追加方法を示したものです。この例では、[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/111/zoo) の Zoo データセットと `Zoo_Classifier_Models` Registryの `Model` コレクションを使用しています。

```python
import wandb

# run を初期化
run = wandb.init(entity = "smle-reg-team-2", project = "zoo_experiment")

# artifact オブジェクト作成
# type パラメータは artifact オブジェクトとコレクションタイプを指定
artifact = wandb.Artifact(name = "zoo_dataset", type = "dataset")

# artifact オブジェクトにファイルを追加
# ローカルマシン上のファイルパスを指定
artifact.add_file(local_path="zoo_dataset.pt", name="zoo_dataset")
artifact.add_file(local_path="zoo_labels.pt", name="zoo_labels")

# この artifact をリンクするコレクションとRegistryを指定
REGISTRY_NAME = "Model"
COLLECTION_NAME = "Zoo_Classifier_Models"
target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

# artifact バージョンをコレクションにリンク
# この artifact バージョンに1つ以上のエイリアスを追加
run.link_artifact(
    artifact = artifact,
    target_path = target_path,
    aliases = ["production-us", "production-eu"]
    )
```

1. まず、artifact オブジェクトを作成します（`wandb.Artifact()`）。
2. 次に、2つの PyTorch tensor データセットを `wandb.Artifact.add_file()` で artifact オブジェクトに追加します。
3. 最後に、`link_artifact()` を用いて artifact バージョンを `Zoo_Classifier_Models` Registry内の `Model` コレクションにリンクします。また、`aliases` パラメータに `production-us` と `production-eu` を渡すことで、2つのカスタムエイリアスも artifact バージョンへ追加されます。