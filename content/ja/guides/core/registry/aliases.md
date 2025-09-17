---
title: Artifacts のバージョンをエイリアスで参照する
menu:
  default:
    identifier: ja-guides-core-registry-aliases
weight: 5
---

特定の [アーティファクトのバージョン]({{< relref path="guides/core/artifacts/create-a-new-artifact-version" lang="ja" >}}) を1つ以上のエイリアスで参照します。[W&B は、同じ名前でリンクする各アーティファクトに自動的にエイリアスを割り当てます]({{< relref path="aliases#default-aliases" lang="ja" >}})。また、特定のアーティファクトのバージョンを参照するために、[1つ以上のカスタムエイリアスを作成する]({{< relref path="aliases#custom-aliases" lang="ja" >}}) こともできます。

エイリアスは、Registry UI に、エイリアス名が記載された長方形として表示されます。エイリアスが [保護されている]({{< relref path="aliases#protected-aliases" lang="ja" >}}) 場合、ロックアイコンが付いた灰色の長方形として表示されます。それ以外の場合、エイリアスはオレンジ色の長方形として表示されます。エイリアスは、複数のレジストリ間で共有されません。

{{% alert title="エイリアスとタグの使い分け" %}}
特定のアーティファクトのバージョンを参照するには、エイリアスを使用します。コレクション内の各エイリアスは一意です。特定のエイリアスを持てるアーティファクトのバージョンは、一度に1つだけです。

タグは、共通のテーマに基づいてアーティファクトのバージョンやコレクションを整理し、グループ化するために使用します。複数のアーティファクトのバージョンやコレクションが同じタグを共有できます。
{{% /alert %}}

アーティファクトのバージョンにエイリアスを追加する際、オプションで [Registry automation]({{< relref path="/guides/core/automations/automation-events/#registry" lang="ja" >}}) を開始して、Slack チャンネルに通知したり、Webhook をトリガーしたりできます。

## デフォルトのエイリアス

W&B は、同じ名前でリンクする各アーティファクトのバージョンに、以下のエイリアスを自動的に割り当てます。

* コレクションにリンクする最も新しいアーティファクトのバージョンには、`latest` エイリアスを割り当てます。
* 一意のバージョン番号。W&B は、リンクする各アーティファクトのバージョンを（ゼロから数えて）カウントします。W&B は、そのカウント番号を使用して、そのアーティファクトに一意のバージョン番号を割り当てます。

例えば、`zoo_model` という名前のアーティファクトを3回リンクした場合、W&B はそれぞれ `v0`、`v1`、`v2` の3つのエイリアスを作成します。`v2` は `latest` エイリアスも持ちます。

## カスタムエイリアス

独自のユースケースに基づいて、特定のアーティファクトのバージョンに1つ以上のカスタムエイリアスを作成します。例：

- `dataset_version_v0`、`dataset_version_v1`、`dataset_version_v2` のようなエイリアスを使用して、モデルがどのデータセットで学習されたかを識別することができます。
- `best_model` エイリアスを使用して、最もパフォーマンスの高いアーティファクトのモデルのバージョンを追跡することができます。

レジストリ上の [**Member** または **Admin** レジストリロール]({{< relref path="guides/core/registry/configure_registry/#registry-roles" lang="ja" >}}) を持つユーザーは、そのレジストリでリンクされたアーティファクトからカスタムエイリアスを追加または削除できます。[**Restricted Viewer** または **Viewer** ロール]({{< relref path="guides/core/registry/configure_registry/#registry-roles" lang="ja" >}}) を持つユーザーは、エイリアスを追加または削除できません。

{{% alert %}}
[保護されたエイリアス]({{< relref path="aliases/#protected-aliases" lang="ja" >}}) は、変更または削除から保護すべきアーティファクトのバージョンをラベル付けし、識別する方法を提供します。
{{% /alert %}}

カスタムエイリアスは、W&B Registry または Python SDK を使用して作成できます。ユースケースに応じて、以下でニーズに最も合ったタブをクリックしてください。

{{< tabpane text=true >}}
{{% tab header="W&B Registry" value="app" %}}

1. W&B Registry に移動します。
2. コレクション内の **View details** ボタンをクリックします。
3. **Versions** セクションで、特定のアーティファクトのバージョンに対する **View** ボタンをクリックします。
4. **Aliases** フィールドの横にある **+** ボタンをクリックして、1つ以上のエイリアスを追加します。

{{% /tab %}}

{{% tab header="Python SDK" value="python" %}}
Python SDK を使用してアーティファクトのバージョンをコレクションにリンクする際、オプションで1つ以上のエイリアスのリストを [`link_artifact()`]({{< relref path="/ref/python/sdk/classes/run.md/#link_artifact" lang="ja" >}}) の `alias` パラメータへの引数として提供できます。提供されたエイリアスがまだ存在しない場合、W&B はエイリアス（[保護されていないエイリアス]({{< relref path="#custom-aliases" lang="ja" >}})）を作成します。

以下のコードスニペットは、Python SDK を使用してアーティファクトのバージョンをコレクションにリンクし、そのアーティファクトのバージョンにエイリアスを追加する方法を示しています。 `<>` 内の値を独自の値に置き換えてください。

```python
import wandb

# run を初期化する
run = wandb.init(entity = "<team_entity>", project = "<project_name>")

# アーティファクトオブジェクトを作成する
# type パラメータは、
# アーティファクトオブジェクトのタイプとコレクションタイプ
# の両方を指定します。
artifact = wandb.Artifact(name = "<name>", type = "<type>")

# ファイルをアーティファクトオブジェクトに追加する。
# ローカルマシン上のファイルへのパスを指定する。
artifact.add_file(local_path = "<local_path_to_artifact>")

# アーティファクトをリンクするコレクションとレジストリを指定する
REGISTRY_NAME = "<registry_name>"
COLLECTION_NAME = "<collection_name>"
target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

# アーティファクトのバージョンをコレクションにリンクする
# このアーティファクトのバージョンに1つ以上のエイリアスを追加する
run.link_artifact(
    artifact = artifact, 
    target_path = target_path, 
    aliases = ["<alias_1>", "<alias_2>"]
    )
```
{{% /tab %}}
{{< /tabpane >}}

### 保護されたエイリアス
[保護されたエイリアス]({{< relref path="aliases/#protected-aliases" lang="ja" >}}) を使用して、変更または削除すべきではないアーティファクトのバージョンにラベルを付け、識別します。例えば、`production` 保護エイリアスを使用して、組織の機械学習プロダクションパイプラインで使用されているアーティファクトのバージョンをラベル付けし、識別することを検討してください。

**Admin** ロールを持つ [Registry admin]({{< relref path="/guides/core/registry/configure_registry/#registry-roles" lang="ja" >}}) ユーザーおよび [サービスアカウント]({{< relref path="/support/kb-articles/service_account_useful" lang="ja" >}}) は、保護されたエイリアスを作成し、アーティファクトのバージョンから保護されたエイリアスを追加または削除できます。**Member**、**Viewer**、**Restricted Viewer** ロールを持つユーザーおよびサービスアカウントは、保護されたバージョンをリンク解除したり、保護されたエイリアスを含むコレクションを削除したりすることはできません。詳細については、[レジストリへのアクセス設定]({{< relref path="/guides/core/registry/configure_registry.md" lang="ja" >}}) を参照してください。

一般的な保護されたエイリアスには、以下が含まれます。

- **Production**: そのアーティファクトのバージョンは、プロダクション環境での使用準備ができています。
- **Staging**: そのアーティファクトのバージョンは、テストの準備ができています。

#### 保護されたエイリアスを作成する

以下の手順は、W&B Registry UI で保護されたエイリアスを作成する方法を示しています。

1. Registry App に移動します。
2. レジストリを選択します。
3. ページ右上の歯車ボタンをクリックして、レジストリの設定を表示します。
4. **Protected Aliases** セクションで、**+** ボタンをクリックして1つ以上の保護されたエイリアスを追加します。

作成後、各保護されたエイリアスは、**Protected Aliases** セクションにロックアイコンが付いた灰色の長方形として表示されます。

{{% alert %}}
保護されていないカスタムエイリアスとは異なり、保護されたエイリアスの作成は W&B Registry UI でのみ利用可能であり、Python SDK を使用してプログラムで作成することはできません。アーティファクトのバージョンに保護されたエイリアスを追加するには、W&B Registry UI または Python SDK を使用できます。
{{% /alert %}}

以下の手順は、W&B Registry UI を使用してアーティファクトのバージョンに保護されたエイリアスを追加する方法を示しています。

1. W&B Registry に移動します。
2. コレクション内の **View details** ボタンをクリックします。
3. **Versions** セクションで、特定のアーティファクトのバージョンに対する **View** ボタンを選択します。
4. **Aliases** フィールドの横にある **+** ボタンをクリックして、1つ以上の保護されたエイリアスを追加します。

保護されたエイリアスが作成された後、管理者は Python SDK を使用してプログラムでそれをアーティファクトのバージョンに追加できます。アーティファクトのバージョンに保護されたエイリアスを追加する方法の例については、上記の [カスタムエイリアスを作成する](#custom-aliases) セクションにある W&B Registry および Python SDK タブを参照してください。

## 既存のエイリアスを見つける
既存のエイリアスは、[W&B Registry のグローバル検索バー]({{< relref path="/guides/core/registry/search_registry/#search-for-registry-items" lang="ja" >}}) で見つけることができます。保護されたエイリアスを見つけるには：

1. W&B Registry App に移動します。
2. ページ上部の検索バーに検索語を入力します。Enter キーを押して検索します。

指定した検索語が既存のレジストリ、コレクション名、アーティファクトのバージョンタグ、コレクションタグ、またはエイリアスと一致する場合、検索結果は検索バーの下に表示されます。

## 例

{{% alert %}}
以下のコード例は、[W&B Registry チュートリアル](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb) の続きです。以下のコードを使用するには、まず [ノートブックで説明されているように Zoo データセットを取得して処理する](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb#scrollTo=87fecd29-8146-41e2-86fb-0bb4e3e3350a) 必要があります。Zoo データセットを入手したら、アーティファクトのバージョンを作成し、それにカスタムエイリアスを追加できます。
{{% /alert %}}

以下のコードスニペットは、アーティファクトのバージョンを作成し、それにカスタムエイリアスを追加する方法を示しています。この例では、[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/111/zoo) の Zoo データセットと、`Zoo_Classifier_Models` レジストリの `Model` コレクションを使用しています。

```python
import wandb

# run を初期化する
run = wandb.init(entity = "smle-reg-team-2", project = "zoo_experiment")

# アーティファクトオブジェクトを作成する
# type パラメータは、
# アーティファクトオブジェクトのタイプとコレクションタイプ
# の両方を指定します。
artifact = wandb.Artifact(name = "zoo_dataset", type = "dataset")

# ファイルをアーティファクトオブジェクトに追加する。
# ローカルマシン上のファイルへのパスを指定する。
artifact.add_file(local_path="zoo_dataset.pt", name="zoo_dataset")
artifact.add_file(local_path="zoo_labels.pt", name="zoo_labels")

# アーティファクトをリンクするコレクションとレジストリを指定する
REGISTRY_NAME = "Model"
COLLECTION_NAME = "Zoo_Classifier_Models"
target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

# アーティファクトのバージョンをコレクションにリンクする
# このアーティファクトのバージョンに1つ以上のエイリアスを追加する
run.link_artifact(
    artifact = artifact,
    target_path = target_path,
    aliases = ["production-us", "production-eu"]
    )
```

1. まず、アーティファクトオブジェクト（`wandb.Artifact()`）を作成します。
2. 次に、`wandb.Artifact.add_file()` を使用して、2つのデータセット PyTorch テンソルをアーティファクトオブジェクトに追加します。
3. 最後に、`link_artifact()` を使用して、アーティファクトのバージョンを `Zoo_Classifier_Models` レジストリの `Model` コレクションにリンクします。また、`aliases` パラメータに `production-us` と `production-eu` を引数として渡すことで、2つのカスタムエイリアスをアーティファクトのバージョンに追加します。