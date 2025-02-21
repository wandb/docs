---
title: Download a model version
description: W&B Python SDK でモデルをダウンロードする方法
menu:
  default:
    identifier: ja-guides-models-registry-model_registry-consume-models
    parent: model-registry
weight: 8
---

W&B Python SDK を使用して、 Model Registry にリンクしたモデル artifact をダウンロードします。

{{% alert %}}
モデルを再構築、デシリアライズして利用できる形式にするための追加の Python 関数、API 呼び出しはお客様が用意する必要があります。

W&B では、モデルをメモリにロードする方法に関する情報をモデルカードに記載することをお勧めします。詳細については、[機械学習モデルのドキュメント]({{< relref path="./create-model-cards.md" lang="ja" >}}) ページを参照してください。
{{% /alert %}}

`<>` 内の 値 を自身の 値 に置き換えます。

```python
import wandb

# run を初期化する
run = wandb.init(project="<project>", entity="<entity>")

# モデルにアクセスしてダウンロードする。ダウンロードした artifact へのパスを返す
downloaded_model_path = run.use_model(name="<your-model-name>")
```

次のいずれかの形式でモデル バージョンを参照します。

* `latest` - 最後にリンクされたモデル バージョンを指定するには、 `latest` エイリアスを使用します。
* `v#` - Registered Model 内の特定の バージョン を取得するには、 `v0` 、 `v1` 、 `v2` などを使用します。
* `alias` - 自身とチームがモデル バージョン に割り当てたカスタム エイリアス を指定します。

可能な パラメータ と戻り値の型の詳細については、API Reference ガイドの[`use_model`]({{< relref path="/ref/python/run.md#use_model" lang="ja" >}})を参照してください。

<details>
<summary> 例: ログに記録されたモデルをダウンロードして使用する </summary>

たとえば、次の コード スニペット では、 ユーザー が `use_model` API を呼び出しました。取得するモデル artifact の名前を指定し、 バージョン / エイリアス も指定しました。次に、API から返された パス を `downloaded_model_path` 変数に格納しました。

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # モデル バージョン のセマンティック ニックネームまたは識別子
model_artifact_name = "fine-tuned-model"

# run を初期化する
run = wandb.init()
# モデルにアクセスしてダウンロードする。ダウンロードした artifact への パス を返す

downloaded_model_path = run.use_model(name=f"{entity/project/model_artifact_name}:{alias}")
```
</details>

{{% alert title="2024 年の W&B Model Registry の計画的非推奨" %}}
次のタブは、間もなく非推奨になる Model Registry を使用してモデル artifact を消費する方法を示しています。

W&B Registry を使用して、モデル artifact を追跡、整理、および消費します。詳細については、[Registry のドキュメント]({{< relref path="/guides/models/registry/" lang="ja" >}})を参照してください。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="CLI" %}}
`<>` 内の 値 を自身の 値 に置き換えます。
```python
import wandb
# run を初期化する
run = wandb.init(project="<project>", entity="<entity>")
# モデルにアクセスしてダウンロードする。ダウンロードした artifact への パス を返す
downloaded_model_path = run.use_model(name="<your-model-name>")
```
次のいずれかの形式でモデル バージョン を参照します。

* `latest` - 最後にリンクされたモデル バージョン を指定するには、 `latest` エイリアス を使用します。
* `v#` - Registered Model 内の特定の バージョン を取得するには、 `v0` 、 `v1` 、 `v2` などを使用します。
* `alias` - 自身とチームがモデル バージョン に割り当てたカスタム エイリアス を指定します。

可能な パラメータ と戻り値の型の詳細については、API Reference ガイドの[`use_model`]({{< relref path="/ref/python/run.md#use_model" lang="ja" >}})を参照してください。
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model) で Model Registry App に移動します。
2. ダウンロードするモデルを含む、登録済みモデルの名前の横にある **詳細を表示** を選択します。
3. [バージョン] セクション内で、ダウンロードするモデル バージョン の横にある [表示] ボタンを選択します。
4. **ファイル** タブを選択します。
5. ダウンロードするモデル ファイル の横にあるダウンロード ボタンをクリックします。
{{< img src="/images/models/download_model_ui.gif" alt="" >}}
  {{% /tab %}}
{{< /tabpane >}}
