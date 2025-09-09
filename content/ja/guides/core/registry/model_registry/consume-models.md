---
title: Model Version をダウンロードする
description: W&B の Python SDK でモデルをダウンロードする方法
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-consume-models
    parent: model-registry
weight: 8
---

W&B Python SDK を使用して、Model Registry にリンクされた Models Artifact をダウンロードします。

{{% alert %}}
モデルを実用的な形式に再構築・逆シリアル化するための追加の Python 関数や API 呼び出しは、ユーザーの責任です。

W&B は、モデルカードにモデルのメモリへのロード方法を記載することを推奨します。詳細については、[機械学習モデルを文書化する]({{< relref path="./create-model-cards.md" lang="ja" >}}) ページを参照してください。
{{% /alert %}}

`<pre>` 内の値を独自のデータに置き換えてください。

```python
import wandb

# Runs を初期化
run = wandb.init(project="<project>", entity="<entity>")

# Models にアクセスしてダウンロードします。ダウンロードされた Artifact へのパスを返します
downloaded_model_path = run.use_model(name="<your-model-name>")
```

以下のいずれかの形式でモデル バージョンを参照します。

*   `latest` - `latest` エイリアスを使用して、最新でリンクされたモデル バージョンを指定します。
*   `v#` - `v0`、`v1`、`v2` などを使用して、Registered Models 内の特定のバージョンを取得します。
*   `alias` - あなたとあなたのチームがモデル バージョンに割り当てたカスタム エイリアスを指定します。

可能なパラメータと戻り値の型に関する詳細については、API リファレンスガイドの[`use_model`]({{< relref path="/ref/python/sdk/classes/run.md#use_model" lang="ja" >}}) を参照してください。

<details>
<summary>例: ログに記録したモデルをダウンロードして使用する</summary>

例えば、次のコード スニペットでは `use_model` API を呼び出し、取得したい Models Artifact の名前とバージョン/エイリアスを指定しています。API から返されたパスは `downloaded_model_path` 変数に格納されます。

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # モデル バージョンのニックネームまたは識別子
model_artifact_name = "fine-tuned-model"

# Runs を初期化
run = wandb.init()
# Models にアクセスしてダウンロードします。ダウンロードされた Artifact へのパスを返します

downloaded_model_path = run.use_model(name=f"{entity/project/model_artifact_name}:{alias}")
```
</details>

{{% alert title="W&B Model Registry の 2024 年の廃止予定" %}}
以下のタブは、間もなく廃止される Model Registry を使用して Models Artifacts を利用する方法を示しています。

W&B Registry を使用して、Models Artifacts を追跡、整理、利用します。詳細については、[Registry ドキュメント]({{< relref path="/guides/core/registry/" lang="ja" >}}) を参照してください。
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="CLI" %}}
`<pre>` 内の値を独自のデータに置き換えてください。
```python
import wandb
# Runs を初期化
run = wandb.init(project="<project>", entity="<entity>")
# Models にアクセスしてダウンロードします。ダウンロードされた Artifact へのパスを返します
downloaded_model_path = run.use_model(name="<your-model-name>")
```
以下のいずれかの形式でモデル バージョンを参照します。

*   `latest` - `latest` エイリアスを使用して、最新でリンクされたモデル バージョンを指定します。
*   `v#` - `v0`、`v1`、`v2` などを使用して、Registered Models 内の特定のバージョンを取得します。
*   `alias` - あなたとあなたのチームがモデル バージョンに割り当てたカスタム エイリアスを指定します。

可能なパラメータと戻り値の型に関する詳細については、API リファレンスガイドの[`use_model`]({{< relref path="/ref/python/sdk/classes/run.md#use_model" lang="ja" >}}) を参照してください。
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1.  [Model Registry App](https://wandb.ai/registry/model) に移動します。
2.  ダウンロードしたいモデルを含む Registered Models の名前の横にある「**View details**」を選択します。
3.  「Versions」セクション内で、ダウンロードしたいモデル バージョンの横にある「View」ボタンを選択します。
4.  「**Files**」タブを選択します。
5.  ダウンロードしたいモデル ファイルの横にあるダウンロード ボタンをクリックします。
{{< img src="/images/models/download_model_ui.gif" alt="UI からモデルをダウンロード" >}}
  {{% /tab %}}
{{< /tabpane >}}