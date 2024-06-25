---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# モデルバージョンのダウンロード

W&B Python SDKを使用して、Model Registryにリンクされたモデルアーティファクトをダウンロードします。モデルのダウンロードは、将来モデルのパフォーマンスを評価したり、データセットで予測を行ったり、モデルをプロダクションに送り込んだりする場合に特に有用です。

:::info
あなた自身がモデルを再構築し、逆シリアル化し、作業できる形式にするための追加のPython関数やAPIコールを提供する責任があります。

W&Bは、モデルカードでモデルをメモリにロードする方法を文書化することを提案しています。詳細については、[Document machine learning models](./create-model-cards.md)ページを参照してください。
:::

`<>`内の値を自分のものに置き換えてください：

```python
import wandb

# runを初期化
run = wandb.init(project="<project>", entity="<entity>")

# モデルにアクセスしてダウンロードします。ダウンロードされたアーティファクトへのパスを返します
downloaded_model_path = run.use_model(name="<your-model-name>")
```

次に示す形式の1つを使用してモデルバージョンを参照できます：

* `latest` - 最新のリンクされたモデルバージョンを指定するには`latest`エイリアスを使用します。
* `v#` - `v0`、`v1`、`v2`などを使用して、Registered Modelの特定のバージョンを取得します。
* `alias` - あなたやチームがモデルバージョンに割り当てたカスタムエイリアスを指定します。

可能なパラメータおよび戻り値の詳細については、APIリファレンスガイドの[`use_model`](../../ref/python/run.md#use_model)を参照してください。

<details>

<summary>例: ログされたモデルをダウンロードして使用する</summary>

例えば、次のコードスニペットではユーザーが`use_model` APIを呼び出しました。取得したいモデルアーティファクトの名前とバージョン/エイリアスを指定し、APIから返されたパスを`downloaded_model_path`変数に保存しました。

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # モデルバージョンのセマンティックニックネームまたは識別子
model_artifact_name = "fine-tuned-model"

# runを初期化
run = wandb.init()
# モデルにアクセスしてダウンロードします。ダウンロードされたアーティファクトへのパスを返します

downloaded_model_path = run.use_model(name=f"{entity/project/model_artifact_name}:{alias}")
```
</details>