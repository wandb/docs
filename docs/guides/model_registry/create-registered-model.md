---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Create a registered model

[Registered model](./model-management-concepts.md#registered-model) を作成して、モデリングタスクの候補モデルを保持します。Model Registry 内で対話型で登録モデルを作成するか、Python SDK を使ってプログラム上で作成できます。

## プログラムで登録モデルを作成する
W&B Python SDK を使ってプログラムでモデルを登録します。登録モデルが存在しない場合、W&B は自動的に登録モデルを作成します。

`<>`で囲まれている他の値を自分の値に置き換えてください:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

`registered_model_name` に指定した名前が [Model Registry App](https://wandb.ai/registry/model) に表示されます。

## 対話型で登録モデルを作成する
[Model Registry App](https://wandb.ai/registry/model) 内で対話型で登録モデルを作成します。

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model) の Model Registry App に移動します。
![モデルを作成](/images/models/create_registered_model_1.png)
2. Model Registry ページの右上にある **New registered model** ボタンをクリックします。
![新しい登録モデル](/images/models/create_registered_model_model_reg_app.png)
3. 表示されたパネルから、登録モデルが所属する Entity を **Owning Entity** ドロップダウンから選択します。
![エンティティを選択](/images/models/create_registered_model_3.png)
4. **Name** フィールドにモデルの名前を入力します。 
5. **Type** ドロップダウンから、登録モデルにリンクするアーティファクトのタイプを選択します。
6. （オプション）**Description** フィールドにモデルに関する説明を追加します。
7. （オプション）**Tags** フィールドに1つ以上のタグを追加します。
8. **Register model** をクリックします。

:::tip
モデルをモデルレジストリに手動でリンクするのは、一度きりのモデルに便利です。しかし、しばしばモデルバージョンをプログラムでモデルレジストリにリンクすることが有用です。

例えば、毎晩実行されるジョブがあるとします。毎晩作成されるモデルを手動でリンクするのは面倒です。その代わりに、モデルを評価し、パフォーマンスが向上した場合に W&B Python SDK でそのモデルをモデルレジストリにリンクするスクリプトを作成できます。
:::
