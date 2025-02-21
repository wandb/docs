---
title: Databricks
description: Databricks と W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-databricks
    parent: integrations
weight: 50
---

W&B は Databricks 環境で W&B Jupyter ノートブックの体験をカスタマイズすることで [Databricks](https://www.databricks.com/) と統合します。

## Databricks の設定

1. クラスターに wandb をインストール

    クラスター設定に移動し、クラスターを選択、**Libraries** をクリックします。**Install New** をクリックし、**PyPI** を選択して、パッケージ `wandb` を追加します。

2. 認証の設定

    W&B アカウントを認証するために、ノートブックがクエリできる Databricks のシークレットを追加することができます。

    ```bash
    # databricks cli をインストール
    pip install databricks-cli

    # databricks UI からトークンを生成
    databricks configure --token

    # 次の 2 つのコマンドのいずれかでスコープを作成 (databricks でセキュリティ機能が有効になっているかどうかに依存):
    # セキュリティ追加機能あり
    databricks secrets create-scope --scope wandb
    # セキュリティ追加機能なし
    databricks secrets create-scope --scope wandb --initial-manage-principal users

    # あなたの api_key を追加: https://app.wandb.ai/authorize
    databricks secrets put --scope wandb --key api_key
    ```

## 例

### シンプルな例

```python
import os
import wandb

api_key = dbutils.secrets.get("wandb", "api_key")
wandb.login(key=api_key)

wandb.init()
wandb.log({"foo": 1})
```

### Sweeps

wandb.sweep() または wandb.agent() を使用しようとするノートブックのために必要な設定 (一時的):

```python
import os

# これらは将来的には不要になります
os.environ["WANDB_ENTITY"] = "my-entity"
os.environ["WANDB_PROJECT"] = "my-project-that-exists"
```