---
title: Databricks
description: W&B を Databricks と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-databricks
    parent: integrations
weight: 50
---

W&B は、Databricks 環境での W&B Jupyter ノートブック体験をカスタマイズすることにより、[Databricks](https://www.databricks.com/) と統合します。

## Databricks の設定

1. クラスターに wandb をインストール

    クラスター設定に移動し、クラスターを選択し、**Libraries** をクリックします。**Install New** をクリックし、**PyPI** を選択してパッケージ `wandb` を追加します。

2. 認証の設定

    あなたの W&B アカウントを認証するために、ノートブックが照会できる Databricks シークレットを追加することができます。

    ```bash
    # databricks cli をインストール
    pip install databricks-cli

    # databricks UIからトークンを生成
    databricks configure --token

    # 2つのコマンドのいずれかでスコープを作成します（databricksでセキュリティ機能が有効かどうかによります）：
    # セキュリティ追加機能あり
    databricks secrets create-scope --scope wandb
    # セキュリティ追加機能なし
    databricks secrets create-scope --scope wandb --initial-manage-principal users

    # こちらから api_key を追加します: https://app.wandb.ai/authorize
    databricks secrets put --scope wandb --key api_key
    ```

## 例

### 簡単な例

```python
import os
import wandb

api_key = dbutils.secrets.get("wandb", "api_key")
wandb.login(key=api_key)

wandb.init()
wandb.log({"foo": 1})
```

### Sweeps

ノートブックが wandb.sweep() または wandb.agent() を使用しようとする際に必要な設定（暫定的）です。

```python
import os

# これらは将来的には不要になります
os.environ["WANDB_ENTITY"] = "my-entity"
os.environ["WANDB_PROJECT"] = "my-project-that-exists"
```