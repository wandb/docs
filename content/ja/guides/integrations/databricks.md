---
title: Databricks
description: W&B と Databricks を連携する方法
menu:
  default:
    identifier: ja-guides-integrations-databricks
    parent: integrations
weight: 50
---

W&B は、Databricks 環境内で W&B の Jupyter Notebook 体験をカスタマイズすることで、[Databricks](https://www.databricks.com/) と統合します。

## Databricks の設定

1. クラスターに wandb をインストール

    クラスターの設定に移動し、クラスターを選択し、**Libraries** をクリックします。**Install New** をクリックし、**PyPI** を選択して、パッケージ `wandb` を追加します。

2. 認証を設定

    W&B アカウントを認証するには、ノートブックから参照できる Databricks シークレットを追加します。

    ```bash
    # Databricks CLI をインストール
    pip install databricks-cli

    # Databricks UI からトークンを生成
    databricks configure --token

    # (Databricks でセキュリティ機能が有効かどうかに応じて) 以下のいずれかのコマンドでスコープを作成:
    # セキュリティ アドオンを使用する場合
    databricks secrets create-scope --scope wandb
    # セキュリティ アドオンを使用しない場合
    databricks secrets create-scope --scope wandb --initial-manage-principal users

    # https://app.wandb.ai/authorize で取得した api_key を追加
    databricks secrets put --scope wandb --key api_key
    ```

## 例

### 簡単な例

```python
import os
import wandb

api_key = dbutils.secrets.get("wandb", "api_key")
wandb.login(key=api_key)

with wandb.init() as run:
    run.log({"foo": 1})
```

### Sweeps

`wandb.sweep()` または `wandb.agent()` を使うノートブックで必要な (一時的な) 設定:

```python
import os

# これらは将来的に不要になります
os.environ["WANDB_ENTITY"] = "my-entity"
os.environ["WANDB_PROJECT"] = "my-project-that-exists"
```