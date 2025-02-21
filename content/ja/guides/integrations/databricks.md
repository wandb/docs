---
title: Databricks
description: W&B を Databricks と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-databricks
    parent: integrations
weight: 50
---

W&B は、Databricks 環境で W&B Jupyter notebook のエクスペリエンスをカスタマイズすることにより、[Databricks](https://www.databricks.com/) と統合されます。

## Databricks の設定

1. クラスター に wandb をインストールする

    クラスター の設定に移動し、 クラスター を選択して、**ライブラリ** をクリックします。**新規インストール** をクリックし、**PyPI** を選択して、パッケージ `wandb` を追加します。

2. 認証の設定

    W&B アカウントを認証するには、notebook がクエリできる Databricks シークレットを追加します。

    ```bash
    # databricks cli をインストール
    pip install databricks-cli

    # databricks UI からトークンを生成
    databricks configure --token

    # 次の 2 つのコマンドのいずれかでスコープを作成します (databricks でセキュリティ機能が有効になっているかどうかによって異なります)。
    # セキュリティアドオンあり
    databricks secrets create-scope --scope wandb
    # セキュリティアドオンなし
    databricks secrets create-scope --scope wandb --initial-manage-principal users

    # https://app.wandb.ai/authorize から api_key を追加します
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

wandb.sweep() または wandb.agent() を使用しようとする notebook に必要な (一時的な) 設定:

```python
import os

# これらは将来的には不要になります
os.environ["WANDB_ENTITY"] = "my-entity"
os.environ["WANDB_PROJECT"] = "my-project-that-exists"
```
