---
description: W&B を Databricks と統合する方法
slug: /guides/integrations/databricks
displayed_sidebar: default
---


# Databricks

W&Bは、Databricksの環境でW&B Jupyterノートブックの体験をカスタマイズすることで[Databricks](https://www.databricks.com/)と統合します。

## Databricks設定

### クラスターにwandbをインストール

クラスター設定に移動し、クラスターを選択し、ライブラリをクリックしてから、新規インストールをクリック、PyPIを選択し、パッケージ `wandb` を追加します。

### 認証

W&Bアカウントを認証するには、ノートブックからクエリできるDatabricksシークレットを追加します。

```bash
# databricks cli をインストール
pip install databricks-cli

# Databricks UIからトークンを生成
databricks configure --token

# 次の2つのコマンドのいずれかでスコープを作成（databricksでセキュリティ機能が有効になっているかどうかに依存します）:
# セキュリティアドオンあり
databricks secrets create-scope --scope wandb
# セキュリティアドオンなし
databricks secrets create-scope --scope wandb --initial-manage-principal users

# こちらからapi_keyを追加: https://app.wandb.ai/authorize
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

ノートブックでwandb.sweep()やwandb.agent()を使用するための設定（暫定）

```python
import os

# これらは将来的には必要なくなります
os.environ["WANDB_ENTITY"] = "my-entity"
os.environ["WANDB_PROJECT"] = "my-project-that-exists"
```