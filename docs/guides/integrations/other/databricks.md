---
slug: /guides/integrations/databricks
description: How to integrate W&B with Databricks.
displayed_sidebar: default
---

# Databricks

W&Bは、Databricks環境でのW&B Jupyterノートブック体験をカスタマイズすることで、[Databricks](https://www.databricks.com/)と統合されます。

## Databricks設定

### クラスターにwandbをインストール

クラスター設定に移動し、クラスターを選択し、ライブラリーをクリックし、新しいインストールを選択します。PyPIを選択し、パッケージ`wandb`を追加します。

### 認証

W&Bアカウントを認証するために、ノートブックでクエリできるdatabricksシークレットを追加できます。

```bash
# databricks cliをインストールする
pip install databricks-cli

# databricks UIからトークンを生成する
databricks configure --token

# 下記の二つのコマンドのうちどちらかでスコープを作成（databricksのセキュリティ機能が有効かどうかによる）：
# セキュリティアドオンがある場合
databricks secrets create-scope --scope wandb
# セキュリティアドオンがない場合
databricks secrets create-scope --scope wandb --initial-manage-principal users
```

# APIキーを追加：https://app.wandb.ai/authorize から取得

databricks secrets put --scope wandb --key api_key

```

## 例

### シンプル

```python
import os
import wandb

api_key = dbutils.secrets.get("wandb", "api_key")
wandb.login(key=api_key)

wandb.init()
wandb.log({"foo": 1})
```

### スイープ

wandb.sweep() や wandb.agent() を使用しようとするノートブック用の一時的なセットアップ：

```python
import os

# これらは将来的に必要がなくなるでしょう。
os.environ["WANDB_ENTITY"] = "my-entity"
os.environ["WANDB_PROJECT"] = "my-project-that-exists"
```