---
title: Databricks
description: W&B を Databricks と統合する方法
menu:
  default:
    identifier: ja-guides-integrations-databricks
    parent: integrations
weight: 50
---

W&B は、Databricks 環境で W&B Jupyter ノートブックの体験をカスタマイズすることで [Databricks](https://www.databricks.com/) と統合します。

## Databricks の設定

1. クラスターに wandb をインストール

    クラスターの設定画面に移動し、対象クラスターを選択して **Libraries** をクリックします。**Install New** をクリックし、**PyPI** を選択し、`wandb` パッケージを追加してください。

2. 認証の設定

    W&B アカウントを認証するには、Databricks シークレットを追加し、ノートブックから参照できるようにします。

    ```bash
    # databricks cli をインストール
    pip install databricks-cli

    # databricks の UI でトークンを生成
    databricks configure --token

    # 以下のいずれかのコマンドでスコープを作成（databricks のセキュリティ機能有無に応じて選択）
    # セキュリティアドオンありの場合
    databricks secrets create-scope --scope wandb
    # セキュリティアドオンなしの場合
    databricks secrets create-scope --scope wandb --initial-manage-principal users

    # https://app.wandb.ai/authorize から取得した api_key を追加
    databricks secrets put --scope wandb --key api_key
    ```

## 例

### シンプルな例

```python
import os
import wandb

api_key = dbutils.secrets.get("wandb", "api_key")
wandb.login(key=api_key)

with wandb.init() as run:
    run.log({"foo": 1})
```

### Sweeps

wandb.sweep() や wandb.agent() をノートブックで使う際に（暫定的に）必要な設定:

```python
import os

# 今後不要となる予定の設定です
os.environ["WANDB_ENTITY"] = "my-entity"
os.environ["WANDB_PROJECT"] = "my-project-that-exists"
```