---
title: Databricks
description: W&B を Databricks と統合する方法
menu:
  default:
    identifier: databricks
    parent: integrations
weight: 50
---

W&B は、Databricks 環境内の W&B Jupyter ノートブック体験をカスタマイズすることで [Databricks](https://www.databricks.com/) と統合します。

## Databricks の設定

1. クラスターに wandb をインストールする

    クラスターの設定画面に移動し、クラスターを選択して **Libraries** をクリックします。**Install New** をクリックし、**PyPI** を選択、パッケージに `wandb` を追加してください。

2. 認証を設定する

    W&B アカウントを認証するには Databricks のシークレットを追加し、ノートブックでその値を参照できるようにします。

    ```bash
    # databricks cli をインストール
    pip install databricks-cli

    # databricks の UI からトークンを生成
    databricks configure --token

    # （セキュリティ機能の有無によって）どちらかのコマンドでスコープを作成
    # セキュリティアドオンがある場合
    databricks secrets create-scope --scope wandb
    # セキュリティアドオンがない場合
    databricks secrets create-scope --scope wandb --initial-manage-principal users

    # api_key を https://app.wandb.ai/authorize から取得し追加
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

notebook で wandb.sweep() や wandb.agent() を使用する場合の（一時的な）セットアップ：

```python
import os

# 今後不要になる予定
os.environ["WANDB_ENTITY"] = "my-entity"
os.environ["WANDB_PROJECT"] = "my-project-that-exists"
```