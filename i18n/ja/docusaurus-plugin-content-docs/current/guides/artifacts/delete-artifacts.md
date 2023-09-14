---
description: >-
  Delete artifacts interactively with the App UI or programmatically with the
  Weights & Biases SDK/
displayed_sidebar: ja
---

# アーティファクトの削除

<head>
  <title>W&Bアーティファクトの削除</title>
</head>
アプリのUIやWeights＆Biases SDKを使って、アーティファクトを対話式に削除したりプログラムで削除できます。Weights & Biases は、アーティファクトとそれに関連するファイルが、前のアーティファクトバージョンまたは後のアーティファクトバージョンで使用されていないかどうか確認してから、アーティファクトを削除します。特定のアーティファクトバージョンを削除することも、アーティファクト全体を削除することもできます。

アーティファクトを削除する前にエイリアスを削除することも、API呼び出しに追加のフラグを渡してアーティファクトを削除することもできます。削除したいアーティファクトに関連するエイリアスを削除することが推奨されます。

アーティファクトを更新するドキュメントを参照して、W&B SDKやアプリUIを使ってエイリアスをプログラムで更新したり対話式に更新する方法について調べてください。
### アーティファクトバージョンの削除

アーティファクトバージョンを削除するには：

1. アーティファクトの名前を選択します。これにより、アーティファクトビューが展開され、そのアーティファクトに関連付けられたすべてのアーティファクトバージョンが表示されます。
2. アーティファクトのリストから、削除するアーティファクトバージョンを選択します。
3. ワークスペースの右側にあるケバブドロップダウンを選択します。
4. 削除を選択します。
### エイリアスを持つ複数のアーティファクトを削除する

次のコード例は、エイリアスが関連付けられたアーティファクトを削除する方法を示しています。アーティファクトを作成したエンティティ、プロジェクト名、およびrun IDを提供してください。

```python
import wandb
```
run = api.run('entity/project/run_id')

アーティファクトが1つ以上のエイリアスを持っている場合、`delete_aliases`パラメータをブール値`True`に設定してエイリアスを削除します。

```python
for artifact in run.logged_artifacts():
    artifact.delete(delete_aliases=True)
```

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    # エイリアスが1つ以上あるアーティファクトを削除するためには、
    # delete_aliases=True を指定します
    artifact.delete(delete_aliases=True)
```
### 特定のエイリアスを持つ複数のアーティファクトバージョンを削除する

以下のコードは、特定のエイリアスを持つ複数のアーティファクトバージョンを削除する方法を示しています。アーティファクトを作成したエンティティ、プロジェクト名、およびrun IDを指定してください。削除ロジックは独自のものに置き換えてください。

```python
import wandb
```
runs = api.run('entity/project_name/run_id')

# エイリアス 'v3' と 'v4' のアーティファクトを削除する
for artifact_version in runs.logged_artifacts():
  # ここに独自の削除ロジックを書く。
  if artifact_version.name[-2:] == 'v3' or artifact_version.name[-2:] == 'v4':
    artifact.delete(delete_aliases=True)
```

### エイリアスのないアーティファクトのすべてのバージョンを削除する

次のコードスニペットは、エイリアスがないアーティファクトのすべてのバージョンを削除する方法を示しています。`wandb.Api`の`project`と`entity`キーにそれぞれプロジェクト名とエンティティ名を指定してください:

```python
import wandb
```

# wandb.Apiメソッドを使用するときに、
# エンティティとプロジェクト名を提供してください。
api = wandb.Api(overrides={
        "project": "プロジェクト", 
        "entity": "エンティティ"
        })
エイリアス名, アーティファクト名 = ... # タイプと名前を提供してください
for v in api.artifact_versions(artifact_type, artifact_name):
  # 'latest' のようなエイリアスがないバージョンをクリーンアップします。
	# 注: 好きな削除ロジックをここに記述できます。
  if len(v.aliases) == 0:
      v.delete()
```