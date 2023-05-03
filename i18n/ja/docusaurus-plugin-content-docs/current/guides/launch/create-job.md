---
description: W&B Launchでジョブを作成する方法を学びましょう。
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# ジョブを作成する

ジョブとは、MLのワークフローでステップを実行する方法の完全な設計図であり、モデルのトレーニング、評価の実行、または推論サーバーへのモデルのデプロイなどが含まれます。詳細については、[launched jobsの詳細セクション](launch-jobs#view-details-of-launched-jobs)を参照してください。

:::info
ジョブを作成するには、`wandb>=0.13.8`が必要です。
:::

## ジョブを作成する方法は？

W&Bでトラッキングするrunのソースコードもトラッキングしていれば、ジョブは**自動的にキャプチャされます**。 runのソースコードを以下のようにしてトラッキングできます。

<Tabs
  defaultValue="git"
  values={[
    {label: 'Git', value: 'git'},
    {label: 'Code artifact', value: 'artifact'},
    {label: 'Docker image', value: 'docker'},
  ]}>

<TabItem value="artifact">

`run.log_code()`を呼び出すことで、コードをArtifactとしてrunに記録します。
例：
```python
run = wandb.init()
run.log_code(".", include_fn=lambda path: path.endswith(".py"))
```

</TabItem>

<TabItem value="git">

実行とgitコミットを関連付けます。`wandb.init()`がリモートURLからクローンされたgitリポジトリ内のコードで呼び出されると、W&Bは自動的にコミットIDと差分を取得します。リポジトリに`requirements.txt`ファイルが含まれている場合、ランチエージェントはジョブを実行する際に指定された依存関係をインストールします。

</TabItem>

<TabItem value="docker">

実行をDockerイメージと関連付けます。W&Bは`WANDB_DOCKER`環境変数内のイメージタグを探し、`WANDB_DOCKER`が設定されている場合、指定されたイメージタグからジョブが作成されます。Dockerイメージからジョブを起動すると、ランチエージェントは指定されたイメージを実行します。

`WANDB_DOCKER`の詳細については、[docker integration](/docs/guides/integrations/other/docker.md)のドキュメントを参照してください。

`WANDB_DOCKER`環境変数に完全なイメージタグを設定して、ランチエージェントがアクセスできるようにしてください。たとえば、エージェントがECRリポジトリからイメージを実行する場合、`WANDB_DOCKER`には、ECRリポジトリのURLを含めた完全なイメージタグを設定してください。例：`123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest`。

最初のコンテナイメージソースジョブを作成するには、次のコマンドを実行してください：
```bash
docker run -e WANDB_DOCKER=<image-tag> <image-tag>
```

</TabItem>

</Tabs>
## ジョブ命名規則

デフォルトでは、W&Bは自動的にジョブ名を生成します。名前は、ジョブがどのように作成されるか（GitHub、コードアーティファクト、またはDockerイメージ）によって生成されます。以下の表は、各ジョブソースで使用されるジョブ命名規則を説明しています：

| ソース         | 命名規則                                 |
| ------------ | --------------------------------------- |
| GitHub       | `job-<git-remote-url>-<path-to-script>` |
| コードアーティファクト | `job-<code-artifact-name>`              |
| Dockerイメージ   | `job-<image-name>`                      |

## コードをジョブに適したものにする

ジョブは`wandb.config`内の値によってパラメータ化されます。ジョブが実行されて`wandb.init`が呼び出されると、`wandb.config`には、この実行で指定された値が格納されます。これは、実行間で変更可能なすべてのパラメータを格納およびアクセスするために`wandb.config`を使用することが重要であることを意味します。

別の方法でパラメータを読み取り、保存している場合でも、W&B Launchを使用してジョブを実行することができますが、`wandb.config`を使用するようにコードを更新する必要があります。

例えば、コマンドライン引数を解析するために`argparse`を使用している場合、引数の値を格納するために`wandb.config`を使用するようにスクリプトを簡単に適応させることができます。例：

```python
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

# 引数をwandb.initに渡して、wandb.configに保存し、
# ランチエージェントがそれらを変更できるようにします。
wandb.init(config=args)
# wandb.configの値をargsの代わりに使用する。
# wandb.configはドットと辞書のスタイルアクセスに対応しています。
# 例：wandb.config.learning_rate または
# wandb.config["learning_rate"]。
args = wandb.config
```

## ジョブを表示する
W&Bアプリで起動したジョブの詳細を表示します。

### 各ジョブの詳細を表示する

W&Bプロジェクトに移動して、ジョブによって作成されたrunsや、ジョブのフルネーム、プロジェクトに関連するバージョンのメタデータなど、各ジョブの詳細を細かく確認します。

1. あなたのW&Bプロジェクトに移動します。
2. 左サイドバーの**ジョブ**アイコンを選択します。
3. **ジョブ**ページが表示されます。このページで、そのプロジェクトで作成されたすべてのジョブを表示することができます。

![](/images/launch/view_jobs.png)

例えば、以下の画像では、二つのジョブがリストされています。
- **job-https___github.com_githubrepo_demo_launch.git_canonical_job_example.py**
- **job-source-launch_demo-canonical_job_example.py**

リストからジョブを選択して、そのジョブについて詳しく調べます。ジョブとバージョンの詳細とともに、ジョブが作成したrunsのリストが表示される新しいページが開きます。この情報は、**Runs**、**ジョブの詳細**、**バージョンの詳細**の3つのタブに収められています。

<Tabs
  defaultValue="runs"
  values={[
    {label: 'Runs', value: 'runs'},
    {label: 'ジョブの詳細', value: 'jobs_details'},
    {label: 'バージョンの詳細', value: 'version_details'},
  ]}>
  <TabItem value="runs">
Runsタブは、ジョブによって作成された各runに関する情報を提供します。例えば:

- **Run**: Runの名前。
- **State**: Runの状態。
- **Job version**: 使用されたジョブのバージョン。
- **Creator**: Runを作成した人。
- **Creation date**: Runが開始されたときのタイムスタンプ。
- **Other**: 残りの列には、`wandb.config`のキーと値のペアが含まれます。

![](/images/launch/runs_in_job.png)

  </TabItem>
  <TabItem value="jobs_details">

**Job details** では以下の情報が提供されます:

* **Description**: ジョブのオプション説明。このフィールドの隣にある鉛筆アイコンを選択して説明を追加します。
* **Owner entity**: ジョブが属するエンティティ。
* **Parent project**: ジョブが属するプロジェクト。
* **Full name**: ジョブのフルネーム
* **Creation date**: ジョブの作成日。

![](/images/launch/job_id_full_name.png)

  </TabItem>
  <TabItem value="version_details">

**Version details** タブを使用して、各ジョブバージョンの入力や出力タイプ、各ジョブバージョンで使用されるファイルなど、特定の情報を閲覧します。
![](/images/launch/version_details_large.png)



  </TabItem>

</Tabs>