---
description: Discover how to automate hyperparamter sweeps on launch.
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 起動時のスイープ

W&Bをローカル環境や選択したクラウドサービスで直接実行し、実験を行って結果を比較するため、W&B ユーザーインターフェースから run やスイープを直接再現できます。

## スイープの作成
Launchを使ってW&Bスイープを作成します。W&BアプリやW&B CLIで対話式にスイープを作成できます。

:::info
W&B Launchでスイープを作成する前に、まずジョブを作成してください。スイープ作成元のrunにコードアーティファクトがあることを確認してください。詳しくは[ジョブの作成](./create-job.md)ページをご覧ください。
:::

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&Bアプリ', value: 'app'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="app">
  W&B アプリで対話式にスイープを作成します。

1. W&Bアプリでプロジェクトページへ移動します。  
2. 左パネルのスイープアイコン（ほうきの画像）を選択します。
3. 次に、**スイープの作成**ボタンを選択します。
4. **Launchを使用 🚀（ベータ版）**スイッチをオンにします。
5. **ジョブ**ドロップダウンメニューから、スイープ作成元のジョブ名とバージョンを選択します。
6. **キュー**ドロップダウンメニューから、ジョブを追加するキューを選択します。
7. **スイープの初期化**を選択します。

![](/images/launch/create_sweep_with_launch.png)

  </TabItem>
  <TabItem value="cli">
W&B CLIを使って、プログラムでW&Bスイープを作成し、Launchを実行します。

1. スイープ構成を作成する
2. スイープ構成内で完全なジョブ名を指定する
3. スイープエージェントを初期化する

:::info
ステップ1と3は、通常のW&Bスイープを作成する際に行う手順と同じです。ただし、スイープのYAML構成ファイル内でジョブの名前を指定する必要があります。
:::

例えば、以下のコードスニペットでは、ジョブ値に`wandb/launch_demo/job-source-launch_demo-canonical_job_example.py:v0`を指定しています。

```yaml
#config.yaml

job: wandb/launch_demo/job-source-launch_demo-canonical_job_example.py:v0
description: sweep examples using launch jobs

method: bayes
metric:
  goal: minimize
  name: ""
parameters:
  learning_rate:
    max: 0.02
    min: 0
    distribution: uniform
  epochs:
    max: 20
    min: 0
    distribution: int_uniform
```
スイープ構成の作成方法については、[スイープ構成の定義](../sweeps/define-sweep-configuration.md)ページをご覧ください。



4. 次に、スイープを初期化します。設定ファイルへのパス、ジョブキューの名前、W＆Bエンティティ、およびキュー、エンティティ、プロジェクトフラグのプロジェクト名をそれぞれ提供してください。



```bash

wandb sweep <path/to/yaml/file> --queue <queue_name> --entity <your_entity>  --project <project_name>

```



W＆Bスイープの詳細については、[ハイパーパラメータチューニング](../sweeps/intro.md)の章を参照してください。





  </TabItem>

</Tabs>