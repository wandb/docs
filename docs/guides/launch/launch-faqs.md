---
description: "W&B Launch\u306B\u95A2\u3059\u308B\u3088\u304F\u3042\u308B\u8CEA\u554F\
  \u3078\u306E\u56DE\u7B54\u3002"
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Launch FAQs

<head>
  <title>Launchに関するよくある質問</title>
</head>

## はじめに

### W&Bにコンテナを作成してほしくないのですが、Launchを使用できますか？

はい。以下のコマンドを実行して、事前にビルドされたDockerイメージを起動できます。`<>`の中身をあなたの情報に置き換えてください：

```bash
wandb launch -d <docker-image-uri> -q <queue-name> -E <entrypoint>
```

これにより、runを作成するとジョブがビルドされます。

または、イメージからジョブを作成することもできます：

```bash
wandb job create image <image-name> -p <project> -e <entity>
```

### Launchを効果的に使用するためのベストプラクティスはありますか？

1. エージェントを開始する前にキューを作成し、エージェントが簡単にポイントできるようにします。これを行わないと、エージェントはエラーを出し、キューを追加するまで動作しません。
2. エージェントを起動するためにW&Bサービスアカウントを作成し、個々のユーザーアカウントに紐付けないようにします。
3. `wandb.config`を使用してハイパーパラメーターを読み書きし、ジョブを再実行する際に上書きできるようにします。argsparseを使用している場合は[このガイド](https://docs.wandb.ai/guides/launch/create-launch-job#making-your-code-job-friendly)を参照してください。

### クリックが嫌いです。UIを通さずにLaunchを使用できますか？

はい。標準の`wandb` CLIには、ジョブを起動するための`launch`サブコマンドが含まれています。詳細については、以下を実行してください：

```bash
wandb launch --help
```

### Launchはターゲット環境で計算リソースを自動的にプロビジョニング（およびスピンダウン）できますか？

これは環境によります。SageMakerやVertexではリソースをプロビジョニングできます。Kubernetesでは、必要に応じてリソースを自動的にスピンアップおよびスピンダウンするためにオートスケーラーを使用できます。W&Bのソリューションアーキテクトは、リトライ、オートスケーリング、およびスポットインスタンスノードプールの使用を容易にするために、基盤となるKubernetesインフラストラクチャーの設定を支援します。support@wandb.comまたは共有Slackチャンネルに連絡してください。

### `wandb launch -d`や`wandb job create image`は、Dockerアーティファクト全体をアップロードしてレジストリからプルしていないのですか？

いいえ。`wandb launch -d`コマンドはレジストリにアップロードしません。イメージを自分でレジストリにアップロードする必要があります。一般的な手順は以下の通りです：

1. イメージをビルドします。
2. イメージをレジストリにプッシュします。

ワークフローは以下のようになります：

```bash
docker build -t <repo-url>:<tag> .
docker push <repo-url>:<tag>
wandb launch -d <repo-url>:<tag>
```

ここから、launchエージェントはそのコンテナを指すジョブをスピンアップします。コンテナレジストリからイメージをプルするためのエージェントのアクセス方法については、[Advanced agent setup](./setup-agent-advanced.md#agent-configuration)を参照してください。

Kubernetesの場合、Kubernetesクラスターのポッドはプッシュ先のレジストリにアクセスする必要があります。

### Dockerfileを指定して、W&BにDockerイメージをビルドしてもらえますか？

はい。これは、頻繁に変更されない多くの要件があるが、頻繁に変更されるコードベースがある場合に特に便利です。

:::important
Dockerfileがマウントを使用するようにフォーマットされていることを確認してください。詳細については、[Docker Docsのマウントに関するドキュメント](https://docs.docker.com/build/guide/mounts/)を参照してください。
:::

Dockerfileが設定されたら、W&BにDockerfileを指定する方法は3つあります：

* Dockerfile.wandbを使用する
* W&B CLIを使用する
* W&B Appを使用する

<Tabs
  defaultValue="dockerfile"
  values={[
    {label: 'Dockerfile.wandb', value: 'dockerfile'},
    {label: 'W&B CLI', value: 'cli'},
    {label: 'W&B App', value: 'app'},
  ]}>
  <TabItem value="dockerfile">

W&B runのエントリーポイントと同じディレクトリーに`Dockerfile.wandb`というファイルを含めます。W&BはW&Bの組み込みDockerfileの代わりに`Dockerfile.wandb`を使用します。

  </TabItem>
  <TabItem value="cli">

[`wandb launch`](../../ref/cli/wandb-launch.md)コマンドでlaunchジョブをキューに追加する際に`--dockerfile`フラグを提供します：

```bash
wandb launch --dockerfile path/to/Dockerfile
```

  </TabItem>
  <TabItem value="app">

W&B Appでジョブをキューに追加する際に、**Overrides**セクションにDockerfileのパスを提供します。具体的には、キーが`"dockerfile"`で値がDockerfileのパスとなるキー-バリューペアとして提供します。

例えば、以下のJSONはローカルディレクトリー内にあるDockerfileを含める方法を示しています：

```json title="Launch job W&B App"
{
  "args": [],
  "run_config": {
    "lr": 0,
    "batch_size": 0,
    "epochs": 0
  },
  "entrypoint": [],
  "dockerfile": "./Dockerfile"
}
```

  </TabItem>
</Tabs>

## 権限とリソース

### 誰がキューにプッシュできるかを制御するにはどうすればよいですか？

キューはユーザーチームにスコープされます。キューを作成する際に所有エンティティを定義します。アクセスを制限するには、チームメンバーシップを変更できます。

### Kubernetesでエージェントに必要な権限は何ですか？

次のKubernetesマニフェストは、`wandb`ネームスペースに`wandb-launch-agent`という名前のロールを作成します。このロールは、エージェントが`wandb`ネームスペースでポッド、コンフィグマップ、シークレット、およびポッド/ログを作成できるようにします。`wandb-cluster-role`は、エージェントが任意のネームスペースでポッド、ポッド/ログ、シークレット、ジョブ、およびジョブ/ステータスを作成できるようにします。

### Launchは並列化をサポートしていますか？ジョブが消費するリソースを制限するにはどうすればよいですか？

はい、Launchは複数のGPUおよび複数のノードにわたるジョブのスケーリングをサポートしています。詳細については[このガイド](https://docs.wandb.ai/tutorials/volcano)を参照してください。

ジョブ間レベルでは、個々のlaunchエージェントは同時に実行できるジョブの数を決定する`max_jobs`パラメーターで設定されます。さらに、特定のキューに対して任意の数のエージェントをポイントできますが、それらのエージェントが起動できるインフラストラクチャーに接続されている必要があります。

CPU/GPU、メモリ、およびその他の要件をlaunchキューまたはジョブrunレベルでリソース設定に制限できます。Kubernetesでリソース制限付きのキューを設定する方法については[こちら](https://docs.wandb.ai/guides/launch/kubernetes#queue-configuration)を参照してください。

スイープの場合、SDKでキュー設定にブロックを追加できます：

```yaml title="queue config"
  scheduler:
    num_workers: 4
```

これにより、スイープから並行して実行されるrunの数を制限できます。

### Dockerキューを使用して複数のジョブを実行し、`use_artifact`で同じArtifactをダウンロードする場合、ジョブの実行ごとにArtifactを再ダウンロードするのですか、それとも内部でキャッシュが行われているのですか？

キャッシュはありません。各ジョブは独立しています。ただし、キュー/エージェントを設定して共有キャッシュをマウントする方法があります。これをキュー設定のdocker argsを介して実現できます。

特別なケースとして、W&Bアーティファクトキャッシュを永続ボリュームとしてマウントすることもできます。

### ジョブ/オートメーションのためにシークレットを指定できますか？例えば、ユーザーに直接見せたくないAPIキーなど。

はい。推奨される方法は次の通りです：

1. runが作成されるネームスペースにバニラk8sシークレットとしてシークレットを追加します。例えば、`kubectl create secret -n <namespace> generic <secret_name> <secret value>`のようにします。

2. シークレットが作成されたら、runが開始されるときにシークレットを注入するためのキュー設定を指定できます。エンドユーザーはシークレットを確認できず、クラスター管理者のみが確認できます。

### MLエンジニアが変更できる内容を管理者が制限するにはどうすればよいですか？例えば、イメージタグの変更は問題ないが、他のジョブ設定は変更できないようにしたい場合。

これは[queue config templates](./setup-queue-advanced.md)で制御できます。これにより、特定のキューフィールドを非チーム管理者ユーザーが管理者ユーザーによって定義された制限内で編集できるようにします。キューの作成や編集、公開されるフィールドやその制限の定義はチーム管理者のみが行えます。

### W&B Launchはどのようにイメージをビルドしますか？

イメージをビルドする手順は、実行されるジョブのソースやリソース設定にアクセラレーターベースイメージが指定されているかどうかによって異なります。

:::note
キュー設定やジョブの提出時に、キューやジョブのリソース設定でベースアクセラレータイメージを指定できます：
```json
{
    "builder": {
        "accelerator": {
            "base_image": "image-name"
        }
    }
}
```
:::

ビルドプロセス中に、提供されたジョブの種類やアクセラレーターベースイメージに応じて以下のアクションが実行されます：

|                                                     | aptを使用してPythonをインストール | Pythonパッケージをインストール | ユーザーと作業ディレクトリを作成 | コードをイメージにコピー | エントリーポイントを設定 |
|-----------------------------------------------------|:------------------------:|:-----------------------:|:-------------------------:|:--------------------:|:--------------:|
| Gitからソースされたジョブ                           |                          |            X            |             X             |           X          |        X       |
| コードからソースされたジョブ                         |                          |            X            |             X             |           X          |        X       |
| Gitからソースされ、アクセラレータイメージが提供されたジョブ |             X            |            X            |             X             |           X          |        X       |
| コードからソースされ、アクセラレータイメージが提供されたジョブ|             X            |            X            |             X             |           X          |        X       |
| イメージからソースされたジョブ                       |                          |                         |                           |                      |                |

### アクセラレーターベースイメージにはどのような要件がありますか？

アクセラレーターを使用するジョブの場合、必要なアクセラレーターコンポーネントがインストールされたアクセラレーターベースイメージを提供できます。提供されたアクセラレータイメージの他の要件は次の通りです：

- Debian互換性（Launch Dockerfileはapt-getを使用してPythonを取得します）
- CPUおよびGPUハードウェア命令セットとの互換性（使用するGPUがサポートするCUDAバージョンを確認してください）
- 提供されたアクセラレーターのバージョンとMLアルゴリズムにインストールされたパッケージとの互換性
- ハードウェアとの互換性を設定するための追加手順が必要なパッケージ

### W&B LaunchをGPU上のTensorflowと連携させるにはどうすればよいですか？

GPU上のTensorflowを使用するジョブの場合、runがGPUを適切に利用できるように、エージェントが実行するコンテナビルドのためにカスタムベースイメージを指定する必要があるかもしれません。これを行うには、リソース設定の`builder.accelerator.base_image`キーの下にイメージタグを追加します。例えば：

```json
{
    "gpus": "all",
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```

wandbバージョン0.15.6以前では、`accelerator`の代わりに`cuda`を親キーとして`base_image`を使用してください。

### Launchがイメージをビルドする際にカスタムリポジトリを使用できますか？

はい。これを行うには、`requirements.txt`に以下の行を追加し、`index-url`および`extra-index-url`に渡す値を自分の値に置き換えます：

```text
----index-url=https://xyz@<your-repo-host> --extra-index-url=https://pypi.org/simple
```

`requirements.txt`はジョブのベースルートに定義する必要があります。

## プリエンプション時の自動run再キューイング

場合によっては、中断された後にジョブを再開するように設定することが有用です。例えば、スポットインスタンスで広範なハイパーパラメータースイープを実行し、より多くのスポットインスタンスがスピンアップしたときに再開したい場合などです。LaunchはKubernetesクラスターでこの設定をサポートできます。

Kubernetesキューがスケジューラーによってプリエンプトされたノードでジョブを実行している場合、そのジョブは自動的にキューの最後に追加され、後で再開できるようになります。この再開されたrunは元のrunと同じ名前を持ち、UIの同じページからフォローできます。この方法でジョブは最大5回まで自動的に再キューイングされます。

Launchは、ポッドがスケジューラーによってプリエンプトされたかどうかを、ポッドが以下の理由のいずれかを持つ`DisruptionTarget`条件を持っているかどうかを確認することで検出します：

- `EvictionByEvictionAPI`
- `PreemptionByScheduler`
- `TerminationByKubelet`

ジョブのコードが再開を許可するように構造化されている場合、これにより再キューイングされたrunは中断した場所から再開できます。そうでない場合、runは再キューイングされると最初から開始されます。詳細については、[runの再開](../runs/resuming.md)に関するガイドを参照してください。

現在、プリエンプトされたノードの自動run再キューイングをオプトアウトする方法はありません。ただし、UIからrunを削除するか、ノードを直接削除すると、再キューイングされません。

自動run再キューイングは現在Kubernetesキューでのみ利用可能です。SagemakerおよびVertexはまだサポートされていません。