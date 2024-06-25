import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Spin up a single node GPU cluster with Minikube

Minikube クラスターで W&B Launch をセットアップして、GPU ワークロードをスケジュールおよび実行します。

:::info
このチュートリアルは、複数の GPU を持つマシンに直接アクセスできるユーザーを対象としています。このチュートリアルは、クラウドマシンをレンタルするユーザーを対象としていません。

クラウドマシンで Minikube クラスターを設定したい場合は、クラウドプロバイダー（例えば AWS、GCP、Azure、Coreweave など）が提供するツールを使って GPU サポート付きの Kubernetes クラスターを作成することを W&B は推奨します。

シングル GPU マシンで GPUs をスケジュールするために Minikube クラスターを設定したい場合は、[Launch Docker queue](/guides/launch/setup-launch-docker) を使用することを W&B は推奨します。楽しみのためにこのチュートリアルを続行することはできますが、GPU スケジューリングはあまり役に立ちません。
:::

## 背景

[Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) により、Docker で GPU 対応のワークフローを実行するのが簡単になりました。ただし、ボリュームによる GPU スケジューリングのネイティブサポートがないことは制約です。`docker run` コマンドで GPU を使用するには、ID で特定の GPU をリクエストするか、またはすべての存在する GPU をリクエストする必要があり、多くの分散 GPU 対応のワークロードには無理があります。Kubernetes はボリュームリクエストに基づくスケジューリングをサポートしていますが、GPU スケジューリングを備えたローカル Kubernetes クラスターのセットアップにはかなりの時間と労力がかかることがあります。Minikube は、シングルノードの Kubernetes クラスターを実行するための最も人気のあるツールの一つであり、最近 [GPU スケジューリングをサポートする機能](https://minikube.sigs.k8s.io/docs/tutorials/nvidia/) をリリースしました 🎉 このチュートリアルでは、複数 GPU マシンに Minikube クラスターを作成し、W&B Launch を使用してクラスターに同時に安定拡散推論ジョブを起動します 🚀

## 前提条件

始めに、次のものが必要です:

1. W&B のアカウント。
2. 次のものがインストールされ実行されている Linux マシン：
   1. Docker ランタイム
   2. 使用する任意の GPU のドライバ
   3. Nvidia コンテナツールキット

:::note
このチュートリアルをテストおよび作成するために、4 つの NVIDIA Tesla T4 GPU が接続された `n1-standard-16` Google Cloud Compute Engine インスタンスを使用しました。
:::

## Launch ジョブのためのキューを作成する

まず、Launch ジョブのためのキューを作成します。

1. [wandb.ai/launch](https://wandb.ai/launch)（またはプライベート W&B サーバーを使用している場合は `<your-wandb-url>/launch`）に移動します。
2. 画面の右上隅にある青い **Create a queue** ボタンをクリックします。右側からキュー作成のドロワーがスライドして表示されます。
3. エンティティを選択し、名前を入力し、キューのタイプとして **Kubernetes** を選択します。
4. ドロワーの **Config** セクションで、Launch キューの [Kubernetes ジョブ仕様](https://kubernetes.io/docs/concepts/workloads/controllers/job/) を入力します。このキューから起動されるすべての runs はこのジョブ仕様を使用して作成されますので、ジョブをカスタマイズするためにこの設定を必要に応じて変更できます。このチュートリアルでは、以下のサンプル設定を YAML または JSON 形式でキュー設定にコピーして貼り付けます：

<Tabs
defaultValue="yaml"
values={[
{ label: "YAML", value: "yaml", },
{ label: "JSON", value: "json", },
]}>

<TabItem value="yaml">

```yaml
spec:
  template:
    spec:
      containers:
        - image: ${image_uri}
          resources:
            limits:
              cpu: 4
              memory: 12Gi
              nvidia.com/gpu: '{{gpus}}'
      restartPolicy: Never
  backoffLimit: 0
```

</TabItem>

<TabItem value="json">

```json
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "image": "${image_uri}",
            "resources": {
              "limits": {
                "cpu": 4,
                "memory": "12Gi",
                "nvidia.com/gpu": "{{gpus}}"
              }
            }
          }
        ],
        "restartPolicy": "Never"
      }
    },
    "backoffLimit": 0
  }
}
```

</TabItem>
</Tabs>

キュー設定に関する詳細については、[Set up Launch on Kubernetes](/guides/launch/setup-launch-kubernetes.md) および [Advanced queue setup guide](/guides/launch/setup-queue-advanced.md) を参照してください。

`${image_uri}` および `{{gpus}}` 文字列は、キュー設定で使用できる二種類の変数テンプレートの例です。`${image_uri}` テンプレートは、起動されたジョブの画像 URI に置き換えられます。`{{gpus}}` テンプレートは、ジョブを送信する際に Launch UI、CLI、または SDK から上書きできるテンプレート変数を作成するために使用されます。これらの値はジョブ仕様に配置され、ジョブで使用される画像および GPU リソースを制御する正しいフィールドを変更します。

5. **Parse configuration** ボタンをクリックして、`gpus` テンプレート変数のカスタマイズを開始します。
6. **Type** を `Integer` に設定し、**Default**、**Min** および **Max** を任意の値に設定します。テンプレート変数の制約に違反するキューへの run 送信の試みは拒否されます。

![Image of queue creation drawer with gpus template variable](/images/tutorials/minikube_gpu/create_queue.png)

7. **Create queue** をクリックしてキューを作成します。新しいキューのキューページにリダイレクトされます。

次のセクションでは、作成したキューからジョブを取得して実行できるエージェントを設定します。

## Docker + NVIDIA CTK の設定

すでにマシンに Docker および Nvidia コンテナツールキットの設定がある場合、このセクションをスキップできます。

システムに Docker コンテナエンジンを設定する手順については、[Docker のドキュメント](https://docs.docker.com/engine/install/) を参照してください。

Docker がインストールされたら、[Nvidia のドキュメントに従って](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) Nvidia コンテナツールキットをインストールします。

コンテナランタイムが GPU へのアクセス権を持っていることを確認するために、次のコマンドを実行します：

```bash
docker run --gpus all ubuntu nvidia-smi
```

マシンに接続された GPU を示す `nvidia-smi` の出力が表示されるはずです。たとえば、我々の設定では出力は次のようになります：

```
Wed Nov  8 23:25:53 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   38C    P8     9W /  70W |      2MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla T4            Off  | 00000000:00:05.0 Off |                    0 |
| N/A   38C    P8     9W /  70W |      2MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  Tesla T4            Off  | 00000000:00:06.0 Off |                    0 |
| N/A   40C    P8     9W /  70W |      2MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  Tesla T4            Off  | 00000000:00:07.0 Off |                    0 |
| N/A   39C    P8     9W /  70W |      2MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## Minikube の設定

Minikube の GPU サポートには `v1.32.0` 以降のバージョンが必要です。最新のインストール支援については、[Minikube のインストールドキュメント](https://minikube.sigs.k8s.io/docs/start/) を参照してください。このチュートリアルでは、以下のコマンドを使用して最新の Minikube リリースをインストールします：

```yaml
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

次のステップは、GPU を使用して minikube クラスターを開始することです。マシンで次のコマンドを実行します：

```yaml
minikube start --gpus all
```

上記のコマンドの出力は、クラスターが正常に作成されたかどうかを示します。

## Launch エージェントの起動

新しいクラスターの Launch エージェントは、`wandb launch-agent` を直接実行するか、[W&B が管理するヘルムチャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使用して Launch エージェントをデプロイすることで起動できます。

このチュートリアルでは、ホストマシン上で直接エージェントを実行します。

:::tip
コンテナの外でエージェントを実行することは、ローカル Docker ホストを使用してクラスター用のイメージを構築できることも意味します。
:::

ローカルでエージェントを実行するには、デフォルトの Kubernetes API コンテキストが Minikube クラスターを参照することを確認します。次に、以下を実行してエージェントの依存関係をインストールします：

```bash
pip install wandb[launch]
```

エージェントの認証を設定するには、`wandb login` を実行するか、`WANDB_API_KEY` 環境変数を設定します。

エージェントを起動するには、次をタイプして実行します：

```bash
wandb launch-agent -j <max-number-concurrent-jobs> -q <queue-name> -e <queue-entity>
```

ターミナル内でエージェントがポーリングメッセージを出力し始めるのを見ることができます。

おめでとうございます、新しい Launch キューをポーリングしている Launch エージェントが起動しました！ キューにジョブが追加されると、エージェントはそれを受け取り、Minikube クラスターで実行するようにスケジュールします。

## ジョブを起動する

エージェントにジョブを送信しましょう。W&B アカウントにログインしたターミナルからシンプルな「Hello World」ジョブを起動できます：

```yaml
wandb launch -d wandb/job_hello_world:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

任意のジョブやイメージでテストできますが、クラスターがイメージをプルできるようにしてください。追加のガイダンスについては[Minikube のドキュメント](https://minikube.sigs.k8s.io/docs/handbook/registry/) を参照してください。[我々の公開ジョブの一つを使ってテストすることもできます](https://wandb.ai/wandb/jobs/jobs?workspace=user-bcanfieldsherman)。

## (Optional) モデルとデータのキャッシング (NFS)

ML のワークロードでは、複数のジョブが同じデータにアクセスできることがよく求められます。例えば、大きなデータセットやモデルの重みを繰り返しダウンロードするのを避けるために、共有キャッシュを持ちたいことがあります。Kubernetes はこれを [永続ボリュームおよび永続ボリューム要求](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) を通じてサポートしています。永続ボリュームを使用して Kubernetes ワークロードに `volumeMounts` を作成し、共有キャッシュへの直接的なファイルシステムアクセスを提供できます。

このステップでは、モデルの重みの共有キャッシュとして使用できるネットワークファイルシステム (NFS) サーバーをセットアップします。最初のステップは NFS をインストールし、設定することです。このプロセスはオペレーティングシステムによって異なります。我々の VM は Ubuntu を実行しているため、nfs-kernel-server をインストールし、`/srv/nfs/kubedata` にエクスポートを設定しました：

```bash
sudo apt-get install nfs-kernel-server
sudo mkdir -p /srv/nfs/kubedata
sudo chown nobody:nogroup /srv/nfs/kubedata
sudo sh -c 'echo "/srv/nfs/kubedata *(rw,sync,no_subtree_check,no_root_squash,no_all_squash,insecure)" >> /etc/exports'
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

サーバーのエクスポート位置と NFS サーバーのローカル IP アドレス（ホストマシン）をメモしておいてください。次のステップでこの情報が必要になります。

次に、この NFS のための永続ボリュームと永続ボリューム要求を作成する必要があります。永続ボリュームは高度にカスタマイズ可能ですが、簡単のためにここではシンプルな設定を使用します。

以下の yaml を `nfs-persistent-volume.yaml` という名前のファイルにコピーし、希望するボリューム容量と要求を入力してください。`PersistentVolume.spec.capcity.storage` フィールドは基礎となるボリュームの最大サイズを制御します。`PersistentVolumeClaim.spec.resources.requests.stroage` は特定の要求に割り当てられるボリューム容量を制限するために使用できます。ここでは、各フィールドに同じ値を使用するのが理にかなっています。

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nfs-pv
spec:
  capacity:
    storage: 100Gi # あなたの希望する容量に設定します。
  accessModes:
    - ReadWriteMany
  nfs:
    server: <your-nfs-server-ip> # TODO: ここに入力してください。
    path: '/srv/nfs/kubedata' # またはあなたのカスタムパス
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nfs-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi # あなたの希望する容量に設定します。
  storageClassName: ''
  volumeName: nfs-pv
```

次のコマンドでクラスター内にリソースを作成します：

```yaml
kubectl apply -f nfs-persistent-volume.yaml