import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Spin up a single node GPU cluster with Minikube

Minikubeクラスター上でW&B Launchを使用してGPUワークロードをスケジュールおよび実行する設定方法

:::info
このチュートリアルは、複数のGPUを備えたマシンに直接アクセスできるユーザーを対象としています。このチュートリアルは、クラウドマシンをレンタルしているユーザー向けではありません。

クラウドマシン上でMinikubeクラスターを設定したい場合は、クラウドプロバイダーを利用したGPUサポート付きのKubernetesクラスターの作成をW&Bは推奨します。たとえば、AWS、GCP、Azure、Coreweaveなどのクラウドプロバイダーには、GPUサポート付きのKubernetesクラスターを作成するためのツールがあります。

単一GPUを持つマシンでGPUをスケジュールするためにMinikubeクラスターを設定したい場合は、[Launch Docker queue](/guides/launch/setup-launch-docker)の使用をW&Bは推奨します。楽しみのためにこのチュートリアルを進めることもできますが、GPUのスケジュールはあまり役に立ちません。
:::

## 背景

[Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)の登場により、Docker上でGPU対応のワークフローを簡単に実行できるようになりました。しかし、ボリュームによるGPUのスケジュールをネイティブにサポートしていないという制約があります。`docker run`コマンドを使用してGPUを利用する場合、特定のIDのGPUか、すべてのGPUをリクエストする必要があり、多くの分散GPU対応ワークロードには非現実的です。Kubernetesはボリュームリクエストによるスケジュールをサポートしていますが、GPUスケジュール付きのローカルKubernetesクラスターを設定するにはかなりの時間と手間がかかります。ただし、最近MinikubeはGPUスケジュールをサポートする機能をリリースし、このチュートリアルでは、複数のGPUを搭載したマシンにMinikubeクラスターを作成し、W&B Launchを使用してクラスターに安定拡散推論ジョブを同時に実行します 🎉

## 前提条件

開始する前に、以下が必要です：

1. W&Bアカウント。
2. 次のものをインストールして実行しているLinuxマシン：
   1. Dockerランタイム
   2. 使用するGPUのドライバ
   3. Nvidia container toolkit

:::note
このチュートリアルのテストと作成には、Google Cloud Compute Engineの`n1-standard-16`インスタンスに4つのNVIDIA Tesla T4 GPUを接続して使用しました。
:::

## Launchジョブのキューを作成

まず、Launchジョブのキューを作成します。

1. [wandb.ai/launch](https://wandb.ai/launch)（またはプライベートW&Bサーバーを使用している場合は`<your-wandb-url>/launch`）に移動します。
2. 画面の右上隅にある青い**Create a queue**ボタンをクリックします。右側からキュー作成のドロワーがスライドして表示されます。
3. エンティティを選択し、名前を入力し、キューのタイプとして**Kubernetes**を選択します。
4. ドロワーの**Config**セクションには、Launchキューの[Kubernetesジョブ仕様](https://kubernetes.io/docs/concepts/workloads/controllers/job/)を入力します。このキューから起動されるすべてのrunsは、このジョブ仕様を使用して作成されるため、必要に応じてこの設定を変更してジョブをカスタマイズできます。このチュートリアルでは、以下のサンプル設定をYAMLまたはJSONでコピーしてキュー設定に貼り付けることができます：

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

キュー設定に関する詳細は、[Set up Launch on Kubernetes](/guides/launch/setup-launch-kubernetes.md)および[Advanced queue setup guide](/guides/launch/setup-queue-advanced.md)をご覧ください。

`${image_uri}`および`{{gpus}}`文字列は、キュー設定に使用できる2種類の変数テンプレートの例です。`${image_uri}`テンプレートは、起動するジョブのイメージURIに置き換えられ、`{{gpus}}`テンプレートは、ジョブを送信する際にLaunch UI、CLI、SDKからオーバーライドできるテンプレート変数を作成するために使用されます。これらの値は、ジョブのイメージおよびGPUリソースを制御するために正しいフィールドを変更するようにジョブ仕様に配置されます。

5. **Parse configuration**ボタンをクリックして`gpus`テンプレート変数のカスタマイズを開始します。
6. **Type**を`Integer`に設定し、**Default**、**Min**、および**Max**を選択した値に設定します。
テンプレート変数の制約に違反するキューへのrun送信は拒否されます。

![gpusテンプレート変数を含むキュー作成ドロワーの画像](/images/tutorials/minikube_gpu/create_queue.png)

7. **Create queue**をクリックしてキューを作成します。新しいキューのキューページにリダイレクトされます。

次のセクションでは、作成したキューからジョブを取得して実行できるエージェントを設定します。

## Docker + NVIDIA CTKの設定

すでにマシンにDockerとNvidia container toolkitがセットアップされている場合、このセクションはスキップできます。

システムにDockerコンテナエンジンをセットアップする手順については、[Dockerのドキュメント](https://docs.docker.com/engine/install/)を参照してください。

Dockerをインストールしたら、Nvidiaのコンテナツールキットを[Nvidiaのドキュメント](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)に従ってインストールします。

コンテナランタイムがGPUにアクセスできるかどうかを確認するには、次のコマンドを実行します：

```bash
docker run --gpus all ubuntu nvidia-smi
```

マシンに接続されたGPUを説明する`nvidia-smi`の出力が表示されるはずです。たとえば、私たちのセットアップでは出力は次のようになっていました：

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

## Minikubeの設定

MinikubeのGPUサポートにはバージョン`v1.32.0`以降が必要です。最新のインストール手順については、[Minikubeのインストールドキュメント](https://minikube.sigs.k8s.io/docs/start/)を参照してください。このチュートリアルでは、最新のMinikubeリリースを次のコマンドを使用してインストールしました：

```yaml
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

次のステップは、GPUを使用してminikubeクラスターを開始することです。マシン上で次のコマンドを実行します：

```yaml
minikube start --gpus all
```

上記のコマンドの出力は、クラスターが正常に作成されたかどうかを示します。

## Launchエージェントを開始

新しいクラスターのLaunchエージェントは、`wandb launch-agent`を直接実行するか、W&Bが管理する[helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)を使用してデプロイすることで開始できます。

このチュートリアルでは、ホストマシン上でエージェントを直接実行します。

:::tip
コンテナ外でエージェントを実行することで、ローカルDockerホストを使用してクラスターのためにイメージをビルドすることもできます。
:::

エージェントをローカルで実行するには、デフォルトのKubernetes APIコンテキストがMinikubeクラスターを指していることを確認します。その後、次のコマンドを実行します：

```bash
pip install wandb[launch]
```

エージェントの依存関係をインストールするために。エージェントの認証を設定するには、`wandb login`を実行するか、`WANDB_API_KEY`環境変数を設定します。

エージェントを開始するには、次のコマンドを入力して実行します：

```bash
wandb launch-agent -j <max-number-concurrent-jobs> -q <queue-name> -e <queue-entity>
```

端末にLaunchエージェントのポーリングメッセージが表示され始めたら成功です。

おめでとうございます、LaunchキューをポーリングするLaunchエージェントが設定されました！キューにジョブが追加されると、エージェントがそれを取得し、Minikubeクラスターで実行するようにスケジュールします。

## ジョブを起動

エージェントにジョブを送信しましょう。W&Bアカウントにログインした端末から簡単な「Hello World」ジョブを起動するには、次のコマンドを実行します：

```yaml
wandb launch -d wandb/job_hello_world:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

任意のジョブやイメージでテストできますが、クラスターがイメージをプルできることを確認してください。追加の指針については[Minikubeのドキュメント](https://minikube.sigs.k8s.io/docs/handbook/registry/)を参照してください。[公開ジョブの1つを使用してテストすることもできます](https://wandb.ai/wandb/jobs/jobs?workspace=user-bcanfieldsherman)。

## （オプション）NFSを使用したモデルとデータのキャッシュ

MLワークロードでは、複数のジョブが同じデータにアクセスすることを望むことがよくあります。たとえば、モデルの重みやデータセットなどの大きなアセットを繰り返しダウンロードするのを避けるために共有キャッシュが必要かもしれません。Kubernetesは[Persistent VolumesおよびPersistent Volume Claims](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)を通じてこれをサポートします。Persistent Volumesを使用すると、Kubernetesワークロードで直接ファイルシステムアクセスを提供する`volumeMounts`を作成できます。

このステップでは、モデルの重みを共有キャッシュとして使用できるネットワークファイルシステム（NFS）サーバーをセットアップします。最初のステップは、NFSをインストールして構成することです。このプロセスはオペレーティングシステムによって異なります。私たちのVMがUbuntuを実行しているので、nfs-kernel-serverをインストールし、`/srv/nfs/kubedata`にエクスポートを構成しました：

```bash
sudo apt-get install nfs-kernel-server
sudo mkdir -p /srv/nfs/kubedata
sudo chown nobody:nogroup /srv/nfs/kubedata
sudo sh -c 'echo "/srv/nfs/kubedata *(rw,sync,no_subtree_check,no_root_squash,no_all_squash,insecure)" >> /etc/exports'
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

ホストファイルシステムのエクスポート場所、およびNFSサーバーのローカルIPアドレス（つまりホストマシン）のメモを取ってください。次のステップでこれらの情報が必要になります。

次に、このNFSのためのPersistent VolumeとPersistent Volume Claimを作成する必要があります。Persistent Volumeは高度にカスタマイズ可能ですが、このチュートリアルではシンプルな設定を使用します。

以下のyamlコードを`nfs-persistent-volume.yaml`という名前のファイルにコピーし、希望するボリューム容量と要求量を記入します。`PersistentVolume.spec.capacity.storage`フィールドは基礎となるボリュームの最大サイズを制御します。`PersistentVolumeClaim.spec.resources.requests.storage`は特定の要求に対して割り当てられたボリューム容量を制限するために使用できます。このユースケースでは、各フィールドに同じ値を使用するのが理にかなっています。

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nfs-pv
spec:
  capacity:
    storage: 100Gi # 希望する容量を設定します。
  accessModes:
    - ReadWriteMany
  nfs:
    server: <your-nfs-server-ip> # TODO: ここにNFSサーバーのIPを記入します。
    path: '/srv/nfs/kubedata' # またはカスタムパス
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
      storage: 100Gi # 希望する容量を設定します。
  storageClassName: ''
  volumeName: nfs-pv
```

次のコマンドでクラスタにリソースを作成します：

```yaml
kubectl apply -f nfs-persistent-volume.yaml
```

runsでこのキャッシュを使用するためには、Launchキュー設定に`volumes`および`volumeMounts`を追加する必要があります。Launchの設定を編集するには、[wandb.ai/launch](http://wandb.ai/launch)（または、W&Bサーバーを使用している