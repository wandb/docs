import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Minikubeで単一ノードのGPUクラスターを立ち上げる

Minikubeクラスター上でGPUワークロードをスケジュールし実行するためのW&B Launchのセットアップ方法をご紹介します。

:::info
このチュートリアルは、複数のGPUsを搭載したマシンに直接アクセスできるユーザー向けです。クラウドマシンをレンタルしているユーザー向けではありません。

クラウドマシン上でminikubeクラスターをセットアップしたい場合は、クラウドプロバイダーを利用してGPUサポートつきのKubernetesクラスターを作成することをW&Bでは推奨しています。例えば、AWS、GCP、Azure、Coreweaveなどのクラウドプロバイダーには、GPUサポートつきのKubernetesクラスターを作成するツールがあります。

単一のGPUを持つマシンでGPUのスケジューリングを行うためのminikubeクラスターをセットアップしたい場合は、[Launch Docker queue](/guides/launch/setup-launch-docker)の使用をW&Bでは推奨しています。このチュートリアルを楽しむために従うことはできますが、GPUのスケジューリングはあまり役立ちません。
:::

## 背景

[Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)のおかげで、Docker上でGPU対応のワークフローを実行することが簡単になりました。しかし、ボリュームによるGPUのスケジューリングのネイティブサポートが欠けているという制約があります。`docker run`コマンドでGPUを使用する場合、特定のGPUをIDで指定するか、すべてのGPUを一度に要求する必要があります。これにより、多くの分散GPU対応ワークロードが実現困難になります。Kubernetesはボリューム要求によるスケジューリングをサポートしていますが、GPUスケジューリングを伴うローカルKubernetesクラスターのセットアップには多大な時間と労力が必要です。しかし最近になり、Minikubeはこのプロセスを簡素化しました。Minikubeは、単一ノードのKubernetesクラスターを実行するための最も人気のあるツールの1つであり、最近GPUのスケジューリングのサポートをリリースしました 🎉 このチュートリアルでは、複数GPUマシンにMinikubeクラスターを作成し、W&B Launchを使ってクラスターに安定したディフュージョン推論ジョブを並行して起動します 🚀

## 前提条件

開始する前に、以下が必要です：

1. W&Bアカウント。
2. 次のものがインストールされ実行されているLinuxマシン:
   1. Dockerランタイム
   2. 使用したいGPUのドライバ
   3. Nvidia container toolkit

:::note
このチュートリアルのテストと作成には、4つのNVIDIA Tesla T4 GPUが接続された`n1-standard-16` Google Cloud Compute Engineインスタンスを使用しました。
:::

## ジョブキューの作成

まず、Launchジョブ用のキューを作成します。

1. [wandb.ai/launch](https://wandb.ai/launch)（またはプライベートW&Bサーバーを使用している場合は`<your-wandb-url>/launch`）に移動します。
2. 画面右上の青い**Create a queue**ボタンをクリックします。右側からキュー作成のドロワーがスライド表示されます。
3. entityを選択し、名前を入力し、キューの種類を**Kubernetes**として選択します。
4. ドロワーの**Config**セクションに、Launchキュー用の[Kubernetesジョブ仕様](https://kubernetes.io/docs/concepts/workloads/controllers/job/)を入力します。このキューから起動されたすべてのrunはこのジョブ仕様を使用して作成されるため、必要に応じてジョブをカスタマイズするためにこの設定を変更できます。このチュートリアルのために、以下のサンプル設定をYAMLまたはJSONでキュー設定にコピー＆ペーストします：

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

キュー設定の詳細については、[Set up Launch on Kubernetes](/guides/launch/setup-launch-kubernetes.md) および [Advanced queue setup guide](/guides/launch/setup-queue-advanced.md) を参照してください。

`${image_uri}` および `{{gpus}}` 文字列は、キュー設定で使用できる2種類の可変テンプレートの例です。`${image_uri}` テンプレートはエージェントによって起動されるジョブのイメージURIに置き換えられ、`{{gpus}}` テンプレートはジョブを提出する際にLaunch UI、CLI、またはSDKから上書きできるテンプレート変数を作成するために使用されます。これらの値はジョブ仕様に配置され、ジョブに使用される画像およびGPUリソースを制御するフィールドを適切に変更します。

5. **Parse configuration** ボタンをクリックして `gpus` テンプレート変数のカスタマイズを開始します。
6. **Type** を `Integer` に設定し、**Default**、**Min** および **Max** を任意の値に設定します。テンプレート変数の制約に違反するrunの提出は拒否されます。

![Image of queue creation drawer with gpus template variable](/images/tutorials/minikube_gpu/create_queue.png)

7. **Create queue** をクリックすると、キューが作成され、その新しいキューのページにリダイレクトされます。

次のセクションでは、作成したキューからジョブを取得し実行できるエージェントをセットアップします。

## Docker + NVIDIA CTKのセットアップ

既にマシンにDockerおよびNVIDIA container toolkitが設定されている場合は、このセクションをスキップできます。

Dockerコンテナエンジンの設定については、[Dockerのドキュメント](https://docs.docker.com/engine/install/) を参照してください。

Dockerがインストールされたら、[Nvidiaのドキュメント](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) の指示に従ってNvidia container toolkitをインストールします。

コンテナランタイムがGPUにアクセスできるかを確認するには、次のコマンドを実行します：

```bash
docker run --gpus all ubuntu nvidia-smi
```

`nvidia-smi`により、マシンに接続されたGPUの情報が表示されるはずです。例えば、私たちの設定では、以下のような出力が得られます：

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

## Minikubeのセットアップ

MinikubeのGPUサポートはバージョン`v1.32.0`以降が必要です。最新のインストールヘルプについては、[Minikubeのインストールドキュメント](https://minikube.sigs.k8s.io/docs/start/)を参照してください。このチュートリアルでは、以下のコマンドを使用して最新のMinikubeリリースをインストールします：

```yaml
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

次のステップは、GPUを使用してminikubeクラスターを開始することです。マシンで次を実行します：

```yaml
minikube start --gpus all
```

上記のコマンドの出力により、クラスターが正常に作成されたかどうかが示されます。

## Launchエージェントの起動

新しいクラスターのLaunchエージェントは、`wandb launch-agent` を直接呼び出すか、[W&Bが管理するhelm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)を使用してデプロイすることで開始できます。

このチュートリアルでは、ホストマシンでエージェントを直接実行します。

:::tip
コンテナ外でエージェントを実行することは、ローカルDockerホストを使用してクラスターで実行するためのイメージをビルドできることも意味します。
:::

ローカルでエージェントを実行するには、デフォルトのKubernetes APIコンテキストがMinikubeクラスターを参照していることを確認します。次に、以下を実行してエージェントの依存関係をインストールします：

```bash
pip install wandb[launch]
```

エージェントの認証を設定するには、`wandb login` を実行するか、`WANDB_API_KEY` 環境変数を設定します。

エージェントを起動するには、以下を入力して実行します：

```bash
wandb launch-agent -j <max-number-concurrent-jobs> -q <queue-name> -e <queue-entity>
```

端末内で、エージェントがポーリングメッセージを出力し始めるのが見えるはずです。

おめでとうございます、LaunchキューをポーリングしているLaunchエージェントが起動しました！ キューにジョブが追加されると、エージェントがそれを取得し、Minikubeクラスターで実行するようにスケジュールします。

## ジョブの起動

エージェントにジョブを送信しましょう。W&Bアカウントにログインした端末から簡単な「hello world」を起動できます：

```yaml
wandb launch -d wandb/job_hello_world:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

任意のジョブやイメージでテストできますが、クラスターがイメージをプルできることを確認してください。追加のガイダンスについては、[Minikubeのドキュメント](https://minikube.sigs.k8s.io/docs/handbook/registry/)を参照してください。また、[私たちの公開ジョブの1つを使用してテストする](https://wandb.ai/wandb/jobs/jobs?workspace=user-bcanfieldsherman)こともできます。

## （オプション）モデルとデータのキャッシュ用NFSの設定

MLワークロードでは、複数のジョブが同じデータにアクセスできることが望まれることがあります。たとえば、大きなアセット（データセットやモデルの重みなど）を何度もダウンロードするのを避けるために、共有キャッシュを持つことが有用です。Kubernetesは[永続ボリュームと永続ボリューム要求](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)を通じてこれをサポートします。永続ボリュームはKubernetesワークロードに`volumeMounts`を作成し、共有キャッシュへの直接ファイルシステムアクセスを提供します。

このステップでは、モデルの重みを共有キャッシュとして使用できるネットワークファイルシステム（NFS）サーバーを設定します。まず、NFSをインストールおよび設定します。このプロセスはオペレーティングシステムによって異なります。私たちのVMではUbuntuが実行されているので、nfs-kernel-serverをインストールし、`/srv/nfs/kubedata`にエクスポートを設定しました：

```bash
sudo apt-get install nfs-kernel-server
sudo mkdir -p /srv/nfs/kubedata
sudo chown nobody:nogroup /srv/nfs/kubedata
sudo sh -c 'echo "/srv/nfs/kubedata *(rw,sync,no_subtree_check,no_root_squash,no_all_squash,insecure)" >> /etc/exports'
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

ホストファイルシステム上のサーバーのエクスポート場所と、NFSサーバーのローカルIPアドレス（つまりホストマシン）をメモしておいてください。次のステップでこれらの情報が必要になります。

次に、このNFSの永続ボリュームと永続ボリューム要求を作成する必要があります。永続ボリュームは高度にカスタマイズ可能ですが、このチュートリアルでは簡単な設定を使用します。

以下のyamlをファイル`nfs-persistent-volume.yaml`にコピーし、希望するボリューム容量および要求を入力して保存します。`PersistentVolume.spec.capcity.storage`フィールドは基礎となるボリュームの最大サイズを制御します。`PersistentVolumeClaim.spec.resources.requests.storage`を使用して、特定の要求に割り当てられるボリューム容量を制限できます。今回のユースケースでは、両方に同じ値を使用するのが理にかなっています。

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nfs-pv
spec:
  capacity:
    storage: 100Gi # 希望の容量を設定してください。
  accessModes:
    - ReadWriteMany
  nfs:
    server: <your-nfs-server-ip> # ここにNFSサーバーのIPを入力。
    path: '/srv/nfs/kubedata' # もしくはカスタムパス
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
      storage: 100Gi # 希望の容量を設定してください。
  storageClassName: ''
  volumeName: nfs-pv
```

クラスターで次のコマンドを実行してリソースを作成します：

```yaml
kubectl apply -f nfs-persistent-volume.yaml
```

runがこのキャッシュを使用するためには、Launchキュー設定に`volumes`