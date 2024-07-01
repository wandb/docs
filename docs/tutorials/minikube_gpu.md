import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Spin up a single node GPU cluster with Minikube

MinikubeクラスターでGPUワークロードをスケジュールし実行するためのW&B Launchのセットアップ方法。

:::info
このチュートリアルは複数GPUを搭載したマシンに直接アクセスできるユーザーを対象としています。このチュートリアルはクラウドマシンをレンタルするユーザー向けではありません。

クラウドマシンにMinikubeクラスターをセットアップしたい場合、W&Bはクラウドプロバイダ（AWS、GCP、Azure、Coreweave など）の提供するGPUサポート付きKubernetesクラスター作成ツールを使用することを推奨します。

1台のGPUを搭載したマシンでGPUをスケジュールするためにMinikubeクラスターをセットアップしたい場合、W&Bは[Launch Docker queue](/guides/launch/setup-launch-docker)を使用することを推奨します。このチュートリアルを参考にしても楽しいですが、GPUのスケジューリングはあまり役に立たないでしょう。
:::

## 背景

[Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)により、Docker上でGPU対応のワークフローを実行することが簡単になりました。制限の一つは、ボリュームによるGPUのスケジューリングのネイティブサポートがないことです。`docker run` コマンドでGPUを使用する場合、特定のGPUをIDで指定するか、全てのGPUを指定する必要があり、多くの分散GPU対応ワークロードが実用的ではありません。Kubernetes はボリュームリクエストによるスケジューリングをサポートしていますが、GPUスケジューリングを持つローカルKubernetesクラスターのセットアップには多くの時間と労力がかかります。しかし最近、人気のある単一ノードのKubernetesクラスターを実行するためのツールの一つ、Minikube が[GPUスケジューリングをサポート](https://minikube.sigs.k8s.io/docs/tutorials/nvidia/)しました 🎉 このチュートリアルでは、複数GPUを搭載したマシンにMinikubeクラスターを作成し、W&B Launch を使用してクラスターに同時実行の安定拡散推論ジョブを送信します 🚀

## 前提条件

開始する前に以下が必要です：

1. W&Bアカウント。
2. 以下の要件を満たすLinuxマシン：
   1. Dockerランタイム
   2. 使用したいGPUのドライバ
   3. Nvidia container toolkit

:::note
このチュートリアルをテストして作成するために、4つのNVIDIA Tesla T4 GPUが接続された`n1-standard-16`のGoogle Cloud Compute Engineインスタンスを使用しました。
:::

## ジョブのキューを作成する

まず、Launchジョブのためのキューを作成します。

1. [wandb.ai/launch](https://wandb.ai/launch)（またはプライベートW&Bサーバーを使用している場合は`<your-wandb-url>/launch`）に移動します。
2. 画面の右上にある青色の**Create a queue**ボタンをクリックします。右側からキュー作成ドロワーがスライドアウトします。
3. エンティティを選択し、名前を入力し、キューのタイプとして**Kubernetes**を選択します。
4. ドロワーの**Config**セクションに[「Kubernetesジョブ仕様](https://kubernetes.io/docs/concepts/workloads/controllers/job/)」を入力します。このキューから起動されたすべてのRuns はこのジョブ仕様で作成されるため、ジョブをカスタマイズするために必要に応じて設定を変更できます。このチュートリアルでは、下記のサンプル構成をYAMLまたはJSON形式でキュー設定にコピー＆ペーストしてください：

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

キュー設定に関する詳細情報については、[Set up Launch on Kubernetes](/guides/launch/setup-launch-kubernetes.md) や [Advanced queue setup guide](/guides/launch/setup-queue-advanced.md) を参照してください。

`${image_uri}` と `{{gpus}}` はそれぞれキュー設定で使用できる2種類の変数テンプレートの例です。`${image_uri}` テンプレートは、エージェントが起動するジョブのイメージURIで置き換えられます。`{{gpus}}` テンプレートは、ジョブを提出する際にLaunch UI、CLI、またはSDKからオーバーライドできるテンプレート変数を作成するのに使用されます。これらの値はジョブ仕様に挿入され、ジョブで使用されるイメージとGPUリソースを制御する正しいフィールドが変更されるようになります。

5. **Parse configuration** ボタンをクリックして、`gpus`テンプレート変数をカスタマイズします。
6. **Type**を`Integer`に設定し、**Default**、**Min**、および**Max**を適宜設定します。このテンプレート変数の制約を違反するキューへのRunの提出は拒否されます。

![Image of queue creation drawer with gpus template variable](/images/tutorials/minikube_gpu/create_queue.png)

7. **Create queue**をクリックしてキューを作成します。新しいキューのキューページにリダイレクトされます。

次のセクションでは、作成したキューからジョブを取得し実行するエージェントをセットアップします。

## Docker + NVIDIA CTKのセットアップ

既にマシンにDockerとNvidia container toolkitがセットアップされている場合は、このセクションをスキップできます。

[Dockerのドキュメント](https://docs.docker.com/engine/install/) を参照して、システムにDockerコンテナエンジンをセットアップする手順を確認してください。

Dockerをインストールしたら、 [Nvidiaのドキュメントの指示に従って](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)Nvidia container toolkitをインストールしてください。

コンテナランタイムがGPUにアクセスできることを確認するため、次のコマンドを実行します：

```bash
docker run --gpus all ubuntu nvidia-smi
```

`nvidia-smi`の出力に、マシンに接続されたGPUの情報が表示されるはずです。例えば、私たちのセットアップでは以下のように表示されます：

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

MinikubeのGPUサポートはバージョン `v1.32.0` 以降が必要です。インストールの詳細については [Minikubeのインストールドキュメント](https://minikube.sigs.k8s.io/docs/start/) を参照してください。このチュートリアルでは、最新のMinikubeリリースを以下のコマンドを使用してインストールしました：

```yaml
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

次のステップはGPUを使用してMinikubeクラスターを開始することです。マシンで次のコマンドを実行します：

```yaml
minikube start --gpus all
```

上記のコマンドの出力にクラスターが正常に作成されたかどうかが表示されます。

## Launchエージェントの開始

新しいクラスターのLaunchエージェントは、`wandb launch-agent`を直接実行するか、[W&Bが管理するhelm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)を使用してデプロイすることで開始できます。

このチュートリアルでは、ホストマシン上でエージェントを直接実行します。

:::tip
コンテナ外でエージェントを実行することで、ローカルのDockerホストを使用してクラスター用のイメージをビルドすることもできます。
:::

エージェントをローカルで実行するには、デフォルトのKubernetes APIコンテキストがMinikubeクラスターを指していることを確認します。それから以下を実行します：

```bash
pip install wandb[launch]
```

エージェントの依存関係をインストールします。エージェントの認証セットアップには、`wandb login`を実行するか、`WANDB_API_KEY`環境変数を設定します。

エージェントを開始するには、次のコマンドを入力して実行します：

```bash
wandb launch-agent -j <max-number-concurrent-jobs> -q <queue-name> -e <queue-entity>
```

ターミナル内でエージェントがポーリングメッセージを出力し始めるのが確認できます。

おめでとうございます、LaunchエージェントがあなたのLaunchキューをポーリングしています！キューにジョブが追加されると、エージェントがそれを取得し、Minikubeクラスター上で実行をスケジューリングします。

## ジョブを起動する

エージェントにジョブを送信しましょう。ターミナルにログインした状態で以下のコマンドを実行して、シンプルな「Hello World」をLaunchします：

```yaml
wandb launch -d wandb/job_hello_world:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

この方法で任意のジョブやイメージをテストできますが、クラスターがイメージをプルできることを確認してください。追加のガイダンスについては[Minikubeのドキュメント](https://minikube.sigs.k8s.io/docs/handbook/registry/)を参照してください。また、[我々の公開ジョブの一つをテストする](https://wandb.ai/wandb/jobs/jobs?workspace=user-bcanfieldsherman)こともできます。

## (オプション) モデルとデータのキャッシング（NFSを使用）

MLワークロードの場合、複数のジョブが同じデータにアクセスすることが望ましい場合があります。例えば、大容量のデータセットやモデルのウェイトを繰り返しダウンロードするのを避けるために、共有キャッシュを持つことが考えられます。Kubernetesは[永続ボリュームと永続ボリューム要求](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)をサポートしています。永続ボリュームはKubernetesワークロードに`volumeMounts`を作成するために使用され、共有キャッシュへの直接的なファイルシステムアクセスを提供します。

このステップでは、モデルウェイトの共有キャッシュとして使用できるネットワークファイルシステム（NFS）サーバーをセットアップします。最初のステップはNFSのインストールと設定です。このプロセスはオペレーティングシステムによって異なります。我々のVMはUbuntuを実行しているため、nfs-kernel-serverをインストールし、`/srv/nfs/kubedata`にエクスポートを設定しました：

```bash
sudo apt-get install nfs-kernel-server
sudo mkdir -p /srv/nfs/kubedata
sudo chown nobody:nogroup /srv/nfs/kubedata
sudo sh -c 'echo "/srv/nfs/kubedata *(rw,sync,no_subtree_check,no_root_squash,no_all_squash,insecure)" >> /etc/exports'
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

ホストファイルシステムのサーバーのエクスポート位置とNFSサーバーのローカルIPアドレス、すなわちホストマシンの情報をメモしておいてください。次のステップでこれらの情報を使用します。

次に、このNFSの永続ボリュームと永続ボリューム要求を作成する必要があります。永続ボリュームは非常にカスタマイズ可能ですが、簡潔にするためにここではシンプルな設定を使用します。

下記のyamlを`nfs-persistent-volume.yaml`というファイルにコピーし、希望するボリューム容量と要求を入力してください。`PersistentVolume.spec.capcity.storage` フィールドは基礎となるボリュームの最大サイズを制御します。`PersistentVolumeClaim.spec.resources.requests.storage` は特定の要求に割り当てるボリューム容量を制限するために使用できます。今回のユースケースでは、これらの値を同じにするのが理にかなっています。

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nfs-pv
spec:
  capacity:
    storage: 100Gi # 希望する容量を設定してください。
  accessModes:
    - ReadWriteMany
  nfs:
    server: <your-nfs-server-ip> # TODO: ここを入力してください
    path: '/srv/nfs/kubedata' # またはお好みのパス
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
      storage: 100Gi # 希望する容量を設定してください。
  storageClassName: ''
  volumeName: nfs-pv
```

クラスターで次のコマンドを実行してリソースを作成します：

```yaml
kubectl apply -f nfs-persistent-volume.yaml
```

Runをこのキャッシュで利用できるようにするには、Launch queue configに`volumes`と`volumeMounts`を追加する必要があります。Launch設定を編集するには、もう一度[wandb.ai/launch](http://wandb.ai/launch)（または\<your-wandb-url\>/launch図で`wandb server`を使用するユーザーの場合）に戻り、キューを検索して、キューページに移動してから**Edit config**タブをクリックします。もとの設定は次のように変更できます：

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
              nvidia.com/gpu: "{{gpus}}"
          volumeMounts:
            - name: nfs-storage
              mountPath: /root/.cache
      restartPolicy: Never
      volumes:
        - name: nfs-storage
          persistentVolumeClaim:
            claimName: nfs-pvc
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
              },
              "volumeMounts": [
                {
                  "name": "nfs-storage",
                  "mountPath": "/root/.cache"
                }
              ]
            }
          }
        ],
        "restartPolicy": "Never",
        "volumes": [
          {
            "name": "nfs-storage",
            "persistentVolumeClaim": {
              "claimName": "nfs-pvc"
            }
          }
        ]
      }
    },
    "backoffLimit": 0
  }
}
```

</TabItem>

</Tabs>

これで、私たちのNFSはコンテナ実行中のジョブで`/root/.cache`にマウントされます。マウントパスは、コンテナが`root`以外のユーザーとして実行される場合には調整が必要です。HuggingfaceのライブラリとW&B Artifactsはどちらもデフォルトで`$HOME/.cache/`を使用するため、ダウンロードは一度だけ行われるでしょう。

## Stable Diffusionで遊ぶ

新しいシステムをテストするために、Stable Diffusionのインファレンスパラメータを実験します。
デフォルトのプロンプトと合理的なパラメータでシンプルなStable Diffusionのインファレンスジョブを実行するには、次のコマンドを実行します：

```
wandb launch -d wandb/job_stable_diffusion_inference:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

上記のコマンドは`wandb/job_stable_diffusion_inference:main`のコンテナイメージをキューに送信します。
エージェントがジョブを取得し、クラスターで実行をスケジュールした後、接続状況に応じてイメージのプルに時間がかかる場合があります。
キューのページ（[wandb.ai/launch](http://wandb.ai/launch)または\<your-wandb-url\>/launch図で`wandb server`を使用するユーザーの場合）でジョブのステータスを確認できます。

Runが終了すると、指定したプロジェクトにジョブアーティファクトが生成されます。
プロジェクトのジョブページ（`<project-url>/jobs`）でジョブアーティファクトを確認できます。デフォルトの名前は`job-wandb_job_stable_diffusion_inference`ですが、ジョブページで名前をクリックして鉛筆アイコンをクリックすることで、好きなように変更できます。

このジョブを使用してクラスタ上でさらにStable Diffusionのインファレンスを実行できます！
ジョブページから、右上の**Launch**ボタンをクリックして新しいインファレンスジョブを設定し、そのキューに送信できます。ジョブ設定ページには、元のRunのパラメータが事前入力されますが、Launchドロワーの**Overrides**セクションでそれらの値を変更することができます。

