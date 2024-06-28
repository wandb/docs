import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Minikubeを使って単一ノードのGPUクラスターを立ち上げる

Minikubeクラスター上で W&B Launch をセットアップし、GPUのワークロードをスケジュールして実行します。

:::info
このチュートリアルは、複数のGPUを持つマシンに直接アクセスできるユーザーを対象としています。クラウドマシンをレンタルしているユーザーにはこのチュートリアルは意図されていません。

クラウドマシン上でminikubeクラスターをセットアップしたい場合、W&Bはクラウドプロバイダが提供するGPUサポート付きのKubernetesクラスターを作成することを推奨します。たとえば、AWS、GCP、Azure、Coreweaveなどのクラウドプロバイダには、GPU対応のKubernetesクラスターを作成するツールがあります。

単一GPUを持つマシンでGPUをスケジュールするためにminikubeクラスターを設定したい場合は、W&Bが[Launch Docker キュー](/guides/launch/setup-launch-docker)の使用を推奨します。このチュートリアルを楽しむために実施することは可能ですが、GPUスケジューリングはあまり役に立たないでしょう。
:::

## 背景

[Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) により、Docker上でGPU対応のワークフローを実行することが容易になりました。しかし、ボリュームによるGPUスケジューリングのネイティブサポートがないという制約があります。`docker run`コマンドでGPUを使用したい場合、特定のGPUをIDで指定するか、全てのGPUを使用するしかなく、多くの分散型GPU対応ワークロードが実用的でなくなります。Kubernetesはボリューム要求によるスケジューリングをサポートしていますが、ローカルKubernetesクラスターでGPUスケジューリングを設定するのは、かなりの時間と労力がかかります。最近まで。この問題を解決するため、シングルノードのKubernetesクラスターを実行するための最も人気のあるツールの一つであるMinikubeが、最近[GPUスケジューリングをサポート](https://minikube.sigs.k8s.io/docs/tutorials/nvidia/)しました 🎉 本チュートリアルでは、複数GPUマシン上にMinikubeクラスターを作成し、W&B Launch 🚀 を使用してクラスターに並行して安定した拡散推論ジョブを起動します。

## 前提条件

始める前に、次のものが必要です:

1. W&Bアカウント。
2. 次のものがインストールされ、稼働しているLinuxマシン:
   1. Dockerランタイム
   2. 使用したい任意のGPUのドライバ
   3. Nvidia container toolkit

:::note
本チュートリアルのテストと作成のため、4つのNVIDIA Tesla T4 GPUが接続された`n1-standard-16` Google Cloud Compute Engineインスタンスを使用しました。
:::

## Launchジョブ用のキューを作成する

最初に、Launchジョブ用のキューを作成します。

1. [wandb.ai/launch](https://wandb.ai/launch)（プライベートW&Bサーバーを使用している場合は`<your-wandb-url>/launch`）に移動します。
2. 画面右上の青い **Create a queue** ボタンをクリックします。キュー作成用のドロワーが画面右側からスライドアウトします。
3. エンティティを選択し、名前を入力し、キューのタイプとして **Kubernetes** を選択します。
4. ドロワーの **Config** セクションに、Launchキューのための[Kubernetesジョブ仕様](https://kubernetes.io/docs/concepts/workloads/controllers/job/)を入力します。このキューから起動される任意のrunは、このジョブ仕様を使用して作成されますので、この設定を必要に応じてカスタマイズすることが可能です。チュートリアル用には、以下のサンプル設定をYAMLまたはJSONとしてキューの設定にコピー＆ペーストしてください:

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

キュー設定の詳細については、[Kubernetes上でLaunchのセットアップ](/guides/launch/setup-launch-kubernetes.md)および[高度なキュー設定ガイド](/guides/launch/setup-queue-advanced.md)をご参照ください。

`${image_uri}` および `{{gpus}}` 文字列は、キュー設定で使用できる2種類の変数テンプレートの例です。`${image_uri}`テンプレートは、起動するジョブのイメージURIにエージェントによって置き換えられます。`{{gpus}}`テンプレートは、ジョブ送信時にLaunch UI、CLI、またはSDKからオーバーライドできるテンプレート変数を作成するために使用されます。これらの値はジョブ仕様に配置され、ジョブで使用される画像およびGPUリソースを制御する正しいフィールドを修正します。

5. **Parse configuration** ボタンをクリックして、`gpus` テンプレート変数をカスタマイズし始めます。
6. **Type** を `Integer` に設定し、**Default**、**Min**、および **Max** を希望の値に設定します。
テンプレート変数の制約に違反するキューへのrunの送信は拒否されます。

![gpusテンプレート変数を含むキュー作成ドロワーの画像](/images/tutorials/minikube_gpu/create_queue.png)

7. **Create queue** をクリックしてキューを作成します。新しいキューのキューページにリダイレクトされます。

次のセクションでは、作成したキューからジョブを取得して実行するエージェントを設定します。

## Docker + NVIDIA CTK の設定

すでにマシン上にDockerとNvidia container toolkitをセットアップしている場合は、このセクションをスキップできます。

Dockerコンテナエンジンのセットアップに関する手順については、[Dockerのドキュメント](https://docs.docker.com/engine/install/)をご参照ください。

Dockerをインストールしたら、Nvidia container toolkitを[インストールガイド](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)に従ってインストールします。

コンテナランタイムがGPUにアクセスできることを検証するために、以下を実行します:

```bash
docker run --gpus all ubuntu nvidia-smi
```

このコマンドの出力には、マシンに接続されたGPUを示す`nvidia-smi`の出力が表示されます。たとえば、私たちの設定では以下のような出力になります:

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

MinikubeのGPUサポートにはバージョン `v1.32.0` 以上が必要です。最新のインストールヘルプについては、[Minikubeのインストールドキュメント](https://minikube.sigs.k8s.io/docs/start/)をご参照ください。このチュートリアルでは、以下のコマンドを使用して最新のMinikubeリリースをインストールしました:

```yaml
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

次のステップは、GPUを使用してminikubeクラスターを起動することです。以下を実行します:

```yaml
minikube start --gpus all
```

上記コマンドの出力により、クラスターが正常に作成されたかどうかが表示されます。

## Launchエージェントの開始

新しいクラスター用のLaunchエージェントは、`wandb launch-agent` を直接実行して開始するか、[W&Bが管理するhelmチャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使用してLaunchエージェントをデプロイすることができます。

このチュートリアルでは、エージェントをホストマシンで直接実行します。

:::tip
コンテナの外でエージェントを実行することにより、ローカルのDockerホストを使用してクラスター用のイメージを構築することもできます。
:::

エージェントをローカルで実行するには、デフォルトのKubernetes APIコンテキストがMinikubeクラスターを参照していることを確認します。その後、次のように実行します:

```bash
pip install wandb[launch]
```

これで、エージェントの依存関係がインストールされます。エージェントの認証を設定するには、`wandb login` を実行するか、`WANDB_API_KEY` 環境変数を設定します。

エージェントを開始するには、以下のコマンドを入力し実行します:

```bash
wandb launch-agent -j <max-number-concurrent-jobs> -q <queue-name> -e <queue-entity>
```

ターミナルにエージェントのポーリングメッセージが表示されるはずです。

おめでとうございます！LaunchキューをポーリングするLaunchエージェントが設定されました。キューにジョブが追加されると、エージェントがそれを受け取り、Minikubeクラスターで実行をスケジュールします。

## ジョブを起動する

さあ、エージェントにジョブを送りましょう。W&Bアカウントにログインしているターミナルから簡単な"hello world"を起動できます:

```yaml
wandb launch -d wandb/job_hello_world:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

好きなジョブやイメージでテストできますが、クラスターがイメージをプルできることを確認してください。詳細については[Minikubeのドキュメント](https://minikube.sigs.k8s.io/docs/handbook/registry/)をご参照ください。[W&Bの公開ジョブの1つを使用してテスト](https://wandb.ai/wandb/jobs/jobs?workspace=user-bcanfieldsherman)することもできます。

## (オプション) NFSによるモデルとデータのキャッシュ

MLワークロードのためには、複数のジョブが同じデータにアクセスできることがよく求められます。例えば、データセットやモデルの重みのような大きなアセットを繰り返しダウンロードするのを避けるために、共有キャッシュを持ちたいかもしれません。Kubernetesはこれを [永続ボリュームと永続ボリューム要求](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) でサポートします。永続ボリュームはKubernetesワークロードに `volumeMounts` を作成するために使用でき、共有キャッシュへの直接ファイルシステムアクセスを提供します。

このステップでは、モデルの重みの共有キャッシュとして使用できるネットワークファイルシステム（NFS）サーバーをセットアップします。最初のステップはNFSをインストールして設定することです。このプロセスはオペレーティングシステムによって異なります。私たちのVMはUbuntuを実行しているので、nfs-kernel-serverをインストールし、`/srv/nfs/kubedata`にエクスポートを設定しました：

```bash
sudo apt-get install nfs-kernel-server
sudo mkdir -p /srv/nfs/kubedata
sudo chown nobody:nogroup /srv/nfs/kubedata
sudo sh -c 'echo "/srv/nfs/kubedata *(rw,sync,no_subtree_check,no_root_squash,no_all_squash,insecure)" >> /etc/exports'
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

ホストファイルシステム内のサーバーのエクスポート場所とNFSサーバーのローカルIPアドレス、つまりホストマシンのIPアドレスをメモしておいてください。次のステップでこれらの情報が必要になります。

次に、このNFS用の永続ボリュームと永続ボリューム要求を作成する必要があります。永続ボリュームは非常にカスタマイズ可能ですが、簡単にするために、ここではシンプルな設定を使用します。

以下のyamlを `nfs-persistent-volume.yaml` という名前のファイルにコピーし、希望するボリューム容量と要求を入力してください。`PersistentVolume.spec.capacity.storage` フィールドは基盤となるボリュームの最大サイズを制御します。`PersistentVolumeClaim.spec.resources.requests.storage` は特定の要求に割り当てるボリューム容量を制限するために使用できます。ユースケース上は、どちらも同じ値を使用するのが理にかなっています。

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nfs-pv
spec:
  capacity:
    storage: 100Gi # 希望する容量に設定してください
  accessModes:
    - ReadWriteMany
  nfs:
    server: <your-nfs-server-ip> # ここにNFSサーバーのIPを入力してください
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
      storage: 100Gi # 希望する容量に設定してください
  storageClassName: ''
  volumeName: nfs-pv
```

以下のコマンドでクラスタにリソースを作成します：

```yaml
kubectl apply -f nfs-persistent-volume.yaml
```

このキャッシュを用いてRunsが利用できるようにするためには、launchキュー設定に `volumes` と `volumeMounts` を追加する必要があります。設定を編集するために、[wandb.ai/launch](http://wandb.ai/launch) (またはwandbサーバーのユーザーの場合は\<your-wandb-url\>/launch) にアクセスし、キューを見つけてキューページに移動し、**Edit config** タブをクリックします。元の設定を以下のように修正できます：

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

これで、NFSはジョブを実行するコンテナ内の `/root/.cache` にマウントされます。コンテナが `root` 以外のユーザーとして実行される場合は、マウントパスを調整する必要があります。HuggingfaceのライブラリとW&B Artifactsはデフォルトで `$HOME/.cache/` を使用するため、ダウンロードは一度だけ行われるはずです。

## ステーブルディフュージョンの実験

新しいシステムをテストするために、ステーブルディフュージョンの推論パラメータで実験します。
デフォルトのプロンプトと健全なパラメータでシンプルなステーブルディフュージョン推論ジョブを実行するには、以下のコマンドを実行します：

```
wandb launch -d wandb/job_stable_diffusion_inference:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

上記のコマンドは、コンテナイメージ `wandb/job_stable_diffusion_inference:main` をキューに提出します。
エージェントがジョブを取得し、クラスターで実行するためにスケジュールすると、
接続状況に応じて、イメージの取得に時間がかかる場合があります。
ジョブのステータスは、 [wandb.ai/launch](http://wandb.ai/launch) (またはwandbサーバーのユーザーの場合は\<your-wandb-url\>/launch) で確認できます。

ジョブが完了すると、指定したプロジェクトにジョブアーティファクトが作成されます。
プロジェクトのジョブページ（`<project-url>/jobs`）でジョブアーティファクトを見つけることができます。デフォルトの名前は `job-wandb_job_stable_diffusion_inference` ですが、ジョブのページでジョブ名の横にある鉛筆アイコンをクリックして、好きな名前に変更できます。

このジョブを使用して、クラスターでさらにステーブルディフュージョン推論を実行できます！
ジョブページから、右上の **Launch** ボタンをクリックして、新しい推論ジョブを構成し、キューに提出できます。
ジョブ設定ページには、元のrunのパラメータが事前入力されていますが、 **Overrides** セクションでこれらの値を変更して、任意の設定にできます。

![Image of launch UI for stable diffusion inference job](/images/tutorials/minikube_gpu/sd_launch_drawer.png)