---
title: Minikube でシングルノード GPU クラスターを起動する
menu:
  launch:
    identifier: ja-launch-integration-guides-minikube_gpu
    parent: launch-integration-guides
url: /ja/tutorials/minikube_gpu
---

W&B LaunchをMinikubeクラスターにセットアップし、GPU のワークロードをスケジュールして実行できるようにします。

{{% alert %}}
このチュートリアルは、複数のGPUを搭載したマシンへの直接アクセスがあるユーザーを対象としています。このチュートリアルは、クラウドマシンをレンタルしているユーザーには意図されていません。

クラウドマシン上でminikubeクラスターをセットアップしたい場合、W&Bはクラウドプロバイダーを使用したGPU対応のKubernetesクラスターを作成することを推奨します。たとえば、AWS、GCP、Azure、Coreweave、その他のクラウドプロバイダーには、GPU対応のKubernetesクラスターを作成するためのツールがあります。

単一のGPUを搭載したマシンでGPUをスケジュールするためにminikubeクラスターをセットアップしたい場合、W&Bは[Launch Dockerキュー]({{< relref path="/launch/set-up-launch/setup-launch-docker" lang="ja" >}})を使用することを推奨します。このチュートリアルを楽しくフォローすることはできますが、GPUのスケジューリングはあまり役に立たないでしょう。
{{% /alert %}}

## 背景

[Nvidia コンテナツールキット](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)のおかげで、DockerでGPUを有効にしたワークフローを簡単に実行できるようになりました。制限の一つに、ボリュームによるGPUのスケジューリングのネイティブなサポートがない点があります。`docker run` コマンドでGPUを使用したい場合は、特定のGPUをIDでリクエストするか、存在するすべてのGPUをリクエストする必要がありますが、これは多くの分散GPUを有効にしたワークロードを非現実的にします。Kubernetesはボリュームリクエストによるスケジューリングをサポートしていますが、GPUスケジューリングを備えたローカルKubernetesクラスターのセットアップには、最近までかなりの時間と労力がかかっていました。Minikubeは、シングルノードKubernetesクラスターを実行するための最も人気のあるツールの1つであり、最近 [GPUスケジューリングのサポート](https://minikube.sigs.k8s.io/docs/tutorials/nvidia/) をリリースしました 🎉 このチュートリアルでは、マルチGPUマシンにMinikubeクラスターを作成し、W&B Launchを使用してクラスターに並行して安定的な拡散推論ジョブを起動します 🚀

## 前提条件

始める前に、次のものが必要です：

1. W&Bアカウント。
2. 以下がインストールされているLinuxマシン：
   1. Docker runtime
   2. 使用したいGPU用のドライバ
   3. Nvidiaコンテナツールキット

{{% alert %}}
このチュートリアルをテストし作成するために、4 NVIDIA Tesla T4 GPUを接続した `n1-standard-16` Google Cloud Compute Engineインスタンスを使用しました。
{{% /alert %}}

## Launchジョブ用のキューを作成

最初に、launchジョブ用のlaunchキューを作成します。

1. [wandb.ai/launch](https://wandb.ai/launch)（またはプライベートW&Bサーバーを使用している場合は `<your-wandb-url>/launch`）に移動します。
2. 画面の右上隅にある青い **Create a queue** ボタンをクリックします。キュー作成のドロワーが画面の右側からスライドアウトします。
3. エンティティを選択し、名前を入力し、キューのタイプとして **Kubernetes** を選択します。
4. ドロワーの **Config** セクションには、launchキュー用の[Kubernetesジョブ仕様](https://kubernetes.io/docs/concepts/workloads/controllers/job/)を入力します。このキューから起動されたrunは、このジョブ仕様を使用して作成されるため、必要に応じてジョブをカスタマイズするためにこの設定を変更できます。このチュートリアルでは、下記のサンプル設定をキューの設定にYAMLまたはJSONとしてコピー＆ペーストできます：

{{< tabpane text=true >}}
{{% tab "YAML" %}}
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
{{% /tab %}}
{{% tab "JSON" %}}
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
{{% /tab %}}
{{< /tabpane >}}

キュー設定の詳細については、 [Set up Launch on Kubernetes]({{< relref path="/launch/set-up-launch/setup-launch-kubernetes.md" lang="ja" >}})と [Advanced queue setup guide]({{< relref path="/launch/set-up-launch/setup-queue-advanced.md" lang="ja" >}}) を参照してください。   

`${image_uri}` と `{{gpus}}` 文字列は、キュー設定で使用できる2種類の変数テンプレートの例です。`${image_uri}` テンプレートは、エージェントが起動するジョブの画像URIに置き換えられます。`{{gpus}}` テンプレートは、ジョブを送信する際にlaunch UI、CLI、またはSDKからオーバーライドできるテンプレート変数の作成に使用されます。これらの値はジョブ仕様に配置され、ジョブで使用される画像とGPUリソースを制御する正しいフィールドを変更します。

5. **Parse configuration** ボタンをクリックして `gpus` テンプレート変数をカスタマイズし始めます。
6. **Type** を `Integer` に設定し、 **Default** 、 **Min** 、 **Max** を選択した値に設定します。このテンプレート変数の制約を違反するrunをこのキューに送信しようとすると、拒否されます。

{{< img src="/images/tutorials/minikube_gpu/create_queue.png" alt="gpusテンプレート変数を使用したキュー作成ドロワーの画像" >}}

7. **Create queue** をクリックしてキューを作成します。新しいキューのキューページにリダイレクトされます。

次のセクションでは、作成したキューからジョブをプルして実行できるエージェントをセットアップします。

## Docker ＋ NVIDIA CTKのセットアップ

既にマシンでDockerとNvidiaコンテナツールキットを設定している場合は、このセクションをスキップできます。

[Dockerのドキュメント](https://docs.docker.com/engine/install/)を参照して、システム上でのDockerコンテナエンジンのセットアップ手順を確認してください。

Dockerがインストールされたら、その後にNvidiaコンテナツールキットを[Nvidiaのドキュメント](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) に従ってインストールします。

コンテナランタイムがGPUにアクセスできることを確認するには、次を実行します：

```bash
docker run --gpus all ubuntu nvidia-smi
```

マシンに接続されているGPUを記述する `nvidia-smi` の出力が表示されるはずです。たとえば、私たちのセットアップでは、出力は次のようになります：

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

MinikubeのGPUサポートにはバージョン`v1.32.0`以上が必要です。[Minikubeのインストールドキュメント](https://minikube.sigs.k8s.io/docs/start/) を参照して、最新のインストール方法を確認してください。このチュートリアルでは、次のコマンドを使用して最新のMinikubeリリースをインストールしました：

```yaml
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

次のステップは、GPUを使用してminikubeクラスターを開始することです。マシン上で以下を実行します：

```yaml
minikube start --gpus all
```

上記のコマンドの出力は、クラスターが正常に作成されたかどうかを示します。

## launch エージェントを開始

新しいクラスター向けのlaunchエージェントは、`wandb launch-agent`を直接呼び出すか、[W&Bによって管理されるHelmチャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)を使用してlaunchエージェントをデプロイすることによって開始されます。

このチュートリアルでは、エージェントをホストマシンで直接実行します。

{{% alert %}}
コンテナ外でエージェントを実行することも、ローカルDockerホストを使用してクラスター用の画像を構築できることを意味します。
{{% /alert %}}

エージェントをローカルで実行するには、デフォルトのKubernetes APIコンテキストがMinikubeクラスターを指していることを確認してください。次に、以下を実行してエージェントの依存関係をインストールします：

```bash
pip install "wandb[launch]"
```

エージェントの認証を設定するには、`wandb login`を実行するか、`WANDB_API_KEY`環境変数を設定します。

エージェントを開始するには、次のコマンドを実行します：

```bash
wandb launch-agent -j <max-number-concurrent-jobs> -q <queue-name> -e <queue-entity>
```

ターミナル内でlaunchエージェントがポーリングメッセージを印刷し始めるのを確認できます。

おめでとうございます、launchエージェントがlaunchキューのポーリングを行っています。キューにジョブが追加されると、エージェントがそれを受け取り、Minikubeクラスターで実行するようスケジュールします。

## ジョブを起動

エージェントにジョブを送信しましょう。W&Bアカウントにログインしているターミナルから、以下のコマンドで簡単な "hello world" を起動できます：

```yaml
wandb launch -d wandb/job_hello_world:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

好きなジョブやイメージでテストできますが、クラスターがイメージをプルできることを確認してください。追加のガイダンスについては、[Minikubeのドキュメント](https://minikube.sigs.k8s.io/docs/handbook/registry/)を参照してください。また、[私たちの公開ジョブを使用してテスト](https://wandb.ai/wandb/jobs/jobs?workspace=user-bcanfieldsherman)することもできます。

## (オプション) NFSを用いたモデルとデータキャッシング

ML ワークロードのために、複数のジョブが同じデータにアクセスできるようにしたい場合があります。たとえば、大規模なアセット（データセットやモデルの重みなど）の再ダウンロードを避けるために共有キャッシュを持つことができます。Kubernetesは、[永続ボリュームと永続ボリュームクレーム](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)を通じてこれをサポートしています。永続ボリュームは、Kubernetesワークロードにおいて `volumeMounts` を作成し、共有キャッシュへの直接ファイルシステムアクセスを提供します。

このステップでは、モデルの重みをキャッシュとして共有するために使用できるネットワークファイルシステム（NFS）サーバーをセットアップします。最初のステップはNFSをインストールし、設定することです。このプロセスはオペレーティングシステムによって異なります。私たちのVMはUbuntuを実行しているので、nfs-kernel-serverをインストールし、`/srv/nfs/kubedata`でエクスポートを設定しました：

```bash
sudo apt-get install nfs-kernel-server
sudo mkdir -p /srv/nfs/kubedata
sudo chown nobody:nogroup /srv/nfs/kubedata
sudo sh -c 'echo "/srv/nfs/kubedata *(rw,sync,no_subtree_check,no_root_squash,no_all_squash,insecure)" >> /etc/exports'
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

ホストファイルシステムのサーバーのエクスポート先と、NFSサーバーのローカルIPアドレスをメモしておいてください。次のステップでこの情報が必要です。

次に、このNFSの永続ボリュームと永続ボリュームクレームを作成する必要があります。永続ボリュームは非常にカスタマイズ可能ですが、シンプlicityのために、ここではシンプルな設定を使用します。

以下のyamlを `nfs-persistent-volume.yaml` というファイルにコピーし、希望のボリュームキャパシティとクレームリクエストを記入してください。`PersistentVolume.spec.capcity.storage` フィールドは、基になるボリュームの最大サイズを制御します。`PersistentVolumeClaim.spec.resources.requests.storage` は、特定のクレームに割り当てられるボリュームキャパシティを制限するために使用できます。私たちのユースケースでは、それぞれに同じ値を使用するのが理にかなっています。

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nfs-pv
spec:
  capacity:
    storage: 100Gi # あなたの希望の容量に設定してください。
  accessModes:
    - ReadWriteMany
  nfs:
    server: <your-nfs-server-ip> # TODO: ここを記入してください。
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
      storage: 100Gi # あなたの希望の容量に設定してください。
  storageClassName: ''
  volumeName: nfs-pv
```

以下のコマンドでクラスターにリソースを作成します：

```yaml
kubectl apply -f nfs-persistent-volume.yaml
```

私たちのrunがこのキャッシュを使用できるようにするためには、launchキューの設定に `volumes` と `volumeMounts` を追加する必要があります。launchの設定を編集するには、再び [wandb.ai/launch](http://wandb.ai/launch)（またはwandbサーバー上のユーザーの場合は\<your-wandb-url\>/launch）に戻り、キューを見つけ、キューページに移動し、その後、**Edit config** タブをクリックします。元の設定を次のように変更できます：

{{< tabpane text=true >}}
{{% tab "YAML" %}}
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
{{% /tab %}}
{{% tab "JSON" %}}
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
{{% /tab %}}
{{< /tabpane >}}

これで、NFSはジョブを実行しているコンテナの `/root/.cache` にマウントされます。コンテナが `root` 以外のユーザーとして実行される場合、マウントパスは調整が必要です。HuggingfaceのライブラリとW&B Artifactsは、デフォルトで `$HOME/.cache/` を利用しているため、ダウンロードは一度だけ行われるはずです。

## 安定拡散と遊ぶ

新しいシステムをテストするために、安定的な拡散の推論パラメータを実験します。
デフォルトのプロンプトと常識的なパラメータでシンプルな安定的拡散推論ジョブを実行するには、次のコマンドを実行します：

```
wandb launch -d wandb/job_stable_diffusion_inference:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

上記のコマンドは、あなたのキューに `wandb/job_stable_diffusion_inference:main` コンテナイメージを送信します。
エージェントがジョブを受け取り、クラスターで実行するためにスケジュールするとき、接続に依存してイメージがプルされるまで時間がかかることがあります。
ジョブのステータスは[wandb.ai/launch](http://wandb.ai/launch)（またはwandbサーバー上のユーザーの場合の\<your-wandb-url\>/launch）キューページで確認できます。

runが終了すると、指定したプロジェクトにジョブアーティファクトがあるはずです。
プロジェクトのジョブページ (`<project-url>/jobs`) にアクセスしてジョブアーティファクトを見つけることができます。デフォルトの名前は `job-wandb_job_stable_diffusion_inference` ですが、ジョブのページでジョブ名の横にある鉛筆アイコンをクリックして好きなように変更できます。

このジョブを使って、クラスター上でさらに安定的な拡散推論を実行することができます。
ジョブページから、右上隅にある **Launch** ボタンをクリックして新しい推論ジョブを設定し、キューに送信します。ジョブの設定ページは、元のrunからのパラメータで自動的に入力されますが、 **Overrides** セクションで値を変更することで好きなように変更できます。

{{< img src="/images/tutorials/minikube_gpu/sd_launch_drawer.png" alt="安定拡散推論ジョブのlaunch UIの画像" >}}