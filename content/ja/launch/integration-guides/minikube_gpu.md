---
title: Spin up a single node GPU cluster with Minikube
menu:
  launch:
    identifier: ja-launch-integration-guides-minikube_gpu
    parent: launch-integration-guides
url: tutorials/minikube_gpu
---

GPU ワークロードをスケジュールおよび実行できる Minikube クラスター上に W&B Launch をセットアップします。

{{% alert %}}
このチュートリアルは、複数の GPU を搭載したマシンに直接アクセスできるユーザーを対象としています。クラウドマシンをレンタルしているユーザーは対象としていません。

クラウドマシン上に minikube クラスターをセットアップする場合は、クラウドプロバイダーが提供する GPU サポートを備えた Kubernetes クラスターを作成することを W&B は推奨します。たとえば、AWS、GCP、Azure、Coreweave、その他のクラウドプロバイダーには、GPU サポートを備えた Kubernetes クラスターを作成するツールがあります。

単一の GPU を搭載したマシンで GPU のスケジューリングを行うために minikube クラスターをセットアップする場合は、[Launch Docker queue]({{< relref path="/launch/set-up-launch/setup-launch-docker/" lang="ja" >}}) を使用することを W&B は推奨します。このチュートリアルも参考にできますが、GPU のスケジューリングはあまり役に立ちません。
{{% /alert %}}

## 背景

[Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) により、Docker で GPU 対応のワークフローを簡単に実行できるようになりました。1 つの制限として、ボリュームによる GPU のスケジューリングに対するネイティブサポートがありません。`docker run` コマンドで GPU を使用するには、ID で特定の GPU をリクエストするか、存在するすべての GPU をリクエストする必要があります。これにより、多くの分散 GPU 対応ワークロードが非現実的になります。Kubernetes はボリュームリクエストによるスケジューリングをサポートしていますが、GPU スケジューリングを備えたローカルの Kubernetes クラスターのセットアップには、最近までかなりの時間と労力がかかっていました。シングルノードの Kubernetes クラスターを実行するための最も一般的なツールの 1 つである Minikube は、最近 [GPU スケジューリングのサポート](https://minikube.sigs.k8s.io/docs/tutorials/nvidia/) 🎉 をリリースしました。このチュートリアルでは、マルチ GPU マシン上に Minikube クラスターを作成し、W&B Launch 🚀 を使用して、クラスターへの同時 Stable Diffusion 推論ジョブを起動します。

## 前提条件

始める前に、以下が必要です。

1. W&B アカウント。
2. 以下がインストールされ実行されている Linux マシン:
   1. Docker ランタイム
   2. 使用する GPU のドライバー
   3. Nvidia container toolkit

{{% alert %}}
このチュートリアルのテストと作成には、4 つの NVIDIA Tesla T4 GPU が接続された `n1-standard-16` Google Cloud Compute Engine インスタンスを使用しました。
{{% /alert %}}

## Launch ジョブのキューを作成する

まず、Launch ジョブの Launch キューを作成します。

1. [wandb.ai/launch](https://wandb.ai/launch)（または、プライベート W&B サーバーを使用している場合は `<your-wandb-url>/launch`）に移動します。
2. 画面の右上隅にある青い [**キューを作成**] ボタンをクリックします。キュー作成ドロワーが画面の右側からスライドして表示されます。
3. エンティティを選択し、名前を入力して、キューのタイプとして [**Kubernetes**] を選択します。
4. ドロワーの [**設定**] セクションでは、Launch キューの [Kubernetes ジョブ仕様](https://kubernetes.io/docs/concepts/workloads/controllers/job/) を入力します。このキューから起動された Run は、このジョブ仕様を使用して作成されるため、必要に応じてこの設定を変更してジョブをカスタマイズできます。このチュートリアルでは、以下のサンプル設定を YAML または JSON 形式でキュー設定にコピーして貼り付けることができます。

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

キュー設定の詳細については、[Kubernetes での Launch のセットアップ]({{< relref path="../set-up-launch/setup-launch-kubernetes.md" lang="ja" >}}) および [キューの高度なセットアップガイド]({{< relref path="../set-up-launch/setup-queue-advanced.md" lang="ja" >}}) を参照してください。

`${image_uri}` および `{{gpus}}` 文字列は、キュー設定で使用できる 2 種類の変数テンプレートの例です。`${image_uri}` テンプレートは、エージェントによって起動されるジョブのイメージ URI に置き換えられます。`{{gpus}}` テンプレートは、ジョブの送信時に Launch UI、CLI、または SDK からオーバーライドできるテンプレート変数を作成するために使用されます。これらの値はジョブ仕様に配置され、ジョブで使用されるイメージと GPU リソースを制御するために正しいフィールドを変更します。

5. [**設定の解析**] ボタンをクリックして、`gpus` テンプレート変数のカスタマイズを開始します。
6. [**タイプ**] を `Integer` に設定し、[**デフォルト**]、[**最小**]、および [**最大**] を選択した値に設定します。
テンプレート変数の制約に違反するこのキューに Run を送信しようとすると、拒否されます。

{{< img src="/images/tutorials/minikube_gpu/create_queue.png" alt="GPU テンプレート変数を含むキュー作成ドロワーの画像" >}}

7. [**キューを作成**] をクリックしてキューを作成します。新しいキューのキューページにリダイレクトされます。

次のセクションでは、作成したキューからジョブをプルして実行できるエージェントをセットアップします。

## Docker + NVIDIA CTK のセットアップ

マシンに Docker と Nvidia container toolkit がすでにセットアップされている場合は、このセクションをスキップできます。

Docker コンテナエンジンをシステムにセットアップする手順については、[Docker のドキュメント](https://docs.docker.com/engine/install/) を参照してください。

Docker をインストールしたら、[Nvidia のドキュメントの手順に従って](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) Nvidia container toolkit をインストールします。

コンテナランタイムが GPU にアクセスできることを検証するには、以下を実行します。

```bash
docker run --gpus all ubuntu nvidia-smi
```

マシンに接続されている GPU を説明する `nvidia-smi` の出力が表示されます。たとえば、セットアップでは、出力は次のようになります。

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

## Minikube のセットアップ

Minikube の GPU サポートには、バージョン `v1.32.0` 以降が必要です。最新のインストールヘルプについては、[Minikube のインストールに関するドキュメント](https://minikube.sigs.k8s.io/docs/start/) を参照してください。このチュートリアルでは、次のコマンドを使用して最新の Minikube リリースをインストールしました。

```yaml
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

次のステップは、GPU を使用して minikube クラスターを起動することです。マシンで、次を実行します。

```yaml
minikube start --gpus all
```

上記のコマンドの出力は、クラスターが正常に作成されたかどうかを示します。

## Launch エージェントの起動

新しいクラスターの Launch エージェントは、`wandb launch-agent` を直接呼び出すか、[W&B によって管理される helm チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使用して Launch エージェントをデプロイすることによって起動できます。

このチュートリアルでは、ホストマシンで直接エージェントを実行します。

{{% alert %}}
コンテナの外部でエージェントを実行すると、ローカルの Docker ホストを使用して、クラスターが実行するイメージを構築することもできます。
{{% /alert %}}

エージェントをローカルで実行するには、デフォルトの Kubernetes API コンテキストが Minikube クラスターを参照していることを確認してください。次に、以下を実行します。

```bash
pip install "wandb[launch]"
```

エージェントの依存関係をインストールします。エージェントの認証を設定するには、`wandb login` を実行するか、`WANDB_API_KEY` 環境変数を設定します。

エージェントを起動するには、次のコマンドを実行します。

```bash
wandb launch-agent -j <max-number-concurrent-jobs> -q <queue-name> -e <queue-entity>
```

ターミナル内に、Launch エージェントがポーリングメッセージの印刷を開始するのが表示されるはずです。

おめでとうございます。Launch キューをポーリングする Launch エージェントができました。ジョブがキューに追加されると、エージェントがそれを取得し、Minikube クラスターで実行するようにスケジュールします。

## ジョブの起動

エージェントにジョブを送信しましょう。W&B アカウントにログインしているターミナルから、次のコマンドで簡単な「hello world」を起動できます。

```yaml
wandb launch -d wandb/job_hello_world:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

任意のジョブまたはイメージでテストできますが、クラスターがイメージをプルできることを確認してください。詳細については、[Minikube のドキュメント](https://minikube.sigs.k8s.io/docs/handbook/registry/) を参照してください。[パブリックジョブのいずれかを使用してテスト](https://wandb.ai/wandb/jobs/jobs?workspace=user-bcanfieldsherman) することもできます。

## （オプション）NFS を使用したモデルとデータのキャッシュ

ML ワークロードでは、多くの場合、複数のジョブが同じデータにアクセスできるようにする必要があります。たとえば、データセットやモデルの重みなどの大きなアセットの繰り返しダウンロードを避けるために、共有キャッシュを用意することができます。Kubernetes は、[パーシステントボリュームとパーシステントボリュームクレーム](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) を介してこれをサポートしています。パーシステントボリュームを使用して、Kubernetes ワークロードに `volumeMounts` を作成し、共有キャッシュへの直接ファイルシステムアクセスを提供できます。

このステップでは、モデルの重みの共有キャッシュとして使用できるネットワークファイルシステム (NFS) サーバーをセットアップします。最初のステップは、NFS をインストールして構成することです。このプロセスはオペレーティングシステムによって異なります。VM は Ubuntu を実行しているため、nfs-kernel-server をインストールし、`/srv/nfs/kubedata` にエクスポートを構成しました。

```bash
sudo apt-get install nfs-kernel-server
sudo mkdir -p /srv/nfs/kubedata
sudo chown nobody:nogroup /srv/nfs/kubedata
sudo sh -c 'echo "/srv/nfs/kubedata *(rw,sync,no_subtree_check,no_root_squash,no_all_squash,insecure)" >> /etc/exports'
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

ホストファイルシステム内のサーバーのエクスポート場所と、NFS サーバーのローカル IP アドレスをメモしておきます。次のステップでこの情報が必要になります。

次に、この NFS のパーシステントボリュームとパーシステントボリュームクレームを作成する必要があります。パーシステントボリュームは高度にカスタマイズできますが、ここでは単純にするために簡単な構成を使用します。

以下の yaml を `nfs-persistent-volume.yaml` という名前のファイルにコピーし、目的のボリューム容量とクレームリクエストを入力してください。`PersistentVolume.spec.capcity.storage` フィールドは、基になるボリュームの最大サイズを制御します。`PersistentVolumeClaim.spec.resources.requests.stroage` は、特定のクレームに割り当てられたボリューム容量を制限するために使用できます。ユースケースでは、それぞれに同じ値を使用するのが理にかなっています。

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nfs-pv
spec:
  capacity:
    storage: 100Gi # 希望する容量に設定してください。
  accessModes:
    - ReadWriteMany
  nfs:
    server: <your-nfs-server-ip> # TODO: これに入力してください。
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
      storage: 100Gi # 希望する容量に設定してください。
  storageClassName: ''
  volumeName: nfs-pv
```

次のコマンドを使用して、クラスターにリソースを作成します。

```yaml
kubectl apply -f nfs-persistent-volume.yaml
```

Run がこのキャッシュを利用できるようにするには、Launch キュー設定に `volumes` と `volumeMounts` を追加する必要があります。Launch 設定を編集するには、[wandb.ai/launch](http://wandb.ai/launch) （または wandb サーバーのユーザーの場合は \<your-wandb-url\>/launch）に戻り、キューを見つけ、キューページをクリックし、[**設定の編集**] タブをクリックします。元の設定は、次のように変更できます。

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

これで、NFS がジョブを実行するコンテナの `/root/.cache` にマウントされます。コンテナが `root` 以外のユーザーとして実行される場合は、マウントパスを調整する必要があります。Huggingface のライブラリと W&B Artifacts は、デフォルトで `$HOME/.cache/` を使用するため、ダウンロードは 1 回のみ実行されます。

## Stable Diffusion を使用する

新しいシステムをテストするために、Stable Diffusion の推論パラメータを試してみましょう。
デフォルトのプロンプトと適切なパラメータを使用して、簡単な Stable Diffusion 推論ジョブを実行するには、次を実行します。

```
wandb launch -d wandb/job_stable_diffusion_inference:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

上記のコマンドは、コンテナイメージ `wandb/job_stable_diffusion_inference:main` をキューに送信します。
エージェントがジョブを取得し、クラスターで実行するようにスケジュールすると、接続によっては、イメージのプルに時間がかかる場合があります。
[wandb.ai/launch](http://wandb.ai/launch) (または、wandb サーバーのユーザーの場合は \<your-wandb-url\>/launch) のキューページでジョブのステータスを確認できます。

Run が完了すると、指定した Project にジョブ Artifact が作成されます。
Project のジョブページ (`<project-url>/jobs`) を確認して、ジョブ Artifact を見つけることができます。デフォルトの名前は `job-wandb_job_stable_diffusion_inference` ですが、ジョブの名前の横にある鉛筆アイコンをクリックして、ジョブのページで好きな名前に変更できます。

これで、このジョブを使用して、クラスターでより多くの Stable Diffusion 推論を実行できます。
ジョブページから、右上隅にある [**Launch**] ボタンをクリックして、新しい推論ジョブを構成し、キューに送信できます。ジョブ構成ページには、元の Run のパラメータが事前に入力されますが、Launch ドロワーの [**オーバーライド**] セクションで値を変更して、好きなように変更できます。

{{< img src="/images/tutorials/minikube_gpu/sd_launch_drawer.png" alt="Stable Diffusion 推論ジョブの Launch UI の画像" >}}
