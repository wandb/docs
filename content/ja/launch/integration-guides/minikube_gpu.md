---
title: Spin up a single node GPU cluster with Minikube
menu:
  launch:
    identifier: ja-launch-integration-guides-minikube_gpu
    parent: launch-integration-guides
url: tutorials/minikube_gpu
---

W&B Launch を Minikube クラスターでセットアップし、GPU ワークロードをスケジュールして実行する

{{% alert %}}
このチュートリアルは、複数の GPU を搭載したマシンに直接アクセスできるユーザーを案内することを目的としています。このチュートリアルは、クラウドマシンをレンタルしているユーザーを対象としていません。

クラウドマシンで Minikube クラスターをセットアップしたい場合、W&B はクラウドプロバイダを使用した GPU サポートのある Kubernetes クラスターを作成することをお勧めします。たとえば、AWS、GCP、Azure、Coreweave、その他のクラウドプロバイダーは、GPU サポートを備えた Kubernetes クラスターを作成するためのツールを提供しています。

マシンに単一の GPU が搭載されている場合、GPU のスケジューリングのために Minikube クラスターをセットアップしたい場合は、[Launch Docker キュー]({{< relref path="/launch/set-up-launch/setup-launch-docker/" lang="ja" >}}) を使用することをお勧めします。楽しみのためにチュートリアルを続けることはできますが、GPU スケジューリングはあまり役に立たないでしょう。
{{% /alert %}}

## 背景

[Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) により、Docker で GPU 対応のワークフローを実行することが容易になりました。唯一の制限は、ボリュームによる GPU スケジューリングのネイティブサポートがないことです。`docker run` コマンドで GPU を使用したい場合、特定の GPU を ID で要求するか、すべての GPU を使用する必要があり、多くの分散 GPU 対応ワークロードにとって不便です。Kubernetes はボリューム リクエストによるスケジューリングをサポートしていますが、GPU スケジューリングを備えたローカル Kubernetes クラスターのセットアップには多くの時間と労力が必要です。しかし、最近では異なります。単一ノード Kubernetes クラスターを実行する最も人気のあるツールの1つであるMinikubeが、最近 [GPU スケジューリングのサポートをリリースしました](https://minikube.sigs.k8s.io/docs/tutorials/nvidia/) 🎉 このチュートリアルでは、複数GPUを搭載したマシンにMinikubeクラスターを作成し、W&B Launchを使用してクラスターに同時に安定したディフュージョン推論ジョブを起動します 🚀

## 前提条件

始める前に、次のものが必要です:

1. W&B アカウント
2. 次のものがインストールされ、実行されている Linux マシン
   1. Docker ランタイム
   2. 使用したい任意の GPU のドライバー
   3. Nvidia container toolkit

{{% alert %}}
このチュートリアルのテストと作成には、4 台の NVIDIA Tesla T4 GPU が接続された `n1-standard-16` Google Cloud Compute Engine インスタンスを使用しました。
{{% /alert %}}

## Launch ジョブのためのキューを作成 

最初に、launch ジョブのための launch キューを作成します。

1. [wandb.ai/launch](https://wandb.ai/launch) (プライベート W&B サーバーを使用している場合は `<your-wandb-url>/launch`) へ移動します。
2. 画面の右上隅にある青色の **Create a queue** ボタンをクリックします。キュー作成用の引き出しメニューが画面の右側からスライドします。
3. エンティティを選択し、名前を入力し、キューのタイプとして **Kubernetes** を選択します。
4. 引き出しの **Config** セクションでは、launch キューのために [Kubernetes ジョブ仕様](https://kubernetes.io/docs/concepts/workloads/controllers/job/) を入力します。このキューから起動された run はこのジョブ仕様を使用して作成されるため、ジョブをカスタマイズするために必要に応じてこの設定を変更できます。このチュートリアルのために、以下のサンプル設定を YAML または JSON としてキュー設定にコピー＆ペーストできます:

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

キュー設定の詳細については、[Set up Launch on Kubernetes]({{< relref path="../set-up-launch/setup-launch-kubernetes.md" lang="ja" >}}) および [Advanced queue setup guide]({{< relref path="../set-up-launch/setup-queue-advanced.md" lang="ja" >}}) を参照してください。

`${image_uri}` と `{{gpus}}` の文字列は、キュー設定で使用できる2種類の変数テンプレートの例です。`${image_uri}` テンプレートは、launchするジョブのイメージ URI でエージェントによって置き換えられます。`{{gpus}}` テンプレートは、ジョブを提出するときに launchのUI、CLI、またはSDKからオーバーライドできるテンプレート変数を作成するために使用されます。これらの値は、ジョブのイメージおよび GPU リソースを制御する正しいフィールドを変更するためにジョブ仕様に配置されます。

5. **Parse configuration** ボタンをクリックして、`gpus` テンプレート変数をカスタマイズ開始します。
6. **Type** を `Integer` に設定し、**Default**, **Min**, **Max** を選択した値に設定します。
テンプレート変数の制約に違反する run をこのキューに送信しようとした場合、その run は拒否されます。

{{< img src="/images/tutorials/minikube_gpu/create_queue.png" alt="Image of queue creation drawer with gpus template variable" >}}

7. **Create queue** をクリックしてキューを作成します。新しいキューのキュー ページにリダイレクトされます。

次のセクションでは、作成したキューからジョブをプルして実行できるエージェントをセットアップします。

## Docker + NVIDIA CTK のセットアップ

既にマシンに Docker と Nvidia container toolkit がセットアップされている場合、このセクションをスキップできます。

システムに Docker コンテナエンジンをセットアップする手順については、[Docker のドキュメント](https://docs.docker.com/engine/install/) を参照してください。

Docker がインストールされたら、Nvidia container toolkit を [Nvidia のドキュメントの指示に従って](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) インストールします。

コンテナランタイムが GPU にアクセスできることを確認するには、次のコマンドを実行します:

```bash
docker run --gpus all ubuntu nvidia-smi
```

このコマンドにより、マシンに接続された GPU を示す `nvidia-smi` の出力が表示されます。たとえば、私たちのセットアップでは、出力は次のようになります:

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

Minikube の GPU サポートには、バージョン `v1.32.0` 以上が必要です。最新のインストール手順については、[Minikube のインストールドキュメント](https://minikube.sigs.k8s.io/docs/start/) を参照してください。このチュートリアルでは、次のコマンドを使用して最新の Minikube リリースをインストールしました：

```yaml
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

次のステップは、GPU を使用して minikube クラスターを開始することです。マシン上で以下を実行します：

```yaml
minikube start --gpus all
```

上記のコマンドの出力により、クラスターが正常に作成されたかどうかが示されます。

## Launch エージェントの開始

新しいクラスターの launch エージェントは、`wandb launch-agent` を直接起動するか、W&B の管理する [helm チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使用して launch エージェントを展開するかのいずれかで開始できます。

このチュートリアルでは、エージェントをホストマシン上で直接実行します。

{{% alert %}}
コンテナの外でエージェントを実行すると、クラスタが実行するためのイメージを構築するためにローカルの Docker ホストを使用できることも意味します。
{{% /alert %}}

エージェントをローカルで実行するには、デフォルトの Kubernetes API コンテキストが Minikube クラスターを指していることを確認します。次に、次のコマンドを実行してエージェントの依存関係をインストールします：

```bash
pip install "wandb[launch]"
```

エージェントの認証情報を設定するには、`wandb login` を実行するか、`WANDB_API_KEY` 環境変数を設定します。

エージェントを開始するには、次のコマンドを実行します：

```bash
wandb launch-agent -j <max-number-concurrent-jobs> -q <queue-name> -e <queue-entity>
```

あなたのターミナルには、launch エージェントが polling メッセージを出力し始めるはずです。

おめでとうございます！あなたは launch エージェントが launch キューをポーリングするようになりました。キューにジョブが追加されると、エージェントがそのジョブを取得し、Minikube クラスター上で実行するためのスケジュールを設定します。

## ジョブを起動

エージェントにジョブを送信してみましょう。W&B アカウントにログインしたターミナルからシンプルな "hello world" をローンチできます：

```yaml
wandb launch -d wandb/job_hello_world:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

任意のジョブやイメージでテストすることができますが、必ずクラスタがイメージをプルできることを確認してください。詳細な手引きについては、[Minikube のドキュメント](https://minikube.sigs.k8s.io/docs/handbook/registry/) を参照してください。また、[我々のパブリックジョブのいずれかを使用してテストすることもできます](https://wandb.ai/wandb/jobs/jobs?workspace=user-bcanfieldsherman)。

## (オプション) モデルとデータの NFS を使ったキャッシュ

ML ワークロードでは、複数のジョブが同じデータにアクセスすることを望むことがよくあります。たたえば、大規模なデータセットやモデルの重みを繰り返しダウンロードするのを避けるために、共用のキャッシュを持ちたいとします。Kubernetes は[永続的なボリュームと永続的なボリュームクレーム](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)を通じてこれをサポートしています。永続的なボリュームは、Kubernetes ワークロードで `volumeMounts` を作成するために使用でき、共有キャッシュへの直接的なファイルシステムアクセスを提供します。

このステップでは、モデルウェイトの共有キャッシュとして使用できるネットワークファイルシステム (NFS) サーバーを設定します。最初のステップは、NFS をインストールし構成することです。プロセスはオペレーティングシステムによって異なります。我々の VM は Ubuntu を実行しているので、nfs-kernel-server をインストールし、`/srv/nfs/kubedata` にエクスポートを設定しました：

```bash
sudo apt-get install nfs-kernel-server
sudo mkdir -p /srv/nfs/kubedata
sudo chown nobody:nogroup /srv/nfs/kubedata
sudo sh -c 'echo "/srv/nfs/kubedata *(rw,sync,no_subtree_check,no_root_squash,no_all_squash,insecure)" >> /etc/exports'
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

エクスポートの場所と NFS サーバーのローカル IP アドレスをホストのファイルシステムに記録しておいてください。次のステップでこの情報が必要になります。

次に、この NFS のために永続的なボリュームと永続的なボリュームクレームを作成する必要があります。永続的なボリュームは非常にカスタマイズ可能ですが、シンプルさを考慮して、ここでは簡単な構成を使用します。

以下の yaml を `nfs-persistent-volume.yaml` という名前のファイルにコピーし、希望のボリューム容量とクレームリクエストを入力してください。`PersistentVolume.spec.capcity.storage` フィールドは、基礎となるボリュームの最大サイズを制御します。`PersistentVolumeClaim.spec.resources.requests.stroage` は特定のクレームに割り当てられるボリューム容量を制限するために使用できます。ケースごとに同じ値を使用するのが理にかなっています。

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
    server: <your-nfs-server-ip> # TODO: ここを記入してください。
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
      storage: 100Gi # 希望の容量を設定してください。
  storageClassName: ''
  volumeName: nfs-pv
```

次のコマンドを実行してクラスタにリソースを作成します：

```yaml
kubectl apply -f nfs-persistent-volume.yaml
```

弊社の `runs` でこのキャッシュを利用するには、`volumes` と `volumeMounts` を launch キューの設定に追加する必要があります。設定を編集するには、[wandb.ai/launch](http://wandb.ai/launch) (または wandb サーバーの `\<your-wandb-url\>/launch` ) に戻り、キューを探し、キューのページを開き、**Edit config** タブをクリックしてください。元の設定を次のように修正できます：

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

現在、NFS は ジョブを実行するコンテナ内の `/root/.cache` にマウントされます。コンテナが `root` 以外のユーザーとして実行される場合、マウントパスの調整が必要です。Huggingface のライブラリと W&B Artifacts はいずれもデフォルトで `$HOME/.cache/` を利用するため、ダウンロードは1度だけで済むはずです。

## ステーブルディフュージョンで遊ぶ

この新しいシステムをテストするために、ステーブルディフュージョンの推論パラメータを実験してみましょう。
デフォルトのプロンプトと適切なパラメータを使用してシンプルなステーブルディフュージョン推論ジョブを実行するには、次のコマンドを実行します：

```
wandb launch -d wandb/job_stable_diffusion_inference:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

上記のコマンドは、コンテナイメージ `wandb/job_stable_diffusion_inference:main` をキューに送信します。
エージェントがジョブを取得し、クラスター上で実行するためにスケジュールを設定すると、接続に応じてイメージのプルに時間がかかる場合があります。
[wandb.ai/launch](http://wandb.ai/launch) (または wandb サーバーの `<your-wandb-url>/launch` ) のキュー ページでジョブのステータスをフォローできます。

run が終了すると、指定したプロジェクトにジョブのアーティファクトが作成されます。
プロジェクトのジョブ ページ (`<project-url>/jobs`) でジョブアーティファクトを見つけることができます。デフォルトの名前は `job-wandb_job_stable_diffusion_inference` ですが、ジョブ ページでジョブ名の横にある鉛筆アイコンをクリックすることで好きな名前に変更できます。

このジョブを使用して、クラスター上でさらにステーブルディフュージョンの推論を実行できます。
ジョブ ページから、右上隅の **Launch** ボタンをクリックして、新しい推論ジョブを設定し、キューに送信します。ジョブ設定ページは元の run からのパラメータでプリセットされますが、launch 引き出しの **Overrides** セクションで値を変更することで任意に変更できます。

{{< img src="/images/tutorials/minikube_gpu/sd_launch_drawer.png" alt="Image of launch UI for stable diffusion inference job" >}}
```