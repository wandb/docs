---
title: Minikube で単一ノードの GPU クラスターを起動する
menu:
  launch:
    identifier: ja-launch-integration-guides-minikube_gpu
    parent: launch-integration-guides
url: tutorials/minikube_gpu
---

GPU ワークロードをスケジュールして実行できる Minikube クラスター上で W&B Launch をセットアップします。

{{% alert %}}
このチュートリアルは、複数の GPU を搭載したマシンに直接 アクセス できる ユーザー 向けです。クラウド マシンをレンタルしている ユーザー は対象外です。

クラウド マシン上で minikube クラスターをセットアップしたい場合は、クラウド プロバイダを使って GPU 対応の Kubernetes クラスターを作成することを W&B は推奨します。たとえば、AWS、GCP、Azure、CoreWeave などのクラウド プロバイダには、GPU 対応の Kubernetes クラスターを作成するための ツール があります。

単一 GPU を搭載したマシンで GPU のスケジューリング用に minikube クラスターをセットアップしたい場合は、[Launch Docker queue]({{< relref path="/launch/set-up-launch/setup-launch-docker" lang="ja" >}}) の使用を W&B は推奨します。このチュートリアルに沿って試してみることはできますが、GPU のスケジューリングはあまり有用ではありません。
{{% /alert %}}

## 背景

[NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) により、Docker 上で GPU 対応のワークフローを簡単に実行できるようになりました。制約の 1 つは、GPU の割り当て数に基づいたネイティブなスケジューリング サポートがないことです。`docker run` コマンドで GPU を使うには、ID で特定の GPU を指定するか、存在するすべての GPU を要求する必要があり、多くの分散 GPU ワークロードが実用的でなくなります。Kubernetes は要求した数量に基づくスケジューリングをサポートしていますが、GPU スケジューリング対応のローカル Kubernetes クラスターを構築するには時間と手間がかかりました。Minikube は、単一ノードの Kubernetes クラスターを動かすための最も一般的な ツール の 1 つですが、最近 [GPU スケジューリングのサポート](https://minikube.sigs.k8s.io/docs/tutorials/nvidia/)をリリースしました。このチュートリアルでは、マルチ GPU マシン上に Minikube クラスターを作成し、W&B Launch を使ってクラスターに安定拡散 (Stable Diffusion) の推論ジョブを同時に投入します。

## 前提条件

開始する前に、次のものが必要です。

1. W&B アカウント。
2. 次のソフトウェアがインストール・稼働している Linux マシン:
   1. Docker ランタイム
   2. 使用する GPU のドライバ
   3. NVIDIA Container Toolkit

{{% alert %}}
このチュートリアルの作成と検証には、4 基の NVIDIA Tesla T4 GPU が接続された `n1-standard-16` の Google Cloud Compute Engine インスタンスを使用しました。
{{% /alert %}}

## Launch ジョブ用のキューを作成

まず、Launch ジョブ用のキューを作成します。

1. [wandb.ai/launch](https://wandb.ai/launch) に移動します（プライベートな W&B サーバー を使っている場合は `<your-wandb-url>/launch`）。
2. 画面右上の青い **Create a queue** ボタンをクリックします。画面右側からキュー作成のドロワーがスライド表示されます。
3. Entity を選択し、名前を入力して、タイプとして **Kubernetes** を選びます。
4. ドロワーの **Config** セクションに、キュー用の [Kubernetes job specification](https://kubernetes.io/docs/concepts/workloads/controllers/job/) を入力します。このキューから Launch された Run は、このジョブ仕様で作成されます。必要に応じてこの 設定 を編集し、ジョブをカスタマイズできます。このチュートリアルでは、以下のサンプルを YAML または JSON としてそのままキュー設定に貼り付けてください。

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

キューの 設定 についての詳細は、[Set up Launch on Kubernetes]({{< relref path="/launch/set-up-launch/setup-launch-kubernetes.md" lang="ja" >}}) と [Advanced queue setup guide]({{< relref path="/launch/set-up-launch/setup-queue-advanced.md" lang="ja" >}}) を参照してください。

`${image_uri}` と `{{gpus}}` は、キュー 設定 で使える 2 種類の変数テンプレートの例です。`${image_uri}` テンプレートは、エージェント によって Launch するジョブのイメージ URI に置き換えられます。`{{gpus}}` テンプレートはテンプレート変数を作成し、ジョブを送信する際に Launch の UI、CLI、または SDK から上書きできます。これらの 値 は、ジョブのイメージと GPU リソースを制御する適切なフィールドが変更されるよう、ジョブ仕様の中に配置されます。

5. **Parse configuration** ボタンをクリックして、`gpus` テンプレート変数のカスタマイズを開始します。
6. **Type** を `Integer` に設定し、**Default**、**Min**、**Max** を任意の 値 に設定します。このテンプレート変数の制約に違反する Run をこのキューに送信しようとすると、却下されます。

{{< img src="/images/tutorials/minikube_gpu/create_queue.png" alt="キュー作成ドロワー" >}}

7. **Create queue** をクリックしてキューを作成します。新しいキューのページにリダイレクトされます。

次のセクションでは、作成したキューからジョブを取得して実行できる エージェント をセットアップします。

## Docker と NVIDIA CTK のセットアップ

すでに Docker と NVIDIA Container Toolkit をマシンにセットアップ済みの場合は、このセクションをスキップしてください。

システムへの Docker コンテナ エンジンのセットアップ手順については、[Docker のドキュメント](https://docs.docker.com/engine/install/)を参照してください。

Docker をインストールしたら、NVIDIA のドキュメントの手順に従って [NVIDIA Container Toolkit をインストール](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) します。

コンテナ ランタイムが GPU に アクセス できることを確認するには、次を実行します:

```bash
docker run --gpus all ubuntu nvidia-smi
```

接続された GPU を示す `nvidia-smi` の出力が表示されるはずです。たとえば、私たちの環境では次のような出力でした:

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

Minikube の GPU サポートには `v1.32.0` 以降が必要です。最新のインストール手順は [Minikube のインストールドキュメント](https://minikube.sigs.k8s.io/docs/start/)を参照してください。このチュートリアルでは、次のコマンドで最新の Minikube リリースをインストールしました:

```yaml
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

次に、GPU を使って minikube クラスターを起動します。マシン上で次を実行します:

```yaml
minikube start --gpus all
```

上記コマンドの出力に、クラスターが正常に作成されたかどうかが表示されます。

## Launch エージェントを起動

新しいクラスター向けの Launch エージェントは、`wandb launch-agent` を直接実行するか、W&B が管理する [Helm チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使ってデプロイできます。

このチュートリアルでは、エージェントをホスト マシン上で直接実行します。

{{% alert %}}
コンテナの外で エージェント を動かすことで、クラスターで実行するイメージをローカルの Docker ホストでビルドできます。
{{% /alert %}}

ローカルで エージェント を実行するには、デフォルトの Kubernetes API コンテキストが Minikube クラスターを指していることを確認し、次を実行します:

```bash
pip install "wandb[launch]"
```

これで エージェント の依存関係がインストールされます。エージェントの認証をセットアップするには、`wandb login` を実行するか、`WANDB_API_KEY` 環境 変数 を設定してください。

エージェントを起動するには、次のコマンドを実行します:

```bash
wandb launch-agent -j <max-number-concurrent-jobs> -q <queue-name> -e <queue-entity>
```

ターミナル 上に、Launch エージェントがポーリング メッセージを出力し始めるのが確認できるはずです。

これで、Launch キューをポーリングするエージェントが起動しました。キューにジョブが追加されると、エージェントがそれを取得して Minikube クラスター上で実行するようにスケジュールします。

## ジョブを起動する

エージェントにジョブを送ってみましょう。W&B アカウントでログイン済みの ターミナル から、次のコマンドでシンプルな "hello world" を Launch できます:

```yaml
wandb launch -d wandb/job_hello_world:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

任意のジョブやイメージで試せますが、クラスターがそのイメージを pull できることを確認してください。詳細は [Minikube のドキュメント](https://minikube.sigs.k8s.io/docs/handbook/registry/) を参照してください。[公開ジョブのいずれかでテスト](https://wandb.ai/wandb/jobs/jobs?workspace=user-bcanfieldsherman) することもできます。

## （オプション）NFS を使った モデル と データ のキャッシュ

ML ワークロードでは、複数のジョブから同じ データ に アクセス したくなることがよくあります。たとえば、データセット や モデル 重みのような大きなアセットを何度もダウンロードしないよう、共有キャッシュを用意したい場合です。Kubernetes では、[Persistent Volume と Persistent Volume Claim](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) を通じてこれをサポートしています。Persistent Volume は Kubernetes のワークロードに `volumeMounts` を作成して、共有キャッシュへのファイルシステム アクセス を直接提供できます。

このステップでは、モデルの重みの共有キャッシュとして使える NFS (Network File System) サーバー をセットアップします。まず NFS のインストールと設定を行います。手順は OS によって異なります。今回の VM は Ubuntu で動いているため、nfs-kernel-server をインストールし、`/srv/nfs/kubedata` にエクスポートを設定しました:

```bash
sudo apt-get install nfs-kernel-server
sudo mkdir -p /srv/nfs/kubedata
sudo chown nobody:nogroup /srv/nfs/kubedata
sudo sh -c 'echo "/srv/nfs/kubedata *(rw,sync,no_subtree_check,no_root_squash,no_all_squash,insecure)" >> /etc/exports'
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

ホストのファイルシステム上でのエクスポートの場所と、NFS サーバー のローカル IP アドレスをメモしておいてください。次のステップで必要になります。

次に、この NFS 用の Persistent Volume と Persistent Volume Claim を作成します。Persistent Volume は高度にカスタマイズ可能ですが、ここではシンプルな 設定 を使います。

以下の yaml を `nfs-persistent-volume.yaml` というファイルにコピーし、希望するボリューム容量とクレームの要求を入力してください。`PersistentVolume.spec.capcity.storage` フィールドは基盤となるボリュームの最大サイズを制御します。`PersistentVolumeClaim.spec.resources.requests.stroage` は特定のクレームに割り当てるボリューム容量を制限するために使えます。今回の ユースケース では、両方に同じ 値 を使うのが合理的です。

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nfs-pv
spec:
  capacity:
    storage: 100Gi # 希望の容量に設定してください。
  accessModes:
    - ReadWriteMany
  nfs:
    server: <your-nfs-server-ip> # TODO: ここに値を入力してください。
    path: '/srv/nfs/kubedata' # または任意のパス
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
      storage: 100Gi # 希望の容量に設定してください。
  storageClassName: ''
  volumeName: nfs-pv
```

次のコマンドでクラスターにリソースを作成します:

```yaml
kubectl apply -f nfs-persistent-volume.yaml
```

Run がこのキャッシュを使えるようにするには、Launch キューの 設定 に `volumes` と `volumeMounts` を追加する必要があります。設定を編集するには、[wandb.ai/launch](https://wandb.ai/launch)（wandb サーバー ユーザーは `<your-wandb-url>/launch`）に戻り、キューを見つけてキューのページに入り、**Edit config** タブをクリックします。元の 設定 を次のように変更できます:

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

これで、ジョブを実行するコンテナ内の `/root/.cache` に NFS がマウントされます。コンテナが `root` 以外のユーザーで動作する場合は、マウント パスを調整してください。Hugging Face のライブラリと W&B Artifacts はどちらもデフォルトで `$HOME/.cache/` を使うため、ダウンロードは一度だけで済むはずです。

## Stable Diffusion で遊んでみる

新しいシステムを試すために、Stable Diffusion の推論パラメータをいじってみましょう。デフォルトのプロンプトと妥当なパラメータでシンプルな推論ジョブを実行するには、次を実行します:

```
wandb launch -d wandb/job_stable_diffusion_inference:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

上のコマンドは、コンテナ イメージ `wandb/job_stable_diffusion_inference:main` をキューに送信します。エージェントがジョブを取得してクラスター上での実行をスケジュールすると、回線状況によってはイメージの pull に時間がかかる場合があります。ジョブのステータスは [wandb.ai/launch](https://wandb.ai/launch) のキュー ページ（wandb サーバー ユーザーは \<your-wandb-url\>/launch）で確認できます。

Run が完了すると、指定した Project にジョブ Artifact が作成されているはずです。Project のジョブ ページ（`<project-url>/jobs`）でジョブ Artifact を確認できます。デフォルト名は `job-wandb_job_stable_diffusion_inference` ですが、ジョブのページでジョブ名の横にある鉛筆アイコンをクリックして任意の名前に変更できます。

このジョブを使って、クラスター上で Stable Diffusion の推論をさらに実行できます。ジョブ ページの右上にある **Launch** ボタンをクリックして、新しい推論ジョブを 設定 し、キューに送信します。ジョブの 設定 ページには元の Run のパラメータがあらかじめ入力されていますが、**Overrides** セクションで 値 を変更すれば、好きなように調整できます。

{{< img src="/images/tutorials/minikube_gpu/sd_launch_drawer.png" alt="Launch の UI" >}}