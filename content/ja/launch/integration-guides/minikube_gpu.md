---
title: Minikube でシングルノードの GPU クラスターを立ち上げる
menu:
  launch:
    identifier: ja-launch-integration-guides-minikube_gpu
    parent: launch-integration-guides
url: tutorials/minikube_gpu
---

Minikube クラスターで W&B Launch をセットアップし、GPU ワークロードをスケジュール・実行できるようにしましょう。

{{% alert %}}
このチュートリアルは、複数 GPU を搭載したマシンに直接アクセスできるユーザー向けです。クラウドマシンをレンタルしているユーザー向けではありません。

クラウドマシン上で minikube クラスターをセットアップしたい場合は、クラウドプロバイダーの GPU サポート付き Kubernetes クラスター作成ツールを使うことを W&B は推奨します。AWS, GCP, Azure, Coreweave など、主要なクラウドプロバイダーが GPU 対応 Kubernetes クラスター作成ツールを提供しています。

また、1 台のマシンで GPU スケジューリング用に minikube クラスターをセットアップしたい場合は [Launch Docker キュー]({{< relref path="/launch/set-up-launch/setup-launch-docker" lang="ja" >}}) の利用をおすすめします。チュートリアルを参考として読むことも可能ですが、GPU スケジューリングの恩恵はあまりありません。
{{% /alert %}}

## 背景

[NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) の登場で、Docker 上で GPU 対応ワークフローが簡単に実行できるようになりました。ただし、GPU をボリューム単位でスケジューリングするネイティブサポートはなく、`docker run` コマンドで GPU を指定する場合 ID を指定するか、全ての GPU をリクエストするしかなく、多くの分散型 GPU ワークロードには向きません。Kubernetes ならボリューム単位のスケジューリングが可能ですが、ローカル Kubernetes クラスターでの GPU スケジューリング環境構築は手間と時間がかかります。ですが最近、シングルノード Kubernetes クラスター構築の人気ツールである Minikube が[GPU スケジューリングサポート](https://minikube.sigs.k8s.io/docs/tutorials/nvidia/)をリリースしました。本チュートリアルでは、複数 GPU マシン上に Minikube クラスターを構築し、W&B Launch を使って安定拡散（stable diffusion）推論ジョブを同時にクラスターへ投げていきます。

## 前提条件

始める前に、下記が必要です。

1. W&B アカウント
2. 下記がインストールされ、稼働中の Linux マシン
   1. Docker ランタイム
   2. 利用する GPU 用のドライバ
   3. Nvidia container toolkit

{{% alert %}}
このチュートリアルの検証には、Google Cloud Compute Engine の `n1-standard-16` インスタンス（NVIDIA Tesla T4 GPU 4台接続）を利用しました。
{{% /alert %}}

## ランチジョブ用キューの作成

まず、Launch ジョブ用のキューを作成します。

1. [wandb.ai/launch](https://wandb.ai/launch) （またはプライベート W&B サーバーの場合は `<your-wandb-url>/launch`）へアクセスします。
2. 画面右上の青い **Create a queue** ボタンをクリックします。画面右側からキュー作成ドロワーがスライド表示されます。
3. Entity を選択し、名前を入力し、キューのタイプに **Kubernetes** を選びます。
4. ドロワー内の **Config** セクションで、[Kubernetes のジョブ仕様](https://kubernetes.io/docs/concepts/workloads/controllers/job/)を指定します。このキューから Launch される Run はすべてこの job spec で作成されます。必要に応じてカスタム化できます。本チュートリアルでは、下記サンプル config をキューに YAML または JSON で貼り付けてください。

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

キュー設定について詳しくは、[Kubernetes で Launch をセットアップ]({{< relref path="/launch/set-up-launch/setup-launch-kubernetes.md" lang="ja" >}})および[高度なキュー設定ガイド]({{< relref path="/launch/set-up-launch/setup-queue-advanced.md" lang="ja" >}})をご覧ください。

`${image_uri}` および `{{gpus}}` は、キュー設定で利用できる 2 種類の変数テンプレート例です。`${image_uri}` テンプレートは agent が Launch するジョブのイメージ URI に置き換えられます。`{{gpus}}` テンプレートはキュー UI/CLI/SDK からジョブ投稿時に上書き可能な変数テンプレートで、指定した値によってジョブの画像や GPU リソースの割当てなど該当フィールドが変更されます。

5. **Parse configuration** ボタンをクリックし、`gpus` テンプレート変数のカスタマイズを始めます。
6. **Type** を `Integer` にし、**Default**, **Min**, **Max** を任意の値に設定します。
変数の制約条件に外れた Run をキューに投稿しようとすると拒否されます。

{{< img src="/images/tutorials/minikube_gpu/create_queue.png" alt="Queue creation drawer" >}}

7. **Create queue** をクリックし、キューを作成します。作成したキューのページにリダイレクトされます。

次に、このキューからジョブを取得して実行するエージェントをセットアップします。

## Docker + NVIDIA Container Toolkit のセットアップ

既にお使いのマシンで Docker と Nvidia container toolkit のセットアップが終わっていれば、このセクションはスキップ可能です。

Docker コンテナエンジンのインストール方法については[Docker 公式ドキュメント](https://docs.docker.com/engine/install/)をご確認ください。

Docker インストール完了後、Nvidia container toolkit を[公式手順](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)に従ってインストールしてください。

コンテナランタイムが GPU へアクセスできるかどうかの確認には、次のコマンドをお使いください。

```bash
docker run --gpus all ubuntu nvidia-smi
```

すると、接続されている GPU の情報を示す `nvidia-smi` の出力が得られます。例えば、このチュートリアルの環境では下記のような出力が得られます。

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

Minikube の GPU サポートには `v1.32.0` 以上が必要です。インストール最新情報は[Minikube 公式ドキュメント](https://minikube.sigs.k8s.io/docs/start/)を参照してください。本チュートリアルでは下記コマンドで最新版をインストールしました。

```yaml
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

続いて、手元の GPU を使い Minikube クラスターを起動します。下記コマンドを実行してください。

```yaml
minikube start --gpus all
```

上記の出力で、クラスターが正常作成されたかどうか確認できます。

## Launch エージェントの起動

新しいクラスター用 launch agent は `wandb launch-agent` を直接起動するか、[W&B 管理の helm チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) でデプロイすることが出来ます。

このチュートリアルではホストマシン上から直接 agent を実行します。

{{% alert %}}
エージェントをコンテナ外で動かすことで、ローカル Docker ホストを使ってクラスター用イメージのビルドも可能です。
{{% /alert %}}

ローカルで agent を走らせるには、デフォルトの Kubernetes API context が Minikube クラスターを指していることを確認し、以下を実行してください。

```bash
pip install "wandb[launch]"
```

これで agent の依存関係がインストールされます。認証には `wandb login` または `WANDB_API_KEY` 環境変数の設定が必要です。

エージェントの起動は下記コマンドで行います。

```bash
wandb launch-agent -j <max-number-concurrent-jobs> -q <queue-name> -e <queue-entity>
```

ターミナル上で launch agent のポーリングメッセージが表示されるはずです。

これで launch agent が launch queue のポーリングを開始しました。ジョブをキューに追加すると、agent がそれを検出し、Minikube クラスターへスケジューリングします。

## ジョブを launch する

次に、エージェント宛てにジョブを送ってみましょう。W&B アカウントでログイン済みのターミナルから、シンプルな "hello world" ジョブを以下で Launch できます。

```yaml
wandb launch -d wandb/job_hello_world:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

ジョブやイメージは任意のものを試せますが、クラスターがそのイメージを pull できることを確認してください。追加案内については[Minikube ドキュメント](https://minikube.sigs.k8s.io/docs/handbook/registry/)をご覧ください。[公開ジョブの一つをテストとして利用]も可能です(https://wandb.ai/wandb/jobs/jobs?workspace=user-bcanfieldsherman)。

## （オプション）NFS によるモデル・データキャッシュ

ML ワークロードでは、複数のジョブが同じデータへアクセスしたい場合がよくあります。たとえば、大規模なデータセットやモデル重みなど毎回ダウンロードしたくない大容量資産の共有キャッシュを持ちたい場合です。Kubernetes は[永続ボリュームと永続ボリュームクレーム](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) をサポートしており、`volumeMounts` を用いて共有キャッシュへのファイルシステムアクセスも可能になります。

ここでは、ネットワークファイルシステム(NFS)サーバーをセットアップし、モデル重みの共有キャッシュとして使用できるようにします。まず最初に NFS をインストール・設定します。手順は OS により異なります。今回、Ubuntu の VM を使用したため `nfs-kernel-server` をインストールし、`/srv/nfs/kubedata` をエクスポートしました。

```bash
sudo apt-get install nfs-kernel-server
sudo mkdir -p /srv/nfs/kubedata
sudo chown nobody:nogroup /srv/nfs/kubedata
sudo sh -c 'echo "/srv/nfs/kubedata *(rw,sync,no_subtree_check,no_root_squash,no_all_squash,insecure)" >> /etc/exports'
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

サーバーのエクスポート先（ホスト内ファイルパス）・NFS サーバーのローカル IP アドレスは控えておいてください。次のステップで利用します。

続いて、この NFS 用の「永続ボリューム」と「永続ボリュームクレーム」を作成します。永続ボリュームはパラメータのカスタマイズが柔軟ですが、ここではシンプルな設定を使います。

以下の yaml を `nfs-persistent-volume.yaml` として保存し、希望の容量・申請値に編集してください。`PersistentVolume.spec.capcity.storage` で物理ボリューム容量上限、`PersistentVolumeClaim.spec.resources.requests.stroage` で各クレーム許容量を制限します。本例では両者を同じ値にしています。

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nfs-pv
spec:
  capacity:
    storage: 100Gi # 希望容量に変更してください
  accessModes:
    - ReadWriteMany
  nfs:
    server: <your-nfs-server-ip> # TODO: ここに記載
    path: '/srv/nfs/kubedata' # カスタムパスなら変更
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
      storage: 100Gi # 希望容量に変更してください
  storageClassName: ''
  volumeName: nfs-pv
```

下記でリソースをクラスターに作成します。

```yaml
kubectl apply -f nfs-persistent-volume.yaml
```

このキャッシュを Run で利用できるようにするため、launch queue 設定に `volumes` と `volumeMounts` を追記します。launch config 編集は、[wandb.ai/launch](https://wandb.ai/launch)（または wandb サーバーご利用の場合 `<your-wandb-url>/launch`）に戻り、該当キューのページで **Edit config** タブを選びます。元の設定を下記のように変更可能です。

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

この設定で NFS はコンテナ内の `/root/.cache` にマウントされます。コンテナが `root` 以外のユーザーで起動する場合はパス調整が必要です。Huggingface ライブラリや W&B Artifacts はデフォルトで `$HOME/.cache/` を用いるため、一度のダウンロードで済むようになります。

## stable diffusion で試す

新しいシステムを試すため、stable diffusion の推論パラメータをいじってみましょう。
デフォルトプロンプトと妥当なパラメータでシンプルな安定拡散推論ジョブを Launch するには:

```
wandb launch -d wandb/job_stable_diffusion_inference:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

このコマンドで `wandb/job_stable_diffusion_inference:main` のコンテナイメージがあなたのキューに送られます。
エージェントがジョブを受け取ってスケジューリングし、クラスター上で実行されます。
イメージの pull には回線状況によって時間がかかる場合があります。
進捗は [wandb.ai/launch](https://wandb.ai/launch)（または wandb サーバーの場合 \<your-wandb-url\>/launch） のキュー画面で確認できます。

Run が完了すると、指定したプロジェクトに job artifact が生成されるはずです。
プロジェクトの jobs ページ（`<project-url>/jobs`）で job artifact を探せます。デフォルト名は
`job-wandb_job_stable_diffusion_inference` ですが、jobページで名前横の鉛筆マークをクリックすることで自由に変更できます。

このジョブを使って、更に安定拡散の推論をクラスターで実行できます。
job ページの右上 **Launch** ボタンから新たな推論ジョブを設定し、キューへ投稿できます。ジョブ設定画面は前回 Run のパラメータで自動入力されていますが、値を自由にカスタマイズし **Overrides** セクションで変更できます。

{{< img src="/images/tutorials/minikube_gpu/sd_launch_drawer.png" alt="Launch UI" >}}