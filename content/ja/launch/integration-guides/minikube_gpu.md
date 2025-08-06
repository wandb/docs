---
title: Minikube でシングルノードの GPU クラスターを立ち上げる
menu:
  launch:
    identifier: minikube_gpu
    parent: launch-integration-guides
url: tutorials/minikube_gpu
---

Minikube クラスター上で、GPU ワークロードをスケジューリングして実行できるように W&B Launch をセットアップしましょう。

{{% alert %}}
このチュートリアルは、複数の GPU を搭載したマシンへ直接アクセスできるユーザー向けのガイドです。クラウドマシンをレンタルしているユーザー向けではありません。

クラウドマシン上で minikube クラスターをセットアップしたい場合は、GPU サポート付きの Kubernetes クラスターをクラウドプロバイダー（例：AWS、GCP、Azure、Coreweave 他）のツールを使って作成することを推奨します。

単一 GPU のマシンで minikube クラスターをセットアップしたい場合は、[Launch Docker キュー]({{< relref "/launch/set-up-launch/setup-launch-docker" >}}) の使用をおすすめします。このチュートリアルも試すことはできますが、GPU スケジューリングの恩恵はあまり受けられません。
{{% /alert %}}

## 背景

[NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) により、Docker で GPU 対応のワークフローを手軽に実行できるようになりました。ただし 1 つの制限として、GPU をボリューム単位でネイティブにスケジューリングする機能がありません。たとえば `docker run` コマンドで GPU を利用したい場合、ID で特定 GPU を指定するか、搭載されているすべての GPU を指定する必要があり、これでは分散 GPU ワークロードの運用が現実的でなくなります。Kubernetes ではボリュームリクエストでの GPU スケジューリングをサポートしていますが、ローカルに Kubernetes クラスターを GPU 対応で構築するのは、これまでかなり手間と時間がかかっていました。minikube はシングルノード Kubernetes クラスターを動かすのに人気のツールですが、最近 [GPU スケジューリングのサポート](https://minikube.sigs.k8s.io/docs/tutorials/nvidia/) が追加されました。このチュートリアルでは、マルチ GPU マシンに Minikube クラスターを構築し、W&B Launch を使って安定拡散（stable diffusion）の推論ジョブを同時に複数キューに投入してみます。

## 事前準備

始める前に、次のものが必要です。

1. W&B アカウント
2. 以下がインストールされ、動作している Linux マシン：
   1. Docker ランタイム
   2. 利用したい GPU 用のドライバ（複数ある場合はすべて）
   3. Nvidia container toolkit

{{% alert %}}
このチュートリアルの作成・動作検証には、Google Cloud Compute Engine の `n1-standard-16` インスタンス（NVIDIA Tesla T4 GPU 4基接続）を使用しました。
{{% /alert %}}

## Launch ジョブ用のキューを作成する

まず、Launch ジョブ用のキューを新規作成します。

1. [wandb.ai/launch](https://wandb.ai/launch) （またはプライベート W&B サーバー利用時は `<your-wandb-url>/launch`）へ移動します。
2. 画面右上の青い **Create a queue** ボタンをクリックします。右側にキュー作成用のドロワーがスライド表示されます。
3. entity を選択し、名前を入力、**Kubernetes** をキュータイプとして選択します。
4. ドロワーの **Config** セクションには [Kubernetes ジョブ定義](https://kubernetes.io/docs/concepts/workloads/controllers/job/) を入力します。このキューから起動される全ての run はこの job spec で作成されます。ジョブをカスタマイズしたい場合は設定を修正してください。今回は下記のサンプル構成（YAML/JSON）をコピーして config 部に貼り付けてください。

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

キューの設定に関する詳細は [Set up Launch on Kubernetes]({{< relref "/launch/set-up-launch/setup-launch-kubernetes.md" >}}) および [Advanced queue setup guide]({{< relref "/launch/set-up-launch/setup-queue-advanced.md" >}}) をご覧ください。

`${image_uri}` と `{{gpus}}` 部分は、キュー設定用の 2 種類のテンプレート変数の例です。`${image_uri}` は、ジョブに割り当てるイメージ URI でエージェントにより自動で置換されます。`{{gpus}}` はテンプレート変数として使われ、Launch UI、CLI、SDK からジョブ送信時に上書き指定できます。これらの値は job spec の該当フィールドに反映され、イメージや GPU リソース数の制御に使われます。

5. **Parse configuration** ボタンを押して `gpus` テンプレート変数の設定に進みます。
6. **Type** を `Integer` にし、**Default**、**Min**、**Max** を希望値に設定します。
テンプレート変数の制約条件を満たさない run をキューに送信しようとすると拒否されます。

{{< img src="/images/tutorials/minikube_gpu/create_queue.png" alt="Queue creation drawer" >}}

7. **Create queue** をクリックしてキューを作成します。作成後はキューのページにリダイレクトされます。

次のセクションでは、作成したキューからジョブを拾って実行できるエージェントをセットアップします。

## Docker + NVIDIA CTK のセットアップ

お使いのマシンで Docker と Nvidia container toolkit のセットアップが既に済んでいれば、このセクションはスキップできます。

Docker のインストールについては [Docker公式ドキュメント](https://docs.docker.com/engine/install/) をご参照ください。

Docker のインストール後、Nvidia container toolkit のインストールは [Nvidia ドキュメントの手順](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) に従ってください。

コンテナランタイムが GPU へアクセスできるか確認するには、以下を実行します：

```bash
docker run --gpus all ubuntu nvidia-smi
```

実行結果として、マシンに接続されている GPU の情報が `nvidia-smi` の出力として表示されます。たとえば、検証環境では以下のような出力になります：

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

Minikube の GPU サポートは `v1.32.0` 以降が必要です。最新版のインストール方法は [Minikube 公式ガイド](https://minikube.sigs.k8s.io/docs/start/) を参照してください。本チュートリアルでは下記コマンドで最新版をインストールしました：

```yaml
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

続いて、GPU を使って minikube クラスターを起動します。マシン上で以下を実行してください：

```yaml
minikube start --gpus all
```

上記コマンドの出力に、クラスターが無事作成されたことが表示されます。

## launch agent の起動

新しく作成したクラスター用の launch agent は、`wandb launch-agent` コマンドで直に起動するか、[W&B 管理の helm チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使ってデプロイできます。

このチュートリアルではホストマシン上で agent を直接実行します。

{{% alert %}}
エージェントをコンテナの外で実行すると、ローカルの Docker ホストを利用してクラスター用イメージのビルドも可能です。
{{% /alert %}}

ローカルで agent を動かすには、デフォルトの Kubernetes API context が Minikube クラスターを指しているか確認し、次を実行してください：

```bash
pip install "wandb[launch]"
```

エージェントの依存関係がインストールされます。認証のセットアップは `wandb login` の実行、または `WANDB_API_KEY` 環境変数の設定で行えます。

agent の起動は次のコマンドです：

```bash
wandb launch-agent -j <max-number-concurrent-jobs> -q <queue-name> -e <queue-entity>
```

ターミナル上で agent がポーリングメッセージを出力し始めるはずです。

これで launch agent が launch queue のポーリングを開始しました。キューへジョブが追加されると、エージェントがそれを拾って Minikube クラスターへスケジューリングします。

## ジョブを Launch する

エージェントへジョブを送ってみましょう。次のような「hello world」を W&B アカウントでログイン済みのターミナルから Launch できます：

```yaml
wandb launch -d wandb/job_hello_world:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

他のジョブやイメージも試せますが、クラスターがそれらのイメージをプルできる必要があります。詳細は [Minikube のドキュメント](https://minikube.sigs.k8s.io/docs/handbook/registry/) をご参照ください。[公開ジョブでもテストできます](https://wandb.ai/wandb/jobs/jobs?workspace=user-bcanfieldsherman)。

## （オプション）NFS でのモデル・データキャッシュ

ML ワークロードでは、複数ジョブが同じデータへアクセスしたいケースがあります。たとえば大きなデータセットやモデル重みの何度ものダウンロードを避け、一度ダウンロードしたら共通キャッシュとして使わせたい場合です。Kubernetes では [永続ボリュームと永続ボリュームクレーム](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) の仕組みを使って対応します。永続ボリュームを `volumeMounts` として Kubernetes ワークロードに指定すれば、ファイルシステム越しにデータ共有できます。

今回は、モデル重みキャッシュとして利用できるネットワークファイルシステム（NFS）サーバーをセットアップします。まず NFS を導入・設定します（OS ごとに異なります）。今回の仮想マシン（Ubuntu）では、`nfs-kernel-server` をインストールし `/srv/nfs/kubedata` をエクスポートしました：

```bash
sudo apt-get install nfs-kernel-server
sudo mkdir -p /srv/nfs/kubedata
sudo chown nobody:nogroup /srv/nfs/kubedata
sudo sh -c 'echo "/srv/nfs/kubedata *(rw,sync,no_subtree_check,no_root_squash,no_all_squash,insecure)" >> /etc/exports'
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

ホストのファイルシステムにおけるエクスポート先、および NFS サーバーのローカル IP アドレスを控えておきましょう。次のステップで必要です。

次に、この NFS 用の永続ボリュームおよびボリュームクレームを作成します。永続ボリュームは高度にカスタマイズ可能ですが、ここではシンプルな構成例とします。

下記 yaml を `nfs-persistent-volume.yaml` というファイル名で保存し、希望する容量等を書き換えます。`PersistentVolume.spec.capacity.storage` が最大容量、`PersistentVolumeClaim.spec.resources.requests.storage` で個々のクレーム割当量（今回は両方を同じ値にするのが適切です）。

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nfs-pv
spec:
  capacity:
    storage: 100Gi # 希望の容量を指定してください
  accessModes:
    - ReadWriteMany
  nfs:
    server: <your-nfs-server-ip> # サーバーのローカル IP を記入
    path: '/srv/nfs/kubedata' # カスタムパスも利用可
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
      storage: 100Gi # 希望の容量を指定してください
  storageClassName: ''
  volumeName: nfs-pv
```

クラスターへリソースを反映させるには：

```yaml
kubectl apply -f nfs-persistent-volume.yaml
```

run でこのキャッシュを活用するには、Launch キューの config に `volumes` と `volumeMounts` の追加が必要です。config 編集は [wandb.ai/launch](https://wandb.ai/launch)（または wandb server 利用時は `<your-wandb-url>/launch`）で該当キューを選択、「Edit config」タブに移動します。設定例をこのように変更します：

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

これで NFS が `/root/.cache` としてジョブ用コンテナ内にマウントされます。なお、コンテナを `root` 以外のユーザーで動かす場合は適宜パスを変更してください。Huggingface ライブラリや W&B Artifacts はデフォルトで `$HOME/.cache/` を利用するので、ダウンロードは 1 度だけで済むはずです。

## stable diffusion を試してみる

セットアップができたら、stable diffusion の推論パラメータで遊んでみましょう。
デフォルトプロンプトと無難なパラメータで、シンプルな推論ジョブを実行するには：

```
wandb launch -d wandb/job_stable_diffusion_inference:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

このコマンドは `wandb/job_stable_diffusion_inference:main` イメージをキューに送信します。
エージェントがジョブを拾いクラスターで実行を始めるまで、ネットワーク状況によってはイメージのプルに時間がかかることもあります。
ジョブのステータスはキュー画面の [wandb.ai/launch](https://wandb.ai/launch) （または wandb server 利用時は \<your-wandb-url\>/launch）で確認できます。

run が完了すると、指定したプロジェクトに job artifact が生成されます。
プロジェクトの job ページ（`<project-url>/jobs`）で確認できます。デフォルトの名前は `job-wandb_job_stable_diffusion_inference` ですが、job のページで鉛筆アイコンをクリックすると名称を自由に変更できます。

このジョブを使って、引き続きクラスター上で stable diffusion 推論を追加実行することもできます。
ジョブページ右上の **Launch** ボタンを押せば、新しい推論ジョブの設定画面が開きます。設定画面は元の run のパラメータが自動で入力済みですが、**Overrides** セクションで値を変更すればパラメータのカスタム実行も可能です。

{{< img src="/images/tutorials/minikube_gpu/sd_launch_drawer.png" alt="Launch UI" >}}