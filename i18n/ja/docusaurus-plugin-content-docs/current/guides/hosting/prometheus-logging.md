---
description: How to configure the W&B Server installation
displayed_sidebar: ja
---
# Prometheusモニタリング

W&Bサーバーで[Prometheus](https://prometheus.io/docs/introduction/overview/)を使用します。Prometheusのインストールは、[kubernetes ClusterIPサービス](https://github.com/wandb/terraform-kubernetes-wandb/blob/main/main.tf#L225)として公開されています。

以下の手順に従って、Prometheusメトリクスのエンドポイント(`/metrics`)にアクセスしてください:

1. Kubernetes CLIツールキットの[kubectl](https://kubernetes.io/docs/reference/kubectl/)を使って、クラスターに接続します。詳細については、kubernetesの[クラスターへのアクセス](https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster/)ドキュメントを参照してください。
2. 以下のコマンドで、クラスターの内部アドレスを探し出します:

```bash
kubectl describe svc prometheus
```

3. [`kubectl exec`](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands) を使って、Kubernetesクラスター内で実行中のコンテナ内でシェルセッションを開始します。`<internal address>/metrics`のエンドポイントにアクセスしてください。

   以下のコマンドをコピーしてターミナルで実行し、`<internal address>`を内部アドレスに置き換えてください:

```bash
   kubectl exec <内部アドレス>/メトリクス
   ```

上記のコマンドは、ダミーポッドを起動し、ネットワーク内の何かにアクセスするためだけに実行できるようになります。

```bash
kubectl run -it testpod --image=alpine bin/ash --restart=Never --rm
```
ここから、ネットワーク内部へのアクセスを維持するか、kubernetesのnodeportサービスを使って自分で公開するかを選択できます。