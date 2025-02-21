---
title: Use Prometheus monitoring
menu:
  default:
    identifier: ja-guides-hosting-monitoring-usage-prometheus-logging
    parent: monitoring-and-usage
weight: 2
---

[Prometheus](https://prometheus.io/docs/introduction/overview/) を W&B サーバーで使用します。Prometheus のインストールは、[kubernetes ClusterIP サービス](https://github.com/wandb/terraform-kubernetes-wandb/blob/main/main.tf#L225)として公開されています。

{{% alert color="secondary" %}}
Prometheus の監視は、[セルフ管理インスタンス]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})でのみ利用可能です。
{{% /alert %}}

以下の手順に従って、Prometheus のメトリクスエンドポイント (`/metrics`) に アクセス してください:

1. Kubernetes CLI ツールキットである [kubectl](https://kubernetes.io/docs/reference/kubectl/) を使用してクラスターに接続します。詳細は、kubernetes の [クラスターへのアクセス](https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster/) ドキュメントを参照してください。
2. 次のコマンドでクラスターの内部 アドレス を見つけます:

    ```bash
    kubectl describe svc prometheus
    ```

3. Kubernetes クラスターで実行中のコンテナ内でシェルセッションを開始し、エンドポイント`<internal address>/metrics`にアクセスします。このために、[`kubectl exec`](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands)を使用します。

    以下のコマンドをコピーし、ターミナルで実行して `<internal address>` をあなたの内部 アドレス に置き換えてください:

   ```bash
   kubectl exec <internal address>/metrics
   ```

テストポッドが起動し、ネットワーク内のすべてに アクセス するために exec することができます:

```bash
kubectl run -it testpod --image=alpine bin/ash --restart=Never --rm
```

そこから、ネットワークへの アクセス を内部のみに保持するか、kubernetes の nodeport サービスで自分自身を公開するかを選択できます。