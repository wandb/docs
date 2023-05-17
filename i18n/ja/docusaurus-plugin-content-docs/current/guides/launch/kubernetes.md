# Kubernetesで起動

このガイドでは、W&B Launchを使用して、kubernetes（k8s）クラスター上でMLワークロードを実行する方法を示します。

<!-- TODO: どのメールアドレスをここで使用しますか？ -->
:::info

Kubernetesでの起動はエンタープライズ機能です。アクセスを取得するには、[このページ](https://wandb.ai/site/pricing)で弊社の営業チームにお問い合わせください。
:::

## Kubernetesでのイメージのビルド

Launchエージェントは、[Kaniko](https://github.com/GoogleContainerTools/kaniko)を使用してk8s内でコンテナイメージをビルドします。Kanikoは、Dockerfileを使用してコンテナまたはk8sクラスター内でコンテナイメージをビルドするツールです。KanikoはDockerデーモンに依存せず、Dockerfile内の各コマンドを完全にユーザースペースで実行します。これにより、標準的なk8sクラスターなど、Dockerデーモンを簡単にまたは安全に実行できない環境でコンテナイメージをビルドできます。

:::tip
新しいイメージをビルドする機能なしでlaunchエージェントを使用したい場合は、launchエージェントを設定するときに`noop`ビルダータイプを使用できます。詳細は[こちら](../launch/run-agent.md#builders)。
:::

## キューの作成

k8sでジョブを起動する前に、W&Bアプリでk8sキューを作成する必要があります。k8sキューを作成するには次の手順を実行します。

1. [Launchアプリケーション](https://wandb.ai/launch)に移動します。
2. **キュー**タブをクリックします。
3. **キューの作成**ボタンをクリックします。
4. キューを作成する**エンティティ**を選択します。
5. キューの名前を入力します。
6. キューの設定を入力します。
7. **キューの作成**ボタンをクリックします。

おめでとうございます！k8sキューが作成されました。
### キュー設定

起動エージェントは、Kubernetesキューから取り出された各runに対して、[Kubernetes Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/)を作成します。Kubernetesキュー用のJSON設定は、エージェントがクラスターに送信するJob specを変更するために使用されます。設定は、[Kubernetes Job spec](https://kubernetes.io/docs/concepts/workloads/controllers/job/#writing-a-job-spec)と同じスキーマに従いますが、YAMLではなくJSONでフォーマットされ、`builder`などの追加の、ユニバーサルなキュー設定フィールドもサポートされています。

ジョブ仕様の制御により、キューレベルでリソース要求、ボリュームマウント、リトライ戦略などを指定できます。たとえば、キューから起動されたすべてのrunに対してカスタム環境変数、リソース要求、およびラベルを設定するには、次の設定のバリエーションを使用できます。

```json
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "env": [
              {
                "name": "MY_ENV_VAR",
                "value": "some-value"
              }
            ],
            "resources": {
              "requests": {
                "cpu": "1000m",
                "memory": "1Gi"
              }
            }
          }
        ]
      }
    }
  },
  "metadata": {
    "labels": {
      "queue": "k8s-test"
    }
  },
  "namespace": "wandb"
}
```

エージェントは、ジョブ仕様の最上位で自動的に以下の値を設定します。

```yaml
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 60
  template:
    spec:
      restartPolicy: Never
      containers:  # これらのセキュリティデフォルトは、pod spec内のすべてのコンテナに適用されます。
      - securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
          seccompProfile:
            type: RuntimeDefault
```

## エージェントのデプロイ

k8s上でrunを開始する前に、クラスターにエージェントをデプロイする必要があります。

### クラスター設定

クラスター内でランチエージェントを実行するためには、クラスター内に他のリソースも作成する必要があります。ここではデモンストレーションの目的のため、それらは別々に配置されていますが、1つのファイルにまとめて一度に適用することもできます。

:::tip
EKSやGKEなど、特定のKubernetesサービス用に準備されたリソースが含まれたk8sマニフェストを、[こちら](https://github.com/wandb/wandb/tree/main/wandb/sdk/launch/deploys)のsdkリポジトリで見つけることができます。
:::
#### ネームスペース

以下のKubernetesマニフェストは、`pod-security.kubernetes.io/enforce` および `pod-security.kubernetes.io/warn` ラベルを `baseline` および `latest` に設定した `wandb` という名前の名前空間を作成します。これにより、この名前空間で作成されるすべてのポッドはベースラインのポッドセキュリティポリシーが適用されます。

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: wandb
  labels:
    pod-security.kubernetes.io/enforce: baseline
    pod-security.kubernetes.io/enforce-version: latest
    pod-security.kubernetes.io/warn: baseline
    pod-security.kubernetes.io/warn-version: latest
```

#### サービスアカウントとロール

以下の Kubernetes マニフェストは、`wandb` ネームスペースに `wandb-launch-agent` という名前のロールを作成します。このロールを使って、エージェントは `wandb` ネームスペースでポッド、コンフィグマップ、シークレット、およびポッド/ログを作成できます。`wandb-cluster-role` は、エージェントが任意の名前空間でポッド、ポッド/ログ、シークレット、ジョブ、およびジョブ/ステータスを作成できるようにします。`ClusterRoleBinding`でTODOを記入して、実行する名前空間を指定してください。

このロールは、`wandb-launch-agent` サービスアカウントにバインドされます。

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: wandb
  name: wandb-launch-agent
rules:
  - apiGroups: [""]
    resources: ["pods", "configmaps", "secrets", "pods/log"]
    verbs: ["create", "get", "watch", "list", "update", "delete", "patch"]
  - apiGroups: ["batch"]
    resources: ["jobs", "jobs/status"]
    verbs: ["create", "get", "watch", "list", "update", "delete", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: job-creator
rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log", "secrets"]
    verbs: ["create", "get", "watch", "list", "update", "delete", "patch"]
  - apiGroups: ["batch"]
    resources: ["jobs", "jobs/status"]
    verbs: ["create", "get", "watch", "list", "update", "delete", "patch"]
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: wandb-launch-serviceaccount
  namespace: wandb
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: wandb-launch-role-binding
  namespace: wandb
subjects:
  - kind: ServiceAccount
    name: wandb-launch-serviceaccount
    namespace: wandb
roleRef:
  kind: Role
  name: wandb-launch-agent
  apiGroup: rbac.authorization.k8s.io

apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: wandb-launch-cluster-role-binding
  namespace: default #TODO: あなたのトレーニングNAMESPACEを設定してください
subjects:
  - kind: ServiceAccount
    name: wandb-launch-serviceaccount
    namespace: wandb
roleRef:
  kind: ClusterRole
  name: job-creator
  apiGroup: rbac.authorization.k8s.io
---
```

#### W&B APIキー

`wandb`の名前空間に、W&B APIキーを含むシークレットを作成する必要があります。このシークレットは、エージェントがW&B APIとの認証を行い、キューからジョブを取得し、起動されたrunsからメトリクスを報告するために使用されます。

```sh
kubectl -n wandb create secret  \
    generic wandb-api-key       \
    --from-literal=password=<あなたのwandb-api-キー>
```

#### エージェントの設定

最後に、`wandb`の名前空間に、エージェントの設定を含むconfigmapを作成する必要があります。このconfigmapは、エージェント自体の設定に使用されます。この設定は、使用するクラウドプロバイダと利用可能なリソースに大きく依存します。詳細については、[エージェントドキュメント](../launch/run-agent.md#agent-configuration)で確認できます。
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: wandb-launch-agent-config
  namespace: wandb
data:
  launch-config.yaml: |
    base_url: https://api.wandb.ai # TODO: wandb base urlを設定してください
    max_jobs: -1 # TODO: ここで最大同時ジョブ数を設定してください
    queues:
    - default # TODO: ここでキュー名を設定してください
    environment:
      type: gcp
      region: us-central1 # TODO: ここでgcpリージョンを設定してください
    registry:
      type: gcr
      repository: # TODO: ここでアーティファクトリポジトリ名を設定してください
      image_name: launch-images # TODO: ここでイメージ名を設定してください
    builder:
      type: kaniko
      build-context-store: gs://my-bucket/... # TODO: ここでビルドコンテキストストアを設定してください
```

### エージェントのデプロイ

エージェントを実行するために必要なすべてのリソースを作成したので、エージェントをクラスターにデプロイできるようになりました。以下のマニフェストは、クラスター内の1つのコンテナでエージェントを実行するk8sデプロイメントを定義します。エージェントは`wandb`名前空間で実行され、`wandb-launch-agent`サービスアカウントを使用します。APIキーはコンテナ内の`WANDB_API_KEY`環境変数としてマウントされます。configmapはコンテナ内の`/home/launch-agent/launch-config.yaml`にボリュームとしてマウントされます。

最新のエージェントイメージを弊社のパブリックdockerレジストリからプルすることをお勧めします。最新のイメージタグは[こちら](https://hub.docker.com/r/wandb/launch-agent-dev/tags?page=1&ordering=last_updated)で見つけることができます。
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: launch-agent
  namespace: wandb
spec:
  replicas: 1
  selector:
    matchLabels:
      app: launch-agent
  template:
    metadata:
      labels:
        app: launch-agent
    spec:
      serviceAccountName: wandb-launch-serviceaccount
      containers:
        - name: launch-agent
          image: <latest-agent-release>
          resources:
            limits:
              memory: "2Gi"
              cpu: "1000m"
          securityContext:
            allowPrivilegeEscalation: false
            runAsNonRoot: true
            runAsUser: 1000
            capabilities:
              drop: ["ALL"]
            seccompProfile:
              type: RuntimeDefault
          env:
            - name: WANDB_API_KEY
              valueFrom:
                secretKeyRef:
                  name: wandb-api-key
                  key: password
          volumeMounts:
            - name: wandb-launch-config
              mountPath: /home/launch_agent/.config/wandb
              readOnly: true
      volumes:
        - name: wandb-launch-config
          configMap:
            name: wandb-launch-configmap
```
デプロイメントを作成した後、以下のコマンドを実行してエージェントのステータスを確認できます。

```sh
kubectl -n wandb describe deployment launch-agent
```