---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# トラブルシューティング

### wandbがクラッシュした場合、トレーニングrunもクラッシュしますか？

トレーニングrunに干渉しないことは非常に重要です。wandbは別のプロセスで実行され、wandbが何らかの理由でクラッシュしてもトレーニングが継続するようにします。インターネットが切断された場合でも、wandbは[wandb.ai](https://wandb.ai)にデータを送り続けようと再試行します。

### ローカルで正常にトレーニングしているのに、W&Bでrunがクラッシュと表示されるのはなぜですか？

これは接続の問題である可能性があります。サーバーがインターネット アクセスを失い、データがW&Bに同期されなくなると、短期間の再試行後にrunがクラッシュとしてマークされます。

### ログはトレーニングをブロックしますか？

「ログ関数は遅延評価されますか？ネットワークに依存してサーバーに結果を送信し、その後ローカル操作を続行したくありません。」

`wandb.log`を呼び出すとローカルファイルに1行書き込まれ、ネットワーク呼び出しをブロックしません。 `wandb.init`を呼び出すと、トレーニングプロセスとは非同期にファイルシステムの変更を監視し、Webサービスと通信する新しいプロセスが同じマシン上で起動されます。

### wandbがターミナルやjupyter notebookの出力に書き込まないようにするにはどうすればよいですか？

環境変数[`WANDB_SILENT`](../track/environment-variables.md)を`true`に設定します。

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Jupyter Notebook', value: 'notebook'},
    {label: 'Command Line', value: 'command-line'},
  ]}>
  <TabItem value="python">

```python
os.environ["WANDB_SILENT"] = "true"
```

  </TabItem>
  <TabItem value="notebook">

```python
%env WANDB_SILENT=true
```

  </TabItem>
  <TabItem value="command-line">

```python
WANDB_SILENT=true
```

  </TabItem>
</Tabs>

### wandbを使用してジョブを停止するにはどうすればよいですか？

キーボードの`Ctrl+D`を押して、wandbを使用して計測されたスクリプトを停止します。

### ネットワークの問題に対処するにはどうすればよいですか？

次のようなSSLまたはネットワークエラーが発生している場合：`wandb: Network error (ConnectionError), entering retry loop`。この問題を解決するためにいくつかの方法を試すことができます：

1. SSL証明書をアップグレードします。Ubuntu サーバーでスクリプトを実行している場合は、`update-ca-certificates`を実行してください。有効なSSL証明書がないと、トレーニングログを同期できません。これはセキュリティ上の脆弱性です。
2. ネットワークが不安定な場合は、[オフラインモード](../track/launch.md)でトレーニングを実行し、インターネット アクセスのあるマシンからファイルを同期します。
3. [W&B Private Hosting](../hosting/intro.md)を試してください。これはマシン上で動作し、クラウドサーバーにファイルを同期しません。

`SSL CERTIFICATE_VERIFY_FAILED`: このエラーは会社のファイアウォールが原因である可能性があります。ローカルCAを設定し、次のコマンドを使用できます：

`export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`

### トレーニング中にインターネット接続が失われた場合、どうなりますか？

ライブラリがインターネットに接続できない場合、再試行ループに入り、ネットワークが復元されるまでメトリクスのストリーミングを試み続けます。この間、プログラムは引き続き実行できます。

インターネットのないマシンで実行する必要がある場合は、`WANDB_MODE=offline`を設定してメトリクスをローカルのハードドライブにのみ保存できます。後で`wandb sync DIRECTORY`を呼び出して、データをサーバーにストリーミングできます。