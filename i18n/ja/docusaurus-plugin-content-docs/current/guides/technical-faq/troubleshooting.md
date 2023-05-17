import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# トラブルシューティング

### もしwandbがクラッシュしたら、私のトレーニング実行もクラッシュする可能性がありますか？

私たちにとっては、あなたのトレーニング実行に干渉しないことが非常に重要です。wandbを別のプロセスで実行しているので、もしwandbが何らかの理由でクラッシュしても、あなたのトレーニングは継続して実行されます。インターネットが切れた場合でも、wandbは[wandb.ai](https://wandb.ai)にデータを送信し続けるためにリトライを続けます。

### なぜローカルでうまくトレーニングしているのに、W&Bで実行がクラッシュしたと表示されるのですか？

これはおそらく接続の問題です。あなたのサーバーがインターネットアクセスを失ってデータがW&Bに同期されなくなると、短時間のリトライの後に実行をクラッシュしたとマークされます。

### ログを記録することが私のトレーニングをブロックしますか？

"ログ機能は遅延しているのですか？結果をあなたのサーバーに送り、それからローカルの操作に戻ることに依存したくありません。"

`wandb.log`を呼び出すと、ローカルのファイルに1行書き込まれます。これはネットワーク呼び出しをブロックしません。`wandb.init`を呼び出すと、ファイルシステムの変更をリッスンしてウェブサービスと非同期で通信する同じマシン上で新しいプロセスが起動されます。

### wandbが私のターミナルやjupyterノートブックの出力に書き込むのをどうやって止めますか？

環境変数[`WANDB_SILENT`](../track/environment-variables.md)を`true`に設定してください。

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


### wandbでジョブを停止する方法は？

wandbを使ってインストゥルメントされたスクリプトを停止するには、キーボードで`Ctrl+D`を押してください。

### ネットワークの問題にどのように対処すればいいですか？

SSLまたはネットワークエラーが表示される場合：`wandb: Network error (ConnectionError), entering retry loop`。この問題を解決するために、いくつかの異なるアプローチを試すことができます：
1. SSL証明書をアップグレードしてください。Ubuntuサーバーでスクリプトを実行している場合は、`update-ca-certificates` を実行します。有効なSSL証明書がないと、セキュリティ上の脆弱性があるためトレーニングログを同期できません。

2. ネットワークが不安定な場合は、[オフラインモード](https://docs.wandb.ai/guides/track/launch#is-it-possible-to-save-metrics-offline-and-sync-them-to-w-and-b-later)でトレーニングを実行し、インターネットアクセスがあるマシンからファイルを同期してください。

3. [W&Bプライベートホスティング](../hosting/intro.md)を試してみてください。これはあなたのマシンで動作し、私たちのクラウドサーバーにファイルを同期しません。



`SSL CERTIFICATE_VERIFY_FAILED`: このエラーは、あなたの会社のファイアウォールが原因である可能性があります。ローカルのCAを設定し、次のように使用できます。



`export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`



### モデルのトレーニング中にインターネット接続が切れた場合、どうなりますか？



ライブラリがインターネットに接続できない場合、リトライループに入り、ネットワークが回復するまでメトリクスをストリームし続けます。その間、プログラムは実行を続けることができます。



インターネットのないマシンで実行する必要がある場合は、`WANDB_MODE=offline` を設定して、メトリクスをハードドライブにローカルに保存します。後で `wandb sync DIRECTORY` を呼び出すことで、データをサーバーにストリームできます。