---
title: JavaScript ライブラリ
description: TypeScript、Node、および最新の Web ブラウザ向けの W&B SDK
---

Python ライブラリと同様に、JavaScript/TypeScript でも実験管理をトラッキングできるクライアントを提供しています。

- Node サーバーからメトリクスをログして、W&B 上でインタラクティブなプロットとして表示
- インタラクティブなトレースで LLM アプリケーションをデバッグ
- [LangChain.js](https://github.com/hwchase17/langchainjs) の利用をデバッグ

このライブラリは Node および最新の JS ランタイムで利用可能です。

JavaScript クライアントのソースコードは [Github repository](https://github.com/wandb/wandb-js) でご覧いただけます。

{{% alert %}}
JavaScript インテグレーションはまだ Beta 版です。もし問題が発生した場合はぜひご連絡ください。
{{% /alert %}}

## インストール

```shell
npm install @wandb/sdk
# または ...
yarn add @wandb/sdk
```

## 使い方

### TypeScript/ESM:

```typescript
import wandb from '@wandb/sdk'

async function track() {
    await wandb.init({config: {test: 1}});
    wandb.log({acc: 0.9, loss: 0.1});
    wandb.log({acc: 0.91, loss: 0.09});
    await wandb.finish();
}

await track()
```

{{% alert color="secondary" %}}
すべての API コールを非同期で処理するため、別スレッドの MessageChannel を生成します。`await wandb.finish()` を呼ばないと、スクリプトが終了しないのでご注意ください。
{{% /alert %}}

### Node/CommonJS:

```javascript
const wandb = require('@wandb/sdk').default;
```

現在、Python SDK で提供している多くの機能はまだありませんが、基本的なログ機能は利用可能です。[Tables]({{< relref "/guides/models/tables/?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme" >}}) など、今後さらに機能追加を予定しています。

## 認証と設定

Node 環境では `process.env.WANDB_API_KEY` を探し、TTY があれば入力を促します。Node 以外の環境では `sessionStorage.getItem("WANDB_API_KEY")` を参照します。追加の設定については [こちら](https://github.com/wandb/wandb-js/blob/main/src/sdk/lib/config.ts) をご覧ください。

## インテグレーション

[Python インテグレーション]({{< relref "/guides/integrations/" >}}) はコミュニティで広く利用されていますが、私たちは JavaScript 向けのインテグレーションも拡充していき、LLM アプリ開発者が好きなツールを活用できるようサポートしていきます。

追加インテグレーションのリクエストがある場合は、ぜひ詳細を記載のうえ issue をオープンしてください。

## LangChain.js

このライブラリは、LLM アプリケーション開発で人気の [LangChain.js](https://github.com/hwchase17/langchainjs) バージョン >= 0.0.75 に対応しています。

```typescript
import {WandbTracer} from '@wandb/sdk/integrations/langchain';

const wbTracer = await WandbTracer.init({project: 'langchain-test'});
// LangChain のワークロードを実行
chain.call({input: "My prompt"}, wbTracer)
await WandbTracer.finish();
```

{{% alert color="secondary" %}}
すべての API コールを非同期で処理するため、別スレッドの MessageChannel を生成します。`await WandbTracer.finish()` を呼ばないと、スクリプトが終了しないのでご注意ください。
{{% /alert %}}

より詳しい例は [このテスト](https://github.com/wandb/wandb-js/blob/main/src/sdk/integrations/langchain/langchain.test.ts) をご参照ください。