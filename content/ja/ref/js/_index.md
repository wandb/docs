---
title: JavaScript ライブラリ
description: TypeScript、Node、および最新の Web ブラウザ向けの W&B SDK
menu:
  reference:
    identifier: ja-ref-js-_index
---

Python ライブラリと同様に、JavaScript/TypeScript でも実験管理を行うためのクライアントを提供しています。

- Node サーバーからメトリクスをログし、W&B でインタラクティブなプロットとして表示できます
- インタラクティブなトレースで LLM アプリケーションのデバッグができます
- [LangChain.js](https://github.com/hwchase17/langchainjs) の利用をデバッグできます

このライブラリは Node および最新の JS ランタイムに対応しています。

JavaScript クライアントのソースコードは [Github リポジトリ](https://github.com/wandb/wandb-js) で公開されています。

{{% alert %}}
JavaScript インテグレーションは現在ベータ版です。不具合があればご連絡ください。
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
すべての API コールを非同期で処理するために、別の MessageChannel を生成しています。 `await wandb.finish()` を呼び出さない場合、スクリプトが終了しなくなりますのでご注意ください。
{{% /alert %}}

### Node/CommonJS:

```javascript
const wandb = require('@wandb/sdk').default;
```

現時点では Python SDK に比べると多くの機能が未実装ですが、基本的なログ機能は利用できます。[Tables]({{< relref path="/guides/models/tables/?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme" lang="ja" >}}) など、今後さらに多くの機能を追加予定です。

## 認証と設定

Node 環境では `process.env.WANDB_API_KEY` を参照し、TTY があれば入力を促します。Node 以外の環境では `sessionStorage.getItem("WANDB_API_KEY")` を参照します。その他の設定については [こちら](https://github.com/wandb/wandb-js/blob/main/src/sdk/lib/config.ts) をご覧ください。

## インテグレーション

[Python統合]({{< relref path="/guides/integrations/" lang="ja" >}}) はコミュニティで広く利用されています。これから LLM アプリの開発者が自由にツールを選択できるよう、JavaScript インテグレーションも強化していきます。

追加してほしいインテグレーションがあれば、お気軽にリクエストの詳細を添えて Issue を作成してください。

## LangChain.js

このライブラリは、LLM アプリ開発で人気の [LangChain.js](https://github.com/hwchase17/langchainjs) （バージョン >= 0.0.75）と連携します。

```typescript
import {WandbTracer} from '@wandb/sdk/integrations/langchain';

const wbTracer = await WandbTracer.init({project: 'langchain-test'});
// langchain のワークロードを実行...
chain.call({input: "My prompt"}, wbTracer)
await WandbTracer.finish();
```

{{% alert color="secondary" %}}
すべての API コールを非同期で処理するために、別の MessageChannel を生成しています。 `await WandbTracer.finish()` を呼び出さないと、スクリプトが終了しない場合があります。
{{% /alert %}}

より詳しい例は [こちらのテスト](https://github.com/wandb/wandb-js/blob/main/src/sdk/integrations/langchain/langchain.test.ts) をご覧ください。