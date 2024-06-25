---
description: W&B SDK for TypeScript、Node、最新のWebブラウザ向け
---


# JavaScript Library

私たちのPythonライブラリと同様に、JavaScript/TypeScriptでExperimentsをトラックするクライアントを提供しています。

- Nodeサーバーからメトリクスをログし、W&Bでインタラクティブなプロットに表示
- インタラクティブなトレースでLLMアプリケーションをデバッグ
- [LangChain.js](https://github.com/hwchase17/langchainjs)の使用をデバッグ

このライブラリはNodeおよびモダンなJSランタイムと互換性があります。

JavaScriptクライアントのソースコードは[Githubリポジトリ](https://github.com/wandb/wandb-js)で見つけることができます。

:::info
私たちのJavaScriptインテグレーションはまだベータ版です。問題が発生した場合はお知らせください！
:::

### インストール

```shell
npm install @wandb/sdk
# or ...
yarn add @wandb/sdk
```

### 使用方法

TypeScript/ESM:

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

:::caution
すべてのAPIコールを非同期で処理するために、別のMessageChannelを生成します。これにより、`await wandb.finish()`を呼び出さないとスクリプトがハングする原因となります。
:::

Node/CommonJS:

```javascript
const wandb = require('@wandb/sdk').default;
```

現在、Python SDKにある多くの機能がまだ欠けていますが、基本的なロギング機能は利用可能です。近日中に[Tabels](https://docs.wandb.ai/guides/tables?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme)などの追加機能を提供する予定です。

### 認証と設定

Node環境では`process.env.WANDB_API_KEY`を探し、TTYがある場合は入力を促します。非Node環境では`sessionStorage.getItem("WANDB_API_KEY")`を探します。追加の設定については[こちら](https://github.com/wandb/wandb-js/blob/main/src/sdk/lib/config.ts)をご覧ください。

# インテグレーション

私たちの[Pythonインテグレーション](https://docs.wandb.ai/guides/integrations)はコミュニティで広く使用されており、LLMアプリケーションビルダーが好きなツールを活用できるように、さらなるJavaScriptインテグレーションを構築したいと考えています。

追加のインテグレーションについてリクエストがある場合は、詳細なリクエスト内容を記載したissueを開いていただきたいです。

## LangChain.js

このライブラリは、LLMアプリケーションを構築するための人気ライブラリ[LangChain.js](https://github.com/hwchase17/langchainjs)バージョン>=0.0.75に統合されています。

### 使用方法

```typescript
import {WandbTracer} from '@wandb/sdk/integrations/langchain';

const wbTracer = await WandbTracer.init({project: 'langchain-test'});
// run your langchain workloads...
chain.call({input: "My prompt"}, wbTracer)
await WandbTracer.finish();
```

:::caution
すべてのAPIコールを非同期で処理するために、別のMessageChannelを生成します。これにより、`await WandbTracer.finish()`を呼び出さないとスクリプトがハングする原因となります。
:::

より詳しい例については[このテスト](https://github.com/wandb/wandb-js/blob/main/src/sdk/integrations/langchain/langchain.test.ts)をご覧ください。