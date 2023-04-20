---
description: The W&B SDK for TypeScript, Node, and modern Web Browsers
---

# JavaScript Library

Similar to our Python library, we offer a client to track experiments in JavaScript/TypeScript.

- Log metrics from your Node server and display them in interactive plots on W&B
- Debug LLM applications with interactive traces
- Debug [LangChain.js](https://github.com/hwchase17/langchainjs) usage

This library is compatible with Node and modern JS run times. 

You can find the source code for the JavaScript client in the [Github repository](https://github.com/wandb/wandb-js).

:::info
Our JavaScript integration is still in Beta, if you run into issues please let us know!
:::

### Installation

```shell
npm install @wandb/sdk
# or ...
yarn add @wandb/sdk
```

### Usage

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
We span a seperate MessageChannel to process all api calls asynchronously.  This will cause your script to hang if you don't call `await wandb.finish()`.
:::

Node/CommonJS:

```javascript
const wandb = require('@wandb/sdk').default;
```

We're currently missing a lot of the functionality found in our Python SDK, but basic logging functionality is available. We'll be adding additional features like [Tables](https://docs.wandb.ai/guides/data-vis?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme) soon.

### Authentication and Settings

In node environments we look for `process.env.WANDB_API_KEY` and prompt for it's input if we have a TTY.  In non-node environments we look for `sessionStorage.getItem("WANDB_API_KEY")`.  Addtional settings can be [found here](https://github.com/wandb/wandb-js/blob/main/src/sdk/lib/config.ts).

# Integrations

Our [Python integrations](https://docs.wandb.ai/guides/integrations) are widely used by our community, and we hope to build out more JavaScript integrations to help LLM app builders leverage whatever tool they want. 

If you have any requests for additional integrations, we'd love you to open an issue with details about the request.

## LangChain.js

This library integrates with the popular library for building LLM applications, [LangChain.js](https://github.com/hwchase17/langchainjs).

### Usage

```typescript
import {WandbTracer} from '@wandb/sdk/integrations/langchain';

await WandbTracer.init({project: 'langchain-test'});
// run your langchain workloads...
await WandbTracer.finish();
```

:::caution
We span a seperate MessageChannel to process all api calls asynchronously.  This will cause your script to hang if you don't call `await WandbTracer.finish()`.
:::

See [this test](https://github.com/wandb/wandb-js/blob/main/src/sdk/integrations/langchain/langchain.test.ts) for a more detailed example. 