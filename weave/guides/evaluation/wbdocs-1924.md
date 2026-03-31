---
title: Score audio conversations with trace aggregation
description: Use monitor aggregation to score multi-call audio conversations after they go idle.
---

import { Callout } from '@/components/callout'

# Score audio conversations with trace aggregation

<Callout type="beta">
Trace aggregation for monitors is in **beta**. To enable this feature for your organization, contact your W&B account team.
</Callout>

Use trace aggregation to score a full audio conversation *after* it goes idle, rather than scoring each call individually as it arrives. This is useful when a single conversation spans multiple calls — for example, a phone support session logged through the OpenAI Realtime API — and you need a single quality score that reflects the entire exchange.

## How it works

When a monitor is configured with aggregation, W&B groups incoming traces by a shared identifier — either **Trace ID** or **Thread ID** — and waits for the conversation to go quiet. After the configured timeout elapses with no new calls arriving in the group, the monitor scores the conversation.

Two aggregation methods are available:

| Method | When to use |
|---|---|
| **Last message** (recommended) | Each call in the group already contains the full conversation (for example, `realtime.response` calls from the OpenAI Realtime API). W&B scores only the most recent call. |
| **All messages** | Each call contains only a fragment of the conversation. W&B extracts and combines audio from every call in the group before scoring. |

The **Last message** method is recommended because it places less load on the server.

## Prerequisites

- Your W&B account must have the trace aggregation feature flag enabled. Contact your account team if you do not see the **Aggregation** section in the monitor creation form.
- At least one trace must be logged to your Weave project before the op appears in the **Operations** list.

## Set up a monitor with trace aggregation

### Preferred method: Last message aggregation (OpenAI Realtime API)

Use this method when your integration logs `realtime.response` calls and each call contains the complete conversation up to that point.

1. Open the [W&B UI](https://wandb.ai/home) and open your Weave project.
2. From the Weave side-nav, select **Monitors**, then select **+ New Monitor**.
3. In the **Create new monitor** dialog, configure the standard monitor fields:
   - **Name**: A descriptive name, for example `audio-conversation-scorer`.
   - **Description** (optional): Explain what the monitor does.
   - **Active monitor** toggle: Set to **on**.
   - **Operations**: Select the op that produces audio traces, for example `realtime.response`.
   - **Sampling rate**: The percentage of conversations to score.

     <Callout type="warning">
     Set a sampling rate below 100% in production environments. Scoring every call at 100% increases server load unnecessarily.
     </Callout>

   - **LLM-as-a-judge configuration**: Configure your scorer name, judge model, system prompt, response format, and scoring prompt. Enable **Score Audio** to restrict the model list to audio-capable models.

4. Expand the **Aggregation** section and configure the following fields:

   | Field | Description |
   |---|---|
   | **Group by** | The identifier used to group related calls. Select **Trace ID** to group all calls that share the same trace. |
   | **Aggregation method** | Select **Last message**. After the timeout, W&B scores the most recent call in the group. |
   | **Timeout** | How long W&B waits after the last call before scoring. For example, enter `5` minutes. A longer timeout captures more complete conversations but delays scoring and increases server load. |

5. Click **Create monitor**.

After your code generates traces, review scores in the **Traces** tab by selecting the monitor's name.

### Alternate method: All messages aggregation

Use this method when individual calls contain only a fragment of the conversation audio — for example, when you are using a custom logging integration that splits audio across calls.

1. Follow steps 1–3 from the preferred method above.
2. Expand the **Aggregation** section and configure:

   | Field | Description |
   |---|---|
   | **Group by** | The identifier used to group related calls. You can select **Thread ID** for greater flexibility when grouping spans across multiple traces. |
   | **Aggregation method** | Select **All messages**. W&B extracts audio from every call in the group before scoring. |
   | **Timeout** | How long W&B waits after the last call before scoring. |

3. Click **Create monitor**.

<Callout type="warning">
The **All messages** method places more load on the server than **Last message** because W&B must process audio from every call in the group. Use this method only when your integration does not include the full conversation in each call.
</Callout>

## Configuration reference

### Aggregation fields

| Field | Required | Description |
|---|---|---|
| **Group by** | Yes | The identifier used to form aggregation groups. Options: **Trace ID**, **Thread ID**. |
| **Aggregation method** | Yes | How W&B selects which call(s) to score. Options: **Last message**, **All messages**. |
| **Timeout** | Yes | The idle duration (in minutes) to wait after the most recent call before scoring. Longer timeouts delay scoring and increase server load. |

### Choosing a timeout

The timeout controls how long W&B waits after the last call in a group before scoring the conversation. Consider the following trade-offs when choosing a value:

- **Shorter timeout** — Scores are generated more quickly, but a call that arrives late may be missed if it arrives after the timeout has already elapsed.
- **Longer timeout** — More complete conversations are captured, but scoring is delayed and server load increases.

For production workloads, a timeout of several minutes is typical. Start with a shorter value during testing to verify the aggregation behavior, then tune upward as needed.

## Limitations

- Trace aggregation is a **beta** feature. Coordinate with your W&B account team before deploying at scale to confirm scalability for your workload.
- The **All messages** method has higher server resource requirements than **Last message**. Monitor your project's performance when using this method at high sampling rates.
- Calls that arrive after the aggregation timeout has elapsed for a group are not included in the score for that group.

## Related pages

- [Set up monitors](https://docs.wandb.ai/weave/guides/evaluation/monitors)
- [Set up guardrails](https://docs.wandb.ai/weave/guides/evaluation/guardrails)
- [Scorers](https://docs.wandb.ai/weave/guides/evaluation/scorers)
