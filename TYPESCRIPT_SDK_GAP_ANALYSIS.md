# Weave TypeScript SDK - Documentation Gap Analysis

**Date:** December 4, 2024  
**SDK Source:** https://github.com/wandb/weave/tree/master/sdks/node/src  
**Docs Location:** `/weave/reference/typescript-sdk/`

---

## Summary

| Category | Documented | Missing | Total |
|----------|------------|---------|-------|
| Top-level exports | 15 | **1** | 16 |
| WeaveClient methods | 16 | **2** | 18 |
| Call interface | 0 | **1** | 1 |
| **Total** | 31 | **4** | 35 |

---

## 🔴 CRITICAL: Missing Documentation

### 1. `withAttributes` (Top-level export)

**File:** `clientApi.ts`  
**Status:** ❌ NOT DOCUMENTED - HIGH PRIORITY

This is a critical user-facing function that's completely missing from the docs!

```typescript
/**
 * Attach attributes to the current execution context so that any calls created
 * inside `fn` automatically inherit them. Attributes are written to the call
 * record on the trace server and surface in the Weave UI/filtering, so they're
 * ideal for tagging runs with request IDs, tenants, experiments, etc.
 *
 * Example:
 * ```ts
 * await withAttributes({requestId: 'abc'}, async () => {
 *   await myOp();
 * });
 * ```
 */
export function withAttributes<T>(
  attrs: Record<string, any>,
  fn: () => Promise<T> | T
): Promise<T> | T
```

**Action:** Create `/weave/reference/typescript-sdk/functions/withattributes.mdx`

---

### 2. `Call` interface with `setDisplayName` method

**File:** `call.ts`  
**Status:** ❌ NOT DOCUMENTED - HIGH PRIORITY

The `Call` interface extends `CallSchema` and adds a `setDisplayName` method. This is the TypeScript equivalent of Python's `call.set_display_name()`.

```typescript
export interface Call extends CallSchema {
  setDisplayName(displayName: string): Promise<void>;
}
```

**How it works:**
- When you use `client.getCall()` or `client.getCalls()`, you get `Call` objects (not just `CallSchema`)
- These `Call` objects have a `setDisplayName()` method
- If the call is still pending/uninitialized, the name is stored for when it finishes
- If the call is already finished, it immediately calls `client.updateCall()`

**Example usage:**
```typescript
const client = await weave.init('my-project')

// Get a call and update its display name
const call = await client.getCall('call-uuid-here')
await call.setDisplayName('My Custom Display Name')
```

**Action:** 
1. Create `/weave/reference/typescript-sdk/interfaces/call.mdx`
2. Update `tracing.mdx` "Set display name" TypeScript section!

---

### 3. `getCurrentAttributes` method on WeaveClient

**File:** `weaveClient.ts`  
**Status:** ❌ NOT DOCUMENTED - MEDIUM PRIORITY

```typescript
public getCurrentAttributes(): Record<string, any> {
  return this.attributesContext.getStore() || {};
}
```

**Action:** Add to `/weave/reference/typescript-sdk/classes/weaveclient.mdx`

---

### 4. `runWithAttributes` method on WeaveClient

**File:** `weaveClient.ts`  
**Status:** ❌ NOT DOCUMENTED - LOW PRIORITY (internal, use `withAttributes` instead)

```typescript
public runWithAttributes<T>(attributes: Record<string, any>, fn: () => T): T
```

This is the underlying method used by `withAttributes`. Users should use `withAttributes` instead.

**Action:** Optional - document as internal method

---

## ✅ Documented Items (Verified)

### Top-level Exports (from `index.ts`)

| Export | Documented | Location |
|--------|------------|----------|
| `init` | ✅ | functions/init.mdx |
| `login` | ✅ | functions/login.mdx |
| `withAttributes` | ❌ | **MISSING** |
| `requireCurrentCallStackEntry` | ✅ | functions/requirecurrentcallstackentry.mdx |
| `requireCurrentChildSummary` | ✅ | functions/requirecurrentchildsummary.mdx |
| `Dataset` | ✅ | classes/dataset.mdx |
| `Evaluation` | ✅ | classes/evaluation.mdx |
| `EvaluationLogger` | ✅ | classes/evaluationlogger.mdx |
| `ScoreLogger` | ✅ | classes/scorelogger.mdx |
| `CallSchema` | ✅ | interfaces/callschema.mdx |
| `CallsFilter` | ✅ | interfaces/callsfilter.mdx |
| `wrapOpenAI` | ✅ | functions/wrapopenai.mdx |
| `weaveAudio` | ✅ | functions/weaveaudio.mdx |
| `weaveImage` | ✅ | functions/weaveimage.mdx |
| `WeaveAudio` | ✅ | interfaces/weaveaudio.mdx |
| `WeaveImage` | ✅ | interfaces/weaveimage.mdx |
| `op` | ✅ | functions/op.mdx |
| `Op` | ✅ | type-aliases/op.mdx |
| `OpDecorator` | ✅ | type-aliases/opdecorator.mdx |
| `WeaveClient` | ✅ | classes/weaveclient.mdx |
| `WeaveObject` | ✅ | classes/weaveobject.mdx |
| `ObjectRef` | ✅ | classes/objectref.mdx |
| `MessagesPrompt` | ✅ | classes/messagesprompt.mdx |
| `StringPrompt` | ✅ | classes/stringprompt.mdx |

### WeaveClient Methods

| Method | Documented | Notes |
|--------|------------|-------|
| `constructor` | ✅ | |
| `waitForBatchProcessing` | ✅ | |
| `publish` | ✅ | |
| `getCall` | ✅ | Returns `Call` (not just `CallSchema`) |
| `getCalls` | ✅ | Returns `Call[]` (not just `CallSchema[]`) |
| `getCallsIterator` | ✅ | |
| `get` | ✅ | |
| `getCallStack` | ✅ | |
| `getCurrentAttributes` | ❌ | **MISSING** |
| `pushNewCall` | ✅ | |
| `runWithCallStack` | ✅ | |
| `runWithAttributes` | ❌ | Internal (use `withAttributes`) |
| `saveOp` | ✅ | |
| `createCall` | ✅ | Internal/advanced |
| `finishCall` | ✅ | Internal/advanced |
| `finishCallWithException` | ✅ | Internal/advanced |
| `updateCall` | ✅ | |
| `addScore` | ✅ | |

---

## 🔄 Documentation Updates Needed for `tracing.mdx`

Based on this analysis, the following sections in `tracing.mdx` can be updated:

### "Set display name" section (Lines 659-695)

**Current status:** Says "This feature is not available in TypeScript yet. Stay tuned!"

**Reality:** This IS available via `call.setDisplayName()`!

**Proposed update:**
```mdx
<Tab title="TypeScript">
To set the display name of a call, first retrieve the call using [`client.getCall`](/weave/reference/typescript-sdk/classes/weaveclient#getcall), then use the [`setDisplayName`](/weave/reference/typescript-sdk/interfaces/call#setdisplayname) method:

```typescript
import * as weave from 'weave'

// Initialize the client
const client = await weave.init('your-project-name')

// Get a specific call by its ID
const call = await client.getCall('call-uuid-here')

// Set the display name of the call
await call.setDisplayName('My Custom Display Name')
```
</Tab>
```

---

## Action Items

1. **[HIGH]** Create `functions/withattributes.mdx` - Document `withAttributes` function
2. **[HIGH]** Create `interfaces/call.mdx` - Document `Call` interface with `setDisplayName`
3. **[HIGH]** Update `tracing.mdx` - Fix "Set display name" TypeScript section
4. **[MEDIUM]** Update `classes/weaveclient.mdx` - Add `getCurrentAttributes` method
5. **[LOW]** Update `typescript-sdk.mdx` index - Add new items to Table of contents

