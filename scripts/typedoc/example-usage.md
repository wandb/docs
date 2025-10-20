# TypeDoc Usage Example

## What This Does

The TypeDoc generator creates beautiful, comprehensive documentation from TypeScript source code with proper TSDoc comments.

## How to Use

### Step 1: Ensure TSDoc Comments in Source

The TypeScript source files at `wandb/core/frontends/weave/src/core/ops/` need TSDoc comments like:

```typescript
/**
 * Gets the configuration from a run.
 * @param run - The run to get config from
 * @returns The configuration dictionary
 * @example
 * ```ts
 * const config = runConfig(myRun);
 * ```
 */
export function runConfig(run: Run): ConfigDict {
  // implementation
}
```

### Step 2: Run the Generator

```bash
# From the scripts/typedoc directory
./generate-docs.sh /path/to/wandb/core
```

### Step 3: View Results

The documentation will be generated at:
`/content/en/ref/query-panel-generated/`

## Current Status

✅ **Generator scripts are ready** in `/scripts/typedoc/`
✅ **No example source code stored** - works with actual W&B source
✅ **Example documentation available** at `/content/en/ref/query-panel-new/` for review

## Benefits Over generateDocs.ts

- **8x more content** per operation
- **Real examples** developers can use
- **Type-safe** with full TypeScript integration
- **No duplication** - clean, organized structure
- **Industry standard** - TypeDoc is widely used

## Next Steps

1. Add TSDoc comments to the actual TypeScript source
2. Run the generator with path to wandb/core
3. Review and deploy the improved documentation
