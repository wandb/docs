---
title: "Available Models"
weight: 10
description: >
  Browse the foundation models available through W&B Inference
---

W&B Inference provides access to several open-source foundation models. Each model has different strengths and use cases.

## Model comparison

| Model | Model ID (for API usage) | Type | Context Window | Parameters | Description |
|-------|--------------------------|------|----------------|------------|-------------|
| Qwen3 235B A22B Thinking-2507 | `Qwen/Qwen3-235B-A22B-Thinking-2507` | Text | 262K | 22B&nbsp;-&nbsp;235B (Active&nbsp;-&nbsp;Total) | High-performance Mixture-of-Experts model optimized for structured reasoning, math, and long-form generation |
| Qwen3 235B A22B-2507 | `Qwen/Qwen3-235B-A22B-Instruct-2507` | Text | 262K | 22B&nbsp;-&nbsp;235B (Active&nbsp;-&nbsp;Total) | Efficient multilingual, Mixture-of-Experts, instruction-tuned model, optimized for logical reasoning |
| Qwen3 Coder 480B A35B | `Qwen/Qwen3-Coder-480B-A35B-Instruct` | Text | 262K | 35B&nbsp;-&nbsp;480B (Active&nbsp;-&nbsp;Total) | Mixture-of-Experts model optimized for coding tasks such as function calling, tooling use, and long-context reasoning |
| MoonshotAI Kimi K2 | `moonshotai/Kimi-K2-Instruct` | Text | 128K | 32B&nbsp;-&nbsp;1T (Active&nbsp;-&nbsp;Total) | Mixture-of-Experts model optimized for complex tool use, reasoning, and code synthesis |
| DeepSeek R1-0528 | `deepseek-ai/DeepSeek-R1-0528` | Text | 161K | 37B&nbsp;-&nbsp;680B (Active&nbsp;-&nbsp;Total) | Optimized for precise reasoning tasks including complex coding, math, and structured document analysis |
| DeepSeek V3-0324 | `deepseek-ai/DeepSeek-V3-0324` | Text | 161K | 37B&nbsp;-&nbsp;680B (Active&nbsp;-&nbsp;Total) | Robust Mixture-of-Experts model tailored for high-complexity language processing and comprehensive document analysis |
| Meta Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | Text | 128K | 8B (Total) | Efficient conversational model optimized for responsive multilingual chatbot interactions |
| Meta Llama 3.3 70B | `meta-llama/Llama-3.3-70B-Instruct` | Text | 128K | 70B (Total) | Multilingual model excelling in conversational tasks, detailed instruction-following, and coding |
| Meta Llama 4 Scout | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | Text, Vision | 64K | 17B&nbsp;-&nbsp;109B (Active&nbsp;-&nbsp;Total) | Multi-modal model integrating text and image understanding, ideal for visual tasks and combined analysis |
| Microsoft Phi 4 Mini 3.8B | `microsoft/Phi-4-mini-instruct` | Text | 128K | 3.8B (Active&nbsp;-&nbsp;Total) | Compact, efficient model ideal for fast responses in resource-constrained environments |

## Model selection guide

Choose a model based on your specific needs:

### For general conversation and chat
- **Meta Llama 3.1 8B**: Fast and efficient for basic chat applications
- **Meta Llama 3.3 70B**: More capable for complex conversations

### For coding and development
- **Qwen3 Coder 480B A35B**: Specialized for coding tasks
- **DeepSeek V3-0324**: Strong at complex coding problems

### For reasoning and analysis
- **Qwen3 235B A22B Thinking-2507**: Excellent for structured reasoning
- **DeepSeek R1-0528**: Precise reasoning for math and logic

### For multi-modal tasks
- **Meta Llama 4 Scout**: Handles both text and images

### For resource-limited environments
- **Microsoft Phi 4 Mini 3.8B**: Small but capable model

## Using model IDs

When using the API, specify the model using its ID from the table above. For example:

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[...]
)
```

## Next steps

- Check [usage limits and pricing]({{< relref "usage-limits" >}}) for each model
- See [API reference]({{< relref "api-reference" >}}) for how to use these models
- Try models in the [W&B Playground]({{< relref "ui-guide" >}}) 