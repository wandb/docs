---
menu:
  support:
    identifier: ja-support-kb-articles-inference_server_errors_500_503
support:
- inference
title: How do I fix server errors (500, 503) with W&B Inference?
toc_hide: true
type: docs
url: /support/:filename
---

Server errors indicate temporary issues with the W&B Inference service.

## Error types

### 500 - Internal Server Error
**Message:** "The server had an error while processing your request"

This is a temporary internal error on the server side.

### 503 - Service Overloaded
**Message:** "The engine is currently overloaded, please try again later"

The service is experiencing high traffic.

## How to handle server errors

1. **Wait before retrying**
   - 500 errors: Wait 30-60 seconds
   - 503 errors: Wait 60-120 seconds

2. **Use exponential backoff**
   ```python
   import time
   import openai
   
   def call_with_retry(client, messages, model, max_retries=5):
       for attempt in range(max_retries):
           try:
               return client.chat.completions.create(
                   model=model,
                   messages=messages
               )
           except Exception as e:
               if "500" in str(e) or "503" in str(e):
                   if attempt < max_retries - 1:
                       wait_time = min(60, (2 ** attempt))
                       time.sleep(wait_time)
                   else:
                       raise
               else:
                   raise
   ```

3. **Set appropriate timeouts**
   - Increase timeout values for your HTTP client
   - Consider async operations for better handling

## When to contact support

Contact support if:
- Errors persist for more than 10 minutes
- You see patterns of failures at specific times
- Error messages contain additional details

Provide:
- Error messages and codes
- Time when errors occurred
- Your code snippet (remove API keys)
- W&B entity and project names