---
menu:
  support:
    identifier: ja-support-kb-articles-inference_error_handling_best_practices
support:
- inference
title: What are the best practices for handling W&B Inference errors?
toc_hide: true
type: docs
url: /support/:filename
---

Follow these best practices to handle W&B Inference errors gracefully and maintain reliable applications.

## 1. Always implement error handling

Wrap API calls in try-except blocks:

```python
import openai

try:
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages
    )
except Exception as e:
    print(f"Error: {e}")
    # Handle error appropriately
```

## 2. Use retry logic with exponential backoff

```python
import time
from typing import Optional

def call_inference_with_retry(
    client, 
    messages, 
    model: str,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            # Calculate delay with exponential backoff
            delay = base_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed, retrying in {delay}s...")
            time.sleep(delay)
    
    return None
```

## 3. Monitor your usage

- Track credit usage in the W&B Billing page
- Set up alerts before hitting limits
- Log API usage in your application

## 4. Handle specific error codes

```python
def handle_inference_error(error):
    error_str = str(error)
    
    if "401" in error_str:
        # Invalid authentication
        raise ValueError("Check your API key and project configuration")
    elif "429" in error_str:
        if "quota" in error_str:
            # Out of credits
            raise ValueError("Insufficient credits")
        else:
            # Rate limited
            return "retry"
    elif "500" in error_str or "503" in error_str:
        # Server error
        return "retry"
    else:
        # Unknown error
        raise
```

## 5. Set appropriate timeouts

Configure reasonable timeouts for your use case:

```python
# For longer responses
client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',
    api_key="your-api-key",
    timeout=60.0  # 60 second timeout
)
```

## Additional tips

- Log errors with timestamps for debugging
- Use async operations for better concurrency handling
- Implement circuit breakers for production systems
- Cache responses when appropriate to reduce API calls