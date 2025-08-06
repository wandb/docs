---
menu:
  support:
    identifier: ko-support-kb-articles-inference_rate_limit_429_error
support:
- inference
title: Why am I getting rate limit errors (429) with W&B Inference?
toc_hide: true
type: docs
url: /support/:filename
---

Rate limit errors (429) occur when you exceed concurrency limits or run out of credits.

## Types of 429 errors

### Concurrency limit reached
**Error:** "Concurrency limit reached for requests"

**Solution:**
- Reduce the number of parallel requests
- Add delays between requests
- Implement exponential backoff
- Note: Rate limits apply per W&B project

### Quota exceeded
**Error:** "You exceeded your current quota, please check your plan and billing details"

**Solution:**
- Check your credit balance in the W&B Billing page
- Purchase more credits or upgrade your plan
- Request a limit increase from support

### Personal account limitation
**Error:** "W&B Inference isn't available for personal accounts"

**Solution:**
- Switch to a non-personal account
- Create a Team to access W&B Inference
- Personal entities were deprecated in May 2024

## Best practices to avoid rate limits

1. **Implement retry logic with exponential backoff:**
   ```python
   import time
   
   def retry_with_backoff(func, max_retries=3):
       for i in range(max_retries):
           try:
               return func()
           except Exception as e:
               if "429" in str(e) and i < max_retries - 1:
                   time.sleep(2 ** i)
               else:
                   raise
   ```

2. **Use batch processing instead of parallel requests**

3. **Monitor your usage in the W&B Billing page**

## Default spending caps

- **Pro accounts:** $6,000/month
- **Enterprise accounts:** $700,000/year

Contact your account executive or support to adjust limits.