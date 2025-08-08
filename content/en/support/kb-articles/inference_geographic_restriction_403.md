---
url: /support/:filename
title: "Why does W&B Inference say my country or region is not supported?"
toc_hide: true
type: docs
support:
  - inference
---

A 403 error with the message "Country, region, or territory not supported" means you're accessing W&B Inference from an unsupported location.

## Why this happens

W&B Inference has geographic restrictions due to compliance and regulatory requirements. The service is only accessible from supported geographic locations.

## What you can do

1. **Check the Terms of Service**
   - Review the [Terms of Service](https://docs.coreweave.com/docs/policies/terms-of-service/terms-of-use#geographic-restrictions) for the current list of supported locations

2. **Use from a supported location**
   - Access the service when in a supported country or region
   - Consider using your organization's resources in supported locations

3. **Contact your account team**
   - Enterprise customers can discuss options with their account executive
   - Some organizations may have special arrangements

## Error details

When you see this error:
```
{
  "error": {
    "code": 403,
    "message": "Country, region, or territory not supported"
  }
}
```

This is determined by your IP address location at the time of the API request. 