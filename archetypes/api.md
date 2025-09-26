---
title: "{{ replace .Name "-" " " | title }}"
date: {{ .Date }}
draft: false
layout: "api"
description: "API documentation for {{ replace .Name "-" " " | title }}"
weight: 100

# API Specification Configuration
# Option 1: External URL
# api_spec_url: "https://example.com/openapi.yaml"

# Option 2: Local file (place in static/api-specs/)
# api_spec_file: "api-specs/your-api-spec.yaml"

# Optional: Redoc configuration
# redoc_options:
#   hideDownloadButton: false
#   disableSearch: false
#   expandResponses: "200,201"
#   requiredPropsFirst: true
---

<!-- Optional introduction content goes here -->
<!-- This will appear above the API documentation -->
