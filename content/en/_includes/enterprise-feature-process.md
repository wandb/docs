<!-- 
INTERNAL PROCESS DOCUMENTATION - DO NOT PUBLISH

# Enterprise Feature Documentation Process

When documenting Enterprise features in W&B docs, follow these guidelines to maintain consistency:

## 1. Identifying Enterprise Features

Check the following sources:
- The pricing page at https://wandb.ai/site/pricing/
- The Enterprise licenses page at /guides/hosting/enterprise-licenses.md
- Internal feature flags in the codebase
- Product team announcements

## 2. Using Include Files

Use the appropriate include file based on the feature's availability:

- **For all Enterprise deployments**: Use `{{< readfile file="/_includes/enterprise-only.md" >}}`
- **For Pro and Enterprise Cloud plans**: Use `{{< readfile file="/_includes/enterprise-cloud-only.md" >}}`
- **For Self-Managed Enterprise only**: Use `{{< readfile file="/_includes/enterprise-self-managed-only.md" >}}`
- **For Dedicated Cloud only**: Use `{{< readfile file="/_includes/enterprise-dedicated-cloud-only.md" >}}`
- **For generic Enterprise features**: Use `{{< readfile file="/_includes/enterprise-feature.md" >}}`

## 3. Placement Guidelines

- Place the include at the beginning of the section describing the Enterprise feature
- For entire pages about Enterprise features, place it after the page title
- For subsections, place it immediately after the subsection heading

## 4. Updating the Enterprise Licenses Page

When a new Enterprise feature is added:
1. Add it to the appropriate category in /guides/hosting/enterprise-licenses.md
2. Include a brief description of what the feature does
3. Note any deployment-specific restrictions

## 5. Review Checklist

Before publishing:
- [ ] Is the feature correctly identified as Enterprise-only?
- [ ] Is the appropriate include file used?
- [ ] Is the feature listed on the Enterprise licenses page?
- [ ] Does the description match the pricing page?

## 6. Staying Informed

To stay updated on Enterprise feature changes:
- Subscribe to product update channels
- Review pricing page changes monthly
- Coordinate with the Sales Engineering team
- Monitor customer feedback about feature availability

-->