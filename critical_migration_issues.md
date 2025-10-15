# Critical Migration Issues Report

These issues will likely cause visible problems in the rendered documentation.


## Summary

- **Total critical issues:** 1285
- **Files affected:** 71
- **Categories:** 4


## Hugo Shortcodes (19 issues)

1. **README.md:96**
   - Issue: Hugo shortcode will render as raw text: {{< tabpane >}}
   ```
   {{< tabpane text=true >}}
   ```
2. **README.md:108**
   - Issue: Hugo shortcode will render as raw text: {{< img >}}
   ```
   {{< img src="/images/app_ui/automated_workspace.svg" >}}
   ```
3. **README.md:111**
   - Issue: Hugo shortcode will render as raw text: {{< img >}}
   ```
   {{< img src="/images/app_ui/automated_workspace.svg" alt="automated workspace icon" >}}
   ```
4. **README.md:114**
   - Issue: Hugo shortcode will render as raw text: {{< img >}}
   ```
   {{< img src="/images/app_ui/automated_workspace.svg" alt="automated workspace icon" width="32px" >}}
   ```
5. **README.md:117**
   - Issue: Hugo shortcode will render as raw text: {{< img >}}
   ```
   {{< img src="/images/app_ui/demo_make_a_custom_chart_bar_chart.gif" alt="Creating a custom bar chart
   ```
6. **README.md:121**
   - Issue: Hugo shortcode will render as raw text: {{< cta >}}
   ```
   {{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs
   ```
7. **README.md:125**
   - Issue: Hugo shortcode will render as raw text: {{< prism >}}
   ```
   {{< prism file="/webhook_test.sh" title="webhook_test.sh">}}{{< /prism >}}
   ```
8. **README.md:131**
   - Issue: Hugo shortcode will render as raw text: {{< readfile >}}
   ```
   {{< readfile file="/_includes/enterprise-only.md" >}}
   ```
9. **README.md:80**
   - Issue: Hugo shortcode will render as raw text: {{% %s %}}
   ```
   {{% alert %}}
   ```
10. **README.md:85**
   - Issue: Hugo shortcode will render as raw text: {{% %s %}}
   ```
   {{% alert title="Undo changes to your workspace" %}}
   ```

*... and 9 more*

## Docusaurus Components (24 issues)

1. **weave/guides/tools/evaluation_playground.mdx:46**
   - Issue: Docusaurus admonition (:::note) - will render as text
   ```
   :::note
   ```
2. **weave/guides/integrations/agno.mdx:7**
   - Issue: Docusaurus admonition (:::tip) - will render as text
   ```
   :::tip
   ```
3. **weave/guides/integrations/agno.mdx:31**
   - Issue: Docusaurus admonition (:::important) - will render as text
   ```
   :::important
   ```
4. **weave/guides/integrations/agno.mdx:46**
   - Issue: Docusaurus admonition (:::important) - will render as text
   ```
   :::important
   ```
5. **weave/guides/integrations/index.mdx:3**
   - Issue: Docusaurus admonition (:::info) - will render as text
   ```
   :::info[Integration Tracking]
   ```
6. **weave/guides/evaluation/evaluation_logger.mdx:23**
   - Issue: Docusaurus admonition (:::important) - will render as text
   ```
   :::important Track token usage and cost
   ```
7. **weave/guides/tracking/threads.mdx:358**
   - Issue: Docusaurus admonition (:::info) - will render as text
   ```
   :::info
   ```
8. **weave/guides/tracking/ops.mdx:25**
   - Issue: Docusaurus admonition (:::note) - will render as text
   ```
   :::note
   ```
9. **weave/guides/core-types/datasets.mdx:186**
   - Issue: Docusaurus admonition (:::important) - will render as text
   ```
   :::important
   ```
10. **weave/guides/core-types/media.mdx:625**
   - Issue: Docusaurus admonition (:::tip) - will render as text
   ```
   :::tip
   ```

*... and 14 more*

## Missing Images (5 issues)

1. **weave/guides/integrations/verifiers.mdx:13**
   - Issue: Image path to /static/ - likely broken
   ```
   ![verifiers wandb run page](/static/img/verifiers.gif)
   ```
2. **weave/guides/core-types/media.mdx:310**
   - Issue: Image path to /static/ - likely broken
   ```
   ![Video logging in Weave](/static/img/video.png)
   ```
3. **weave/guides/core-types/media.mdx:448**
   - Issue: Image path to /static/ - likely broken
   ```
   ![PDF document logging in Weave](/static/img/pdf.png)
   ```
4. **weave/guides/core-types/media.mdx:554**
   - Issue: Image path to /static/ - likely broken
   ```
   ![Audio logging in Weave](/static/img/audio.png)
   ```
5. **weave/guides/core-types/media.mdx:738**
   - Issue: Image path to /static/ - likely broken
   ```
   ![HTML logging in Weave](/static/img/html.png)
   ```

## Broken Syntax (1237 issues)

1. **weave/reference/typescript-sdk/weave/classes/WeaveClient.mdx:319**
   - Issue: Table with empty column header - breaks MDX parsing
   ```
   | Name |
|
   ```
2. **ja/platform/app/settings-page/teams.mdx:67**
   - Issue: Table with empty column header - breaks MDX parsing
   ```
   | チームメンバーを追加         |          |
   ```
3. **ja/platform/app/settings-page/teams.mdx:68**
   - Issue: Table with empty column header - breaks MDX parsing
   ```
   | チームメンバーを削除         |          |
   ```
4. **ja/platform/app/settings-page/teams.mdx:69**
   - Issue: Table with empty column header - breaks MDX parsing
   ```
   | チーム設定を管理             |          |
   ```
5. **ja/platform/app/settings-page/teams.mdx:76**
   - Issue: Table with empty column header - breaks MDX parsing
   ```
   | エイリアスを追加する           |          |
   ```
6. **ja/platform/app/settings-page/teams.mdx:77**
   - Issue: Table with empty column header - breaks MDX parsing
   ```
   | モデルをレジストリに追加する   |          |
   ```
7. **ja/platform/app/settings-page/teams.mdx:80**
   - Issue: Table with empty column header - breaks MDX parsing
   ```
   | レジストリ管理者を追加または削除する |          |
   ```
8. **ja/platform/app/settings-page/teams.mdx:81**
   - Issue: Table with empty column header - breaks MDX parsing
   ```
   | 保護されたエイリアスを追加または削除する |          |
   ```
9. **ja/platform/app/settings-page/teams.mdx:91**
   - Issue: Table with empty column header - breaks MDX parsing
   ```
   | レポートを作成する |          |
   ```
10. **ja/platform/app/settings-page/teams.mdx:92**
   - Issue: Table with empty column header - breaks MDX parsing
   ```
   | レポートを編集する |          |
   ```

*... and 1227 more*

## Most Affected Files

- **ja/models/ref/query-panel/string.mdx**: 92 critical issues
- **ko/models/ref/query-panel/string.mdx**: 92 critical issues
- **ja/models/ref/query-panel/float.mdx**: 78 critical issues
- **ja/models/ref/query-panel/number.mdx**: 78 critical issues
- **ja/models/ref/query-panel/int.mdx**: 78 critical issues
- **ko/models/ref/query-panel/float.mdx**: 78 critical issues
- **ko/models/ref/query-panel/number.mdx**: 78 critical issues
- **ko/models/ref/query-panel/int.mdx**: 78 critical issues
- **ko/models/ref/python/public-api/api.mdx**: 59 critical issues
- **ja/models/ref/query-panel/run.mdx**: 46 critical issues