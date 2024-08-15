#!/bin/usr


initialDocURLs="http://localhost:3000/guides/"

# Create docs
npx docs-to-pdf --initialDocURLs=$initialDocURLs \
    --contentSelector="article" \
    --paginationSelector="a.pagination-nav__link.pagination-nav__link--next" \
    --excludeSelectors=".margin-vert--xl a,[class^='tocCollapsible'],.breadcrumbs,.theme-edit-this-page" \
    --protocolTimeout="3600000" \
    --coverSub="Documentation" \
    --coverTitle="Weights & Biases" \
    --outputPDFFilename="wb_dev_guide.pdf"
    # --coverImage="https://docusaurus.io/img/docusaurus.png" \