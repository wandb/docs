// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Weights & Biases Documentation',
  staticDirectories: ['static'],
  tagline: 'The developer-first MLOps platform',
  url: 'https://docs.wandb.ai',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  // onBrokenLinks: 'ignore',
  onBrokenMarkdownLinks: 'warn',
  favicon: '/img/docs-favicon.png',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'wandb', // Usually your GitHub org/user name.
  projectName: 'wandb/docodile', // Usually your repo name.

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ja'],
    path: 'i18n',
    localeConfigs: {
      en : {
        label: 'English',
        path: 'en'
      },
      ja : {
        label: 'Japanese',
        path: 'ja'
      },
    },
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          // editUrl: 'https://github.com/wandb/docodile/tree/main', // We're removing this because the repo is private so public viewers don't see a broken edit link.
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],

  plugins: [
    [
      '@laxels/docusaurus-plugin-segment',
      {
        apiKey: 'NYcqWZ8sgOCplYnItFyBaZ5ZRClWlVgl',
        host: 'wandb.ai',
        ajsPath: '/sa-docs.min.js',
        page: false,
        excludeUserAgents: ['GoogleSecurityScanner'],
      },
    ],
    [
      '@docusaurus/plugin-google-tag-manager',
      {
        containerId: 'GTM-5BL5RTH',
      },
    ],
    [
      '@docusaurus/plugin-google-gtag',
      {
        trackingID: 'G-5JYCHZZP7K',
        anonymizeIP: true,
      },
    ],
    require.resolve('docusaurus-plugin-image-zoom'),
    // require.resolve('docusaurus-lunr-search'),
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    {
      algolia: {
        appId: '2D210VB5MP',
        apiKey: '730cfa02025b8ba2e95d4c33b1e38cc7',
        indexName: 'docodile',

        // Optional: see doc section below
        // contextualSearch: true,
        contextualSearch: false,

        // Optional: Specify domains where the navigation should occur through window.location instead on history.push. Useful when our Algolia config crawls multiple documentation sites and we want to navigate with window.location.href to them.
        // externalUrlRegex: 'external\\.com|domain\\.com',

        // Optional: Replace parts of the item URLs from Algolia. Useful when using the same search index for multiple deployments using a different baseUrl. You can use regexp or string in the `from` param. For example: localhost:3000 vs myCompany.com/docs
        // replaceSearchResultPathname: {
        //   from: '/docs/', // or as RegExp: /\/docs\//
        //   to: '/',
        // },

        // Optional: Algolia search parameters
        // searchParameters: {},

        // Optional: path for search page that enabled by default (`false` to disable it)
        searchPagePath: 'search',
      },
      colorMode: {
        defaultMode: 'light',
        disableSwitch: true,
      },
      navbar: {
        logo: {
          alt: 'W&B Logo',
          src: '/img/DocsLogo.svg',
        },
        items: [
          {
            type: 'search',
            position: 'right',
          },
          {
            type: 'doc',
            docId: 'guides/intro',
            label: 'Developer guide',
            position: 'right',
          },
          {
            type: 'doc',
            docId: 'ref/README',
            label: 'Reference',
            position: 'right',
          },
          {
            type: 'doc',
            docId: 'tutorials/intro_to_tutorials',
            label: 'Tutorials',
            position: 'right',
          },
          {
            type: 'localeDropdown',
            position: 'right',
          },
          {
            href: 'https://github.com/wandb/wandb',
            html: 'GitHub<img src="/img/icon-open-new-tab.svg" class="navbar__link__icon" />',
            position: 'right',
          },
          {
            href: 'https://app.wandb.ai/login',
            html: 'Log in<img src="/img/icon-open-new-tab.svg" class="navbar__link__icon" />',
            position: 'right',
          },
        ],
      },
      docs: {
        sidebar: {
          autoCollapseCategories: true,
        },
      },
      zoom: {
        // CSS selector to apply the plugin to, defaults to '.markdown img'
        selector: '.markdown img',
        // Optional medium-zoom options
        // see: https://www.npmjs.com/package/medium-zoom#options
        options: {
          margin: 24,
          background: '#BADA55',
          scrollOffset: 0,
          container: '#zoom-container',
          template: '#zoom-template',
        },
      },
      footer: {
        style: 'light',
        copyright: `Copyright © ${new Date().getFullYear()} Weights & Biases. All rights reserved.`,
        links: [
          {
            html: `<a href="https://wandb.ai/site/terms" target="_blank" rel="noopener noreferrer" class="footer__link-item">Terms of Service</a>`,
          },
          {
            html: `<a href="https://wandb.ai/site/privacy" target="_blank" rel="noopener noreferrer" class="footer__link-item">Privacy Policy</a>`,
          },
        ],
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    },
};

module.exports = config;
