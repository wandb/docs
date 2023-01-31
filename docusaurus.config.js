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
  // onBrokenLinks: 'throw',
  onBrokenLinks: 'ignore',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/docs-favicon.png',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'wandb', // Usually your GitHub org/user name.
  projectName: 'wandb/docodile', // Usually your repo name.

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
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
          editUrl: 'https://github.com/wandb/docodile',
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
      'docusaurus-plugin-segment',
      {
        apiKey: 'NYcqWZ8sgOCplYnItFyBaZ5ZRClWlVgl',
        host: 'wandb.ai',
        ajsPath: '/sa-docs.min.js',
      },
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    {
      colorMode: {
        defaultMode: 'light',
        disableSwitch: true,
      },
      navbar: {
        title: 'Documentation',
        logo: {
          alt: 'W&B Logo',
          src: 'img/docs-favicon.png',
        },
        items: [
          {
            type: 'doc',
            docId: 'guides/intro',
            position: 'left',
            label: 'Developer Guide',
          },
          {
            type: 'doc',
            position: 'left',
            docId: 'ref/README',
            label: 'Reference',
          },
          {
            href: 'https://github.com/wandb/docodile',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'light',
        copyright: `Copyright © ${new Date().getFullYear()} Weights & Biases. All rights reserved.`, 
        links: [
          {
            label: 'Terms of Service',
            href: 'https://wandb.ai/site/terms',
          },
          {
            label: 'Privacy Policy',
            href: 'https://wandb.ai/site/privacy',
          },
    ]
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    },
};

module.exports = config;
