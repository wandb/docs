// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'W&B Docs',
  tagline: 'W&B are cool',
  url: 'https://your-docusaurus-test-site.com',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  // favicon: 'img/favicon.ico',
  favicon: 'img/docs-favicon.png',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'facebook', // Usually your GitHub org/user name.
  projectName: 'docusaurus', // Usually your repo name.

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
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/wandb/docodile',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'Documentation',
        logo: {
          alt: 'W&B Logo',
          src: 'img/docs-favicon.png',          
        },
        items: [
          {
            type: 'doc',
            docId: 'reference-guide/intro',
            position: 'left',
            label: 'Developer Guide',
          },
          {
            type: 'doc',
            docId: 'integrations/intro',
            position: 'left',
            label: 'Integrations',
          },
          {
            type: 'docSidebar',
            position: 'left',
            sidebarId: 'howtoguides',
            label: 'How-to Guides',
          },
          {
            type: 'docSidebar',
            position: 'left',
            sidebarId: 'api',
            label: 'API',
          },
          {
            href: 'https://github.com/wandb/docodile',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Developer Guide',
                to: '/docs/reference-guide/intro',
              },
              {
                label: 'Integrations',
                to: '/docs/integrations/intro',
              },
              {
                label: 'How-to Guides',
                to: '/docs/howtoguides/intro',
              },
              {
                label: 'API',
                to: '/docs/api/intro'
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'W&B Community',
                href: 'https://community.wandb.ai/',
              },
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/wandb',
              },
              // {
              //   label: 'Discord',
              //   href: 'https://discordapp.com/invite/docusaurus',
              // },
            ],
          },
          {
            title: 'Connect',
            items: [
              {
                label: 'W&B Fully Connected',
                href: 'https://wandb.ai/fully-connected',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/wandb/wandb',
              },
              {
                label: 'Twitter',
                href: 'https://twitter.com/weights_biases',
              },
              {
                label: 'YouTube',
                href: 'https://www.youtube.com/c/WeightsBiases',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} My Project, Inc. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;
