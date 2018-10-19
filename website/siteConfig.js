/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// See https://docusaurus.io/docs/site-config.html for all the possible
// site configuration options.

/* List of projects/orgs using your project for the users page */
const users = [
  {
    caption: 'Open AI',
    image: '/img/openai.png',
    infoLink: 'https://www.openai.com',
    pinned: true,
  },
  {
    caption: 'Toyota Research Institute',
    image: '/img/tri.png',
    infoLink: 'https://www.tri.com',
    pinned: true,
  },
];

const siteConfig = {
  title: 'W&B' /* title for your website */,
  tagline: 'Documentation, Guides and Examples',
  url: 'https://docs.wandb.com' /* your website url */,
  // For github.io type URLs, you would set the url and baseUrl like:
  //   url: 'https://facebook.github.io',
  //   baseUrl: '/test-site/',
  baseUrl: '/',
  cname: 'https://docs.wandb.com',

  // Used for publishing and more
  projectName: 'docs',
  organizationName: 'wandb',
  // For top-level user or org sites, the organization is still the same.
  // e.g., for the https://JoelMarcey.github.io site, it would be set like...
  //   organizationName: 'JoelMarcey'

  // For no header links in the top nav bar -> headerLinks: [],
  headerLinks: [
  //  {doc: 'gallery', label: 'Gallery'},
     
      {doc: 'install', label: 'Docs'},
      {href: 'https://wandb.com/company', label: 'Company'},
      {href: 'https://app.wandb.ai', label: 'Login'},
      { search: true },
  //  {blog: true, label: 'Blog'},
  ],

  // If you have users set above, you add it here:
  users,

  /* path to images for header/footer */
  headerIcon: 'img/wandb-long.svg',
  footerIcon: 'img/wandb.svg',
  favicon: 'img/favicon.png',
  algolia: {
    apiKey: '0510e717ad08795f12b729ed540687cb', 
    indexName: 'wandb',
    algoliaOptions: {}
  },

  /* colors for website */
  colors: {
    primaryColor: '#55565B',
    secondaryColor: '#ECBB33',
  },

  
  /* custom fonts for website */
  /*fonts: {
    myFont: [
      "Times New Roman",
      "Serif"
    ],
    myOtherFont: [
      "-apple-system",
      "system-ui"
    ]
  },*/

  // This copyright info is used in /core/Footer.js and blog rss/atom feeds.
  copyright:
    'Copyright © ' +
    new Date().getFullYear() +
    ' Weights & Biases, Inc.',

  highlight: {
    // Highlight.js theme to use for syntax highlighting in code blocks
    theme: 'default',
  },

  // Add custom scripts here that would be placed in <script> tags
  scripts: ['https://buttons.github.io/buttons.js'],

  /* On page navigation for the current documentation page */
  onPageNav: 'separate',

  /* Open Graph and Twitter card images */
  ogImage: 'img/wandb.png',
  twitterImage: 'img/wandb.png',

  // You may provide arbitrary config keys to be used as needed by your
  // template. For example, if you need your repo's URL...
  //   repoUrl: 'https://github.com/facebook/test-site',
};

module.exports = siteConfig;
