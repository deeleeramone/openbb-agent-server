// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'OpenBB Agent Server',
  tagline:
    'Pluggable, multi-tenant agent backend that speaks the OpenBB Workspace custom-agent protocol.',
  url: 'https://deeleeramone.github.io',
  baseUrl: '/openbb-agent-server/',
  organizationName: 'deeleeramone',
  projectName: 'openbb-agent-server',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',

  // The existing docs are CommonMark, not MDX — `detect` parses `.md` as
  // Markdown (so `<...>` / `{...}` are literal) and `.mdx` as MDX.
  markdown: {
    format: 'detect',
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },

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
          path: '../docs',
          routeBasePath: '/',
          sidebarPath: './sidebars.js',
          exclude: ['_archive/**', 'reference/_vendor/**'],
          editUrl: ({docPath}) => {
            const normalized = String(docPath)
              .replace(/\\/g, '/')
              .replace(/^(\.\.\/)+/, '')
              .replace(/^docs\//, '');
            return `https://github.com/deeleeramone/openbb-agent-server/tree/main/docs/${normalized}`;
          },
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        defaultMode: 'dark',
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'OpenBB Agent Server',
        items: [
          {type: 'docSidebar', sidebarId: 'docsSidebar', position: 'left', label: 'Docs'},
          {to: '/installation', label: 'Install', position: 'left'},
          {to: '/reference/', label: 'API', position: 'left'},
          {
            href: 'https://github.com/deeleeramone/openbb-agent-server',
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
              {label: 'Installation', to: '/installation'},
              {label: 'Quick Start', to: '/quick-start'},
              {label: 'API Reference', to: '/reference/'},
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/deeleeramone/openbb-agent-server',
              },
              {label: 'OpenBB Workspace', href: 'https://docs.openbb.co/workspace'},
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} OpenBB.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['bash', 'toml', 'json', 'python'],
      },
    }),
};

export default config;
