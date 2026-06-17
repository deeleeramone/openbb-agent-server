# Documentation site

The OpenBB Agent Server docs, built with [Docusaurus](https://docusaurus.io/).
The Markdown content lives one level up in [`../docs`](../docs); this
folder holds only the Docusaurus app (config, sidebar, theme).

## Develop

```sh
cd website
npm install
npm start          # live-reload dev server at http://localhost:3000/openbb-agent-server/
```

## Build

```sh
npm run build      # static site -> website/build/
npm run serve      # preview the production build
```

The build runs with strict link checking (`onBrokenLinks: 'throw'`).

## API Reference

The `../docs/reference/` pages are **generated from the package docstrings**
by [`pydoc-markdown`](https://niklasrosenstein.github.io/pydoc-markdown/),
run isolated via `uvx` (so it never touches the project's Python env, which
it conflicts with). Regenerate after changing docstrings:

```sh
npm run gen-api    # runs scripts/gen-api.mjs -> uvx pydoc-markdown + normalize
```

Requires [`uv`](https://docs.astral.sh/uv/) on PATH. The generated
`../docs/reference/` tree is committed, so a plain `npm run build` does not
need Python.
