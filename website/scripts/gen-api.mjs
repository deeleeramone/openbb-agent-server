// Regenerate the API Reference: run pydoc-markdown (isolated via uvx) to
// emit Markdown from the package docstrings, then normalize the tree for
// Docusaurus:
//   - lift docs/reference/openbb_agent_server/* up to docs/reference/*
//     (so pages live at reference/<subpath>, matching cross-links)
//   - rename every __init__.md to index.md (Docusaurus category index)
//     and fix its sidebar_label to the package name
//   - drop pydoc-markdown's own sidebar.json (Docusaurus autogenerates)
//
// Usage:  npm run gen-api        (from ./website)
import {execFileSync} from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const repoRoot = path.resolve(import.meta.dirname, '..', '..');
const refDir = path.join(repoRoot, 'docs', 'reference');

fs.rmSync(refDir, {recursive: true, force: true});

// The package docstrings use Sphinx :role:`target` cross-references (the
// numpy convention). pydoc-markdown's renderer mis-resolves single-backtick
// refs that match a same-module symbol, so we render from a TEMP COPY of the
// package in which roles are rewritten to ``code`` literals — the real source
// keeps its Sphinx roles untouched. (:class:`~a.b.Name` -> ``Name``.)
function rolesToCode(text) {
  return text.replace(/:[a-zA-Z]+:`(~?)([^`]+)`/g, (_m, tilde, target) =>
    '``' + (tilde ? target.split('.').pop() : target) + '``',
  );
}
function copyPackage(srcDir, dstDir) {
  fs.mkdirSync(dstDir, {recursive: true});
  for (const e of fs.readdirSync(srcDir, {withFileTypes: true})) {
    if (e.name === '__pycache__') continue;
    const s = path.join(srcDir, e.name);
    const d = path.join(dstDir, e.name);
    if (e.isDirectory()) copyPackage(s, d);
    else if (e.name.endsWith('.py'))
      fs.writeFileSync(d, rolesToCode(fs.readFileSync(s, 'utf8')));
  }
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'oas-genapi-'));
try {
  copyPackage(path.join(repoRoot, 'openbb_agent_server'),
    path.join(tmp, 'openbb_agent_server'));
  const cfg = `loaders:
  - type: python
    search_path: ["."]
    packages: ["openbb_agent_server"]
processors:
  - type: filter
    documented_only: false
    exclude_private: true
    exclude_special: true
    do_not_filter_modules: true
    skip_empty_modules: false
    expression: "default() and obj.__class__.__name__ != 'Indirection' and not (obj.__class__.__name__ == 'Variable' and not obj.docstring)"
renderer:
  type: docusaurus
  docs_base_path: ${JSON.stringify(path.join(repoRoot, 'docs'))}
  relative_output_path: reference
  sidebar_top_level_label: null
`;
  fs.writeFileSync(path.join(tmp, 'pydoc-markdown.yml'), cfg);
  console.log('• running pydoc-markdown (uvx, isolated) on a roles-as-code copy…');
  execFileSync('uvx', ['pydoc-markdown@4.8.2'], {cwd: tmp, stdio: 'inherit', shell: true});
} finally {
  fs.rmSync(tmp, {recursive: true, force: true});
}

// Lift docs/reference/openbb_agent_server/* up one level.
const pkgDir = path.join(refDir, 'openbb_agent_server');
if (fs.existsSync(pkgDir)) {
  for (const entry of fs.readdirSync(pkgDir)) {
    fs.renameSync(path.join(pkgDir, entry), path.join(refDir, entry));
  }
  fs.rmdirSync(pkgDir);
}
fs.rmSync(path.join(refDir, 'sidebar.json'), {force: true});

// Convert Sphinx cross-reference roles to plain inline code:
//   :class:`~a.b.Name` -> `Name`   :meth:`show_tvchart` -> `show_tvchart`
function stripSphinxRoles(text) {
  return text.replace(/:[a-zA-Z]+:`(~?)([^`]+)`/g, (_m, tilde, target) =>
    '`' + (tilde ? target.split('.').pop() : target) + '`',
  );
}

// Walk the tree: clean every page, __init__.md -> index.md, fix labels.
function walk(dir) {
  for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(full);
      continue;
    }
    if (entry.name.endsWith('.md')) {
      const cleaned = stripSphinxRoles(fs.readFileSync(full, 'utf8'));
      fs.writeFileSync(full, cleaned);
    }
    if (entry.name === '__init__.md') {
      const indexPath = path.join(dir, 'index.md');
      let body = fs.readFileSync(full, 'utf8');
      // sidebar_label "__init__" -> the package folder name.
      const label = path.basename(dir) === 'reference' ? 'openbb_agent_server' : path.basename(dir);
      body = body.replace(/^sidebar_label:.*$/m, `sidebar_label: ${label}`);
      fs.writeFileSync(indexPath, body);
      fs.rmSync(full);
    } else if (path.basename(entry.name, '.md') === path.basename(dir)) {
      // A module named the same as its package (e.g. app/app.md) matches
      // Docusaurus's folder-index convention and collides with the
      // package's index.md. Pin an explicit slug so it keeps its own
      // route instead of claiming the folder route.
      const rel = path.relative(refDir, full).replace(/\\/g, '/').replace(/\.md$/, '');
      let body = fs.readFileSync(full, 'utf8');
      if (!/^slug:/m.test(body)) {
        // Generated files may have CRLF endings — match either.
        body = body.replace(/^---\r?\n/, `---\nslug: /reference/${rel}\n`);
        fs.writeFileSync(full, body);
      }
    }
  }
}
walk(refDir);

// Replace the package-root page (the bare __init__ docstring) with a
// curated API Reference landing.
const landing = `---
sidebar_label: API Reference
title: API Reference
---

# API Reference

These pages are **generated automatically** from the \`openbb_agent_server\`
package docstrings (via \`pydoc-markdown\`, parsing the source statically),
so they always match the installed code. Regenerate them with
\`npm run gen-api\` from \`website/\`.

For narrative contracts and architecture that signatures can't express,
see [Explanation](../explanation/index.md); for task-oriented walkthroughs,
see the [User Guides](../guides/index.md).

## Top-level packages

| Package | What's in it |
| --- | --- |
| [\`app\`](app/index.md) | FastAPI surface — app factory, router, settings, config cascade |
| [\`runtime\`](runtime/index.md) | The agent loop, plugin ABCs, context, registry, jobs, stores |
| [\`protocol\`](protocol/index.md) | Wire types — SSE schemas, the DeepAgents→OpenBB adapter |
| [\`plugins\`](plugins/index.md) | Built-in auth, models, tools, middleware, sub-agents, checkpointers |
| [\`memory\`](memory/index.md) | Vector memory + retrieval pipeline |
| [\`persistence\`](persistence/index.md) | History / usage / artifact stores |
| [\`acp\`](acp/index.md) | PyWry chat shim + the live canvas |
| [\`observability\`](observability/index.md) | Structured logging |
| [\`prompts\`](prompts/index.md) | Packaged system prompts |

Use the sidebar to browse every module.
`;
fs.writeFileSync(path.join(refDir, 'index.md'), landing);

console.log('• API reference written to docs/reference/');
