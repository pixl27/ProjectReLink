import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { getSiteConfig } from '../site.config.mjs';
import { assertReviewedSources } from '../scripts/verify-compatibility-sources.mjs';

test('hosting defaults to upstream and supports fork preview overrides', () => {
  const upstream = getSiteConfig({});
  assert.equal(upstream.origin, 'https://spacefarergames.github.io');
  assert.equal(upstream.editBase, 'https://github.com/spacefarergames/RedTunic/edit/main/');
  const fork = getSiteConfig({ DOCS_SITE_URL: 'https://bibigurl-yum.github.io', DOCS_SOURCE_REPOSITORY: 'bibigurl-yum/Hookshot', DOCS_SOURCE_BRANCH: 'docs/starlight-preview' });
  assert.equal(fork.editBase, 'https://github.com/bibigurl-yum/Hookshot/edit/docs/starlight-preview/');
  assert.throws(() => getSiteConfig({ DOCS_SITE_URL: 'https://example.com/subdirectory' }));
});

test('README requirements and GPU-status changes require a calculator review', async () => {
  const readme = await readFile(new URL('../../readme.md', import.meta.url), 'utf8');
  const review = JSON.parse(await readFile(new URL('../compatibility-review.json', import.meta.url), 'utf8'));
  assert.doesNotThrow(() => assertReviewedSources(readme, review));
  for (const [before, after] of [['v131072', 'v131073'], ['headless Windows 11 smoke test', 'Windows 11 gameplay test']]) {
    assert(readme.includes(before));
    assert.throws(() => assertReviewedSources(readme.replace(before, after), review), /changed since compatibility review/);
  }
  assert.doesNotThrow(() => assertReviewedSources(readme + '\nUnrelated footer edit.\n', review));
});
