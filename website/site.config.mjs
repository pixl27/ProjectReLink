export function getSiteConfig(env = process.env) {
  const origin = new URL(env.DOCS_SITE_URL || 'https://spacefarergames.github.io');
  if (origin.protocol !== 'https:' || origin.pathname !== '/' || origin.search || origin.hash) {
    throw new Error('DOCS_SITE_URL must be an HTTPS origin, without a path, query, or fragment.');
  }
  const repository = env.DOCS_SOURCE_REPOSITORY || 'spacefarergames/RedTunic';
  if (!/^[\w.-]+\/[\w.-]+$/.test(repository)) throw new Error('Invalid DOCS_SOURCE_REPOSITORY.');
  const branch = env.DOCS_SOURCE_BRANCH || 'main';
  const ref = branch.split('/').map(encodeURIComponent).join('/');
  return { origin: origin.origin, base: '/RedTunic/', editBase: `https://github.com/${repository}/edit/${ref}/` };
}

export const siteConfig = getSiteConfig();
