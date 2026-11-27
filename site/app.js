document.querySelectorAll('.video-button').forEach(button=>button.addEventListener('click',()=>{
  const mount=button.closest('[data-video]');
  const frame=document.createElement('iframe');
  frame.src=`https://www.youtube-nocookie.com/embed/${mount.dataset.video}?autoplay=1`;
  frame.title=button.getAttribute('aria-label');
  frame.allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share';
  frame.allowFullscreen=true;
  frame.referrerPolicy='strict-origin-when-cross-origin';
  mount.replaceChildren(frame);
}));
const platform=/Linux|Android/i.test(navigator.userAgent)?'linux':/Windows/i.test(navigator.userAgent)?'windows':null;
if(platform)document.querySelector(`[data-os="${platform}"]`)?.classList.add('recommended');

const motion = matchMedia('(prefers-reduced-motion: reduce)');
if (!motion.matches && 'IntersectionObserver' in window) {
  const layers = [...document.querySelectorAll('[data-parallax]')].map(section => ({
    section, pieces: [...section.querySelectorAll('[data-drift]')],
  }));
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.add('in-view');
      observer.unobserve(entry.target);
    }
  });
  document.querySelectorAll('[data-reveal]').forEach(section => observer.observe(section));
  document.documentElement.classList.add('motion-ready');
  let queued = false;
  function scheduleDrift() {
    if (queued || motion.matches) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      const height = innerHeight;
      const offsets = layers.map(({ section }) => {
        const box = section.getBoundingClientRect();
        if (box.bottom < 0 || box.top > height) return null;
        return Math.max(-26, Math.min(26, (height / 2 - box.top - box.height / 2) * .055));
      });
      layers.forEach(({ pieces }, i) => {
        if (offsets[i] === null) return;
        pieces.forEach((piece, j) => {
          piece.style.setProperty('--drift', offsets[i] * (j % 2 ? -.8 : 1) + 'px');
        });
      });
    });
  }
  addEventListener('scroll', scheduleDrift, { passive: true });
  addEventListener('resize', scheduleDrift);
  scheduleDrift();
}
