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
