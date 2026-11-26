"""Check the buildless project website's local links and images before Pages publication."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote

ROOT = Path(__file__).resolve().parents[1] / 'site'


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.images = set(), [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag in ('a', 'link') and 'href' in attrs:
            self.links.append(attrs['href'])
        if tag in ('img', 'script') and 'src' in attrs:
            self.links.append(attrs['src'])
        if tag == 'img':
            self.images.append(attrs)


def main():
    page = Page()
    page.feed((ROOT / 'index.html').read_text(encoding='utf8'))
    for link in page.links:
        url = urlsplit(link)
        if url.scheme or url.netloc:
            continue
        if url.path:
            path = (ROOT / unquote(url.path)).resolve()
            assert path.is_relative_to(ROOT.resolve()) and path.is_file(), link
        elif url.fragment:
            assert unquote(url.fragment) in page.ids, link
    assert page.images and all('alt' in image for image in page.images)
    print(f'PASS: {len(page.links)} website references, all local assets and section anchors resolve; images have alt text')


if __name__ == '__main__':
    main()
