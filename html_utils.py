from bs4 import BeautifulSoup
from typing import Dict


def extract_zoned_text(html: str) -> Dict[str, str]:
    """
    Extract text from potentially broken HTML into zones:
      - title: <title>
      - headers: <h1><h2><h3>
      - bold: <b><strong>
      - body: visible text from the whole page
    if parsing fails, fall back to plain text.
    """
    if not html:
        return {"title": "", "headers": "", "bold": "", "body": ""}

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        # fallback parser
        soup = BeautifulSoup(html, "html.parser")

    # Title
    title_text = ""
    try:
        if soup.title and soup.title.string:
            title_text = soup.title.get_text(" ", strip=True)
    except Exception:
        title_text = ""

    # Headers
    header_parts = []
    for tag in soup.find_all(["h1", "h2", "h3"]):
        try:
            header_parts.append(tag.get_text(" ", strip=True))
        except Exception:
            continue
    headers_text = " ".join(header_parts)

    # Bold
    bold_parts = []
    for tag in soup.find_all(["b", "strong"]):
        try:
            bold_parts.append(tag.get_text(" ", strip=True))
        except Exception:
            continue
    bold_text = " ".join(bold_parts)

    # Body text (remove scripts/styles)
    for bad in soup(["script", "style", "noscript"]):
        try:
            bad.decompose()
        except Exception:
            pass
    try:
        body_text = soup.get_text(" ", strip=True)
    except Exception:
        body_text = str(soup)

    return {"title": title_text, "headers": headers_text, "bold": bold_text, "body": body_text}