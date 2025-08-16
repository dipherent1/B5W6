import re
import unicodedata

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # normalize unicode
    s = unicodedata.normalize("NFKC", s)
    # lower
    s = s.lower()
    # remove boilerplate patterns (add more as discovered)
    boilerplate_patterns = [
        r"i am writing to file a complaint",
        r"dear (sir|madam|team)",
    ]
    for pat in boilerplate_patterns:
        s = re.sub(pat, " ", s)
    # strip urls/emails
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[\w\.-]+@[\w\.-]+", " ", s)
    # keep basic punctuation & words
    s = re.sub(r"[^a-z0-9\s\.,;:!\?\-']", " ", s)
    # collapse spaces
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s
