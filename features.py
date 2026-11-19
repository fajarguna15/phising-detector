"""Ekstraksi fitur leksikal dari sebuah URL (tanpa akses jaringan)."""
import math
import re
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "update", "secure", "account", "bank", "confirm",
    "password", "paypal", "free", "bonus", "wallet", "support", "billing", "webscr",
]
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "cutt.ly", "rebrand.ly"}
IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {c: s.count(c) for c in set(s)}
    return -sum((n / len(s)) * math.log2(n / len(s)) for n in freq.values())


def normalize(url: str) -> str:
    url = str(url).strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


def get_hostname(url: str) -> str:
    try:
        return (urlparse(normalize(url)).hostname or "").lower()
    except ValueError:
        return ""


def extract_features(url: str) -> dict:
    raw = str(url).strip()
    u = normalize(raw)
    try:
        p = urlparse(u)
        host = (p.hostname or "").lower()
        port = p.port
    except ValueError:
        p, host, port = urlparse("http://invalid"), "", None
    path, query = p.path or "", p.query or ""
    parts = [x for x in host.split(".") if x]
    low = raw.lower()

    return {
        "url_length": len(raw),
        "hostname_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": raw.count("."),
        "num_hyphens": raw.count("-"),
        "num_underscores": raw.count("_"),
        "num_slashes": raw.count("/"),
        "num_digits": sum(c.isdigit() for c in raw),
        "digit_ratio": sum(c.isdigit() for c in raw) / max(len(raw), 1),
        "letter_ratio": sum(c.isalpha() for c in raw) / max(len(raw), 1),
        "num_special": sum(not c.isalnum() for c in raw),
        "num_percent": raw.count("%"),
        "num_params": query.count("&") + (1 if query else 0),
        "has_at": int("@" in raw),
        "has_ip": int(bool(IP_RE.match(host))),
        "is_https": int(p.scheme == "https"),
        "has_port": int(port is not None),
        "num_subdomains": max(len(parts) - 2, 0),
        "tld_length": len(parts[-1]) if parts else 0,
        "host_hyphens": host.count("-"),
        "host_digits": sum(c.isdigit() for c in host),
        "path_depth": len([x for x in path.split("/") if x]),
        "has_punycode": int("xn--" in host),
        "double_slash_in_path": int("//" in path),
        "is_shortener": int(host in SHORTENERS),
        "suspicious_words": sum(w in low for w in SUSPICIOUS_WORDS),
        "entropy_url": _entropy(raw),
        "entropy_host": _entropy(host),
    }


FEATURE_NAMES = list(extract_features("http://example.com").keys())
