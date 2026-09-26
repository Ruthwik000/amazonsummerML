"""Text normalisation for business names and addresses.

Everything here is rule-based or learned from the provided training data only
(the Indic-script -> Latin token dictionary is learned from train ground truth
pairs by `learn_translit.py`).  No external data or services are used.
"""
import json
import os
import re
import unicodedata
from functools import lru_cache

from rapidfuzz.distance import OSA

# --------------------------------------------------------------------------
# Indic script handling
# --------------------------------------------------------------------------
INDIC_RE = re.compile(r"[ऀ-ൿ]")
INDIC_TOKEN_RE = re.compile(r"[ऀ-ൿ‌‍]+")

# Rule-based romaniser (fallback for tokens missing from the learned dictionary).
# All Brahmic blocks share the Devanagari layout, so map each code point to its
# Devanagari equivalent (offset within the 0x80 block) and romanise that.
_DEV_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
    0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
    0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
    0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh", 0x35: "v", 0x36: "sh", 0x37: "sh",
    0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r",
    0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_DEV_VOW = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
            0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au"}
_DEV_SIGN = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x45: "e",
             0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au",
             0x57: "au", 0x62: "l", 0x63: "l"}
_MAL_CHILLU = {0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k"}


def romanize(tok: str) -> str:
    out = []
    pending = False  # consonant waiting for vowel
    for ch in tok:
        cp = ord(ch)
        if cp in _MAL_CHILLU:
            if pending:
                out.append("a")
            out.append(_MAL_CHILLU[cp]); pending = False; continue
        if not (0x0900 <= cp <= 0x0D7F):
            continue
        off = (cp - 0x0900) % 0x80
        if off == 0x3C:  # nukta
            continue
        if off in _DEV_CONS:
            if pending:
                out.append("a")
            out.append(_DEV_CONS[off]); pending = True
        elif off in _DEV_SIGN:
            out.append(_DEV_SIGN[off]); pending = False
        elif off == 0x4D:  # virama
            pending = False
        elif off in _DEV_VOW:
            if pending:
                out.append("a")
            out.append(_DEV_VOW[off]); pending = False
        elif off in (0x01, 0x02):  # candrabindu / anusvara
            if pending:
                out.append("a"); pending = False
            out.append("n")
        elif off == 0x03:
            if pending:
                out.append("a"); pending = False
            out.append("h")
    # schwa deletion at word end (pending consonant => no trailing 'a')
    return "".join(out)


_DICT = None


def _load_dict():
    global _DICT
    if _DICT is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "indic_token_dict.json")
        _DICT = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    return _DICT


def translit_text(s: str) -> str:
    """Replace every Indic-script token by its Latin equivalent."""
    if not s or not INDIC_RE.search(s):
        return s
    d = _load_dict()

    def rep(m):
        t = m.group(0).replace("‌", "").replace("‍", "")
        if t in d:
            return d[t]
        t2 = m.group(0)
        if t2 in d:
            return d[t2]
        return romanize(t)
    return INDIC_TOKEN_RE.sub(rep, s)


# --------------------------------------------------------------------------
# Canonical token maps
# --------------------------------------------------------------------------
NAME_CANON = {
    "private": "pvt", "pvt": "pvt", "prv": "pvt", "pte": "pvt", "priv": "pvt",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd", "lt": "ltd",
    "corporation": "corp", "corp": "corp", "corpn": "corp", "corporate": "corp",
    "incorporated": "inc", "inc": "inc", "incorporation": "inc",
    "company": "co", "co": "co", "cos": "co", "compagnie": "co", "cie": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pc": "pc", "pllc": "pllc",
    "international": "intl", "intl": "intl", "internationale": "intl",
    "manufacturing": "mfg", "mfg": "mfg",
    "associates": "assoc", "assoc": "assoc", "associate": "assoc", "assocs": "assoc",
    "brothers": "bros", "bros": "bros",
    "group": "group", "grp": "group", "groupe": "group",
    "holdings": "holdings", "hldgs": "holdings", "holding": "holdings",
    "management": "mgmt", "mgmt": "mgmt", "mgt": "mgmt",
    "national": "natl", "natl": "natl",
    "center": "center", "centre": "center", "ctr": "center", "cntr": "center",
    "services": "services", "svcs": "services", "service": "services", "svc": "services",
    "technologies": "technologies", "technology": "technologies", "tech": "technologies",
    "industries": "industries", "inds": "industries", "industry": "industries",
    "enterprises": "enterprises", "ent": "enterprises", "enterprise": "enterprises",
    "solutions": "solutions", "soln": "solutions", "solution": "solutions",
    "university": "univ", "univ": "univ",
    "saint": "saint", "st": "saint", "ste": "sainte", "sainte": "sainte",
    "sri": "shree", "sree": "shree", "shri": "shree", "shree": "shree", "shrí": "shree",
    "jai": "jai", "jay": "jai", "jaya": "jai",
    "lakshmi": "laxmi", "laxmi": "laxmi", "laksmi": "laxmi",
    "shiv": "shiva", "shiva": "shiva",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "ei": "ei",
    "sci": "sci", "snc": "snc", "scop": "scop", "selarl": "selarl", "scm": "scm",
    "selas": "selas", "gie": "gie", "scea": "scea", "societe": "societe", "soc": "societe",
    "and": "and", "et": "and", "und": "and",
    "the": "the", "of": "of", "le": "le", "la": "la", "les": "les", "de": "de", "du": "du", "des": "des",
    "dds": "dds", "md": "md", "cpa": "cpa", "pa": "pa",
    "opc": "opc",
}
# tokens that carry (almost) no identity -> removed from the "core" name
NAME_STOP = {
    "pvt", "ltd", "corp", "inc", "co", "llc", "llp", "lp", "plc", "pc", "pllc", "the", "and",
    "of", "sarl", "sas", "sasu", "eurl", "sa", "ei", "sci", "snc", "scop", "selarl", "scm",
    "opc", "le", "la", "les", "de", "du", "des", "d", "l", "a", "an", "dba", "aka", "fka",
    "et", "fils", "null", "na", "none", "selas", "gie", "scea",
}
LEGAL = {"pvt", "ltd", "corp", "inc", "co", "llc", "llp", "lp", "plc", "pc", "pllc", "sarl",
         "sas", "sasu", "eurl", "sa", "ei", "sci", "snc", "scop", "selarl", "opc", "selas", "gie", "scea"}
# legal suffixes written without separator ("Pvt.Ltd" -> "pvtltd")
GLUED = {
    "pvtltd": ["pvt", "ltd"], "privateltd": ["pvt", "ltd"], "privatelimited": ["pvt", "ltd"],
    "pvtlimited": ["pvt", "ltd"], "coltd": ["co", "ltd"], "coinc": ["co", "inc"], "llcinc": ["llc", "inc"],
}
# Long generic words whose typos ("Coropration", "Limted", "Inorporated") should still be
# recognised.  A token that is not already canonical is mapped to one of these when it has the
# same first letter and a small optimal-string-alignment distance (transpositions count as 1).
NAME_FUZZY = {w: NAME_CANON[w] for w in (
    "private", "limited", "corporation", "incorporated", "company", "international", "manufacturing",
    "associates", "brothers", "holdings", "management", "technologies", "industries", "enterprises",
    "solutions", "services", "university", "incorporation", "corporate")}

ADDR_CANON_COMMON = {
    "street": "st", "st": "st", "str": "st", "strt": "st",
    "road": "rd", "rd": "rd", "raod": "rd", "roda": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave", "aven": "ave", "avn": "ave",
    "drive": "dr", "dr": "dr", "drv": "dr",
    "boulevard": "blvd", "blvd": "blvd", "boul": "blvd", "bd": "blvd", "bvd": "blvd",
    "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct", "crt": "ct",
    "place": "pl", "pl": "pl", "plc": "pl",
    "square": "sq", "sq": "sq",
    "highway": "hwy", "hwy": "hwy", "hiway": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy", "pky": "pkwy",
    "circle": "cir", "cir": "cir", "circ": "cir",
    "terrace": "ter", "ter": "ter", "terr": "ter",
    "trail": "trl", "trl": "trl",
    "point": "pt", "pt": "pt",
    "mount": "mt", "mt": "mt", "mountain": "mtn", "mtn": "mtn",
    "fort": "ft", "ft": "ft",
    "north": "n", "n": "n", "south": "s", "s": "s", "so": "s", "east": "e", "e": "e",
    "west": "w", "w": "w", "northeast": "ne", "ne": "ne", "northwest": "nw", "nw": "nw",
    "southeast": "se", "se": "se", "southwest": "sw", "sw": "sw",
    "suite": "ste", "ste": "ste", "apartment": "apt", "apt": "apt", "floor": "fl", "fl": "fl",
    "flr": "fl", "building": "bldg", "bldg": "bldg", "bld": "bldg", "unit": "unit",
    "way": "way", "wy": "way", "expressway": "expy", "expy": "expy", "freeway": "fwy",
    "fwy": "fwy", "route": "rte", "rte": "rte", "rt": "rte", "plaza": "plz", "plz": "plz",
    "center": "ctr", "centre": "ctr", "ctr": "ctr", "heights": "hts", "hts": "hts",
    "junction": "jct", "jct": "jct", "crossing": "xing", "xing": "xing",
    "near": "near", "nr": "near", "opp": "opposite", "opposite": "opposite",
    "cross": "cross", "main": "main", "nagar": "nagar", "ngr": "nagar",
    "sector": "sector", "sec": "sector", "phase": "phase", "ph": "phase",
    "ground": "ground", "grd": "ground", "gf": "ground",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "apartments": "apts", "apts": "apts", "complex": "complex", "cmplx": "complex",
    "industrial": "indl", "indl": "indl", "estate": "estate", "est": "estate",
    "colony": "colony", "col": "colony", "post": "po", "po": "po", "box": "box",
    "district": "dist", "dist": "dist", "dt": "dist", "taluk": "taluk", "tq": "taluk",
    "village": "vill", "vill": "vill", "vil": "vill", "city": "city",
    "bengaluru": "bangalore", "bangalore": "bangalore", "bombay": "mumbai", "mumbai": "mumbai",
    "gurugram": "gurgaon", "gurgaon": "gurgaon", "trivandrum": "thiruvananthapuram",
    "calcutta": "kolkata", "madras": "chennai", "poona": "pune", "odisha": "orissa",
    "keralam": "kerala", "ahmadabad": "ahmedabad", "vishakhapatnam": "visakhapatnam",
    "vizag": "visakhapatnam", "mysuru": "mysore", "mangaluru": "mangalore", "cochin": "kochi",
    "belagavi": "belgaum", "kozhikode": "calicut", "calicut": "calicut",
    "number": "", "no": "", "nos": "", "num": "", "h": "", "hno": "", "house": "", "door": "",
    "plot": "", "flat": "", "shop": "", "null": "", "na": "", "none": "", "nil": "",
}
ADDR_CANON_FR = {
    "rue": "rue", "r": "rue", "avenue": "ave", "av": "ave", "ave": "ave", "avn": "ave",
    "boulevard": "blvd", "bd": "blvd", "bld": "blvd", "blvd": "blvd", "boul": "blvd",
    "chemin": "chemin", "ch": "chemin", "chem": "chemin", "che": "chemin",
    "impasse": "impasse", "imp": "impasse", "place": "pl", "pl": "pl",
    "route": "rte", "rte": "rte", "allee": "allee", "all": "allee", "al": "allee",
    "square": "sq", "sq": "sq", "faubourg": "fbg", "fbg": "fbg", "fg": "fbg",
    "cours": "crs", "crs": "crs", "quai": "quai", "q": "quai", "qu": "quai",
    "saint": "saint", "st": "saint", "sainte": "sainte", "ste": "sainte",
    "general": "gen", "gen": "gen", "gal": "gen", "docteur": "dr", "dr": "dr",
    "professeur": "prof", "prof": "prof", "marechal": "mal", "mal": "mal",
    "president": "pres", "pres": "pres", "residence": "res", "res": "res",
    "batiment": "bat", "bat": "bat", "bis": "bis", "b": "bis", "ter": "ter", "t": "ter",
    "lieu": "lieu", "lieudit": "lieudit", "zone": "zone", "za": "za", "zi": "zi", "zac": "zac",
    "no": "", "n": "", "numero": "", "null": "", "na": "",
}
ADDR_FUZZY_COMMON = {w: ADDR_CANON_COMMON[w] for w in (
    "street", "avenue", "boulevard", "highway", "parkway", "circle", "terrace", "apartment", "building",
    "sector", "colony", "village", "district", "industrial", "opposite", "junction", "complex", "apartments")}
ADDR_FUZZY_FR = {w: ADDR_CANON_FR[w] for w in ("avenue", "boulevard", "chemin", "impasse", "faubourg", "residence")}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va",
    "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "keralam": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "orissa": "od", "odisha": "od",
    "or": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "tamilnadu": "tn", "telangana": "tg", "ts": "tg", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "uttaranchal": "uk", "ut": "uk", "west bengal": "wb", "delhi": "dl",
    "new delhi": "dl", "nct of delhi": "dl", "jammu and kashmir": "jk", "jammu & kashmir": "jk",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py", "dadra and nagar haveli": "dn",
    "daman and diu": "dd", "ladakh": "la", "lakshadweep": "ld", "andaman and nicobar islands": "an",
}
# French regions (and a few departments that appear as their own address component).  City
# names such as "Paris" or "Lyon" are deliberately NOT here: removing them as a "state" would
# drop the city token from the address.
FR_REGIONS = {
    "hauts de france": "hdf", "nord": "hdf", "pas de calais": "hdf", "somme": "hdf", "aisne": "hdf", "oise": "hdf",
    "nord pas de calais": "hdf", "picardie": "hdf",
    "nouvelle aquitaine": "naq", "gironde": "naq", "landes": "naq", "aquitaine": "naq", "limousin": "naq",
    "poitou charentes": "naq",
    "pays de la loire": "pdl", "loire atlantique": "pdl", "vendee": "pdl",
    "ile de france": "idf", "idf": "idf",
    "auvergne rhone alpes": "ara", "rhone alpes": "ara", "auvergne": "ara",
    "bourgogne franche comte": "bfc", "bourgogne": "bfc", "franche comte": "bfc",
    "bretagne": "bre", "brittany": "bre",
    "centre val de loire": "cvl",
    "corse": "cor", "corsica": "cor",
    "grand est": "ges", "alsace": "ges", "lorraine": "ges", "champagne ardenne": "ges",
    "normandie": "nor", "normandy": "nor", "haute normandie": "nor", "basse normandie": "nor",
    "occitanie": "occ", "languedoc roussillon": "occ", "midi pyrenees": "occ",
    "provence alpes cote d azur": "pac", "provence alpes cote dazur": "pac", "paca": "pac",
    "provence": "pac", "cote d azur": "pac",
    "guadeloupe": "gp", "martinique": "mq", "guyane": "gf", "la reunion": "re", "reunion": "re", "mayotte": "yt",
}
STATE_MAPS = {"US": US_STATES, "India": IN_STATES, "France": FR_REGIONS}
# abbreviations are canonical themselves
for _m in (US_STATES, IN_STATES):
    for _v in list(_m.values()):
        _m.setdefault(_v, _v)
# postal-code shape per country (unknown countries fall back to a generic 4-6 digit rule)
POSTAL_RE = {"US": re.compile(r"\b(\d{5})(?:-\d{4})?\b"), "India": re.compile(r"\b(\d{3})\s?(\d{3})\b"),
             "France": re.compile(r"\b(\d{5})\b")}
POSTAL_GENERIC = re.compile(r"\b(\d{4,6})\b")
ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th)$")
FLOOR_WORDS = {"floor", "fl", "flr", "etage", "level"}

LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b", "@": "a", "$": "s"})

DBA_RE = re.compile(
    r"\s+(?:doing business as|d\s*/\s*b\s*/\s*a|d\.b\.a\.?|dba|t\s*/\s*a|trading as|a\s*/\s*k\s*/\s*a|aka|"
    r"f\s*/\s*k\s*/\s*a|fka|formerly known as|formerly)\s*:?\s+", re.I)
DOMAIN_RE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:co\.in|com|net|org|in|fr|co|biz|info|us|io)\b", re.I)


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


_non_alnum = re.compile(r"[^a-z0-9]+")
_DOTTED_RE = re.compile(r"[a-z0-9]+(?:\.+[a-z0-9]+)+\.*|(?<![a-z0-9])[a-z0-9]{1,3}\.{2,}")
# legal acronyms a dotted abbreviation may be a (typo'd) spelling of: "S.A.Rrl." -> sarl
LEGAL_ACRONYMS = {w: NAME_CANON[w] for w in ("sarl", "eurl", "sasu", "sas", "selarl", "llc", "llp", "pllc", "ltd", "pvt", "inc", "corp")}


def _fuzzy_canon(t: str, targets: dict):
    """Map a typo of a long generic word to its canonical form (None if nothing is close enough)."""
    n = len(t)
    if n < 5 or t.isdigit():
        return None
    best, best_d = None, 3
    for w, c in targets.items():
        if w[0] != t[0] or abs(len(w) - n) > 2:
            continue
        lim = 1 if min(n, len(w)) < 8 else 2
        d = OSA.distance(t, w, score_cutoff=lim)
        if d <= lim and d < best_d:
            best, best_d = c, d
    return best


def _undot(m):
    w = m.group(0)
    parts = [x for x in w.split(".") if x]
    if any(len(x) > 3 for x in parts):
        return " ".join(parts)
    j = "".join(parts)
    if j in NAME_CANON or j in GLUED or len(j) < 3:
        return " " + j + " "
    near = [c for a, c in LEGAL_ACRONYMS.items() if abs(len(a) - len(j)) <= 1 and OSA.distance(j, a, score_cutoff=1) <= 1]
    return " " + (near[0] if near else j) + " "


@lru_cache(maxsize=500000)
def _canon_name_tok(t: str):
    if t in NAME_CANON:
        return (NAME_CANON[t],)
    if t in GLUED:
        return tuple(GLUED[t])
    f = _fuzzy_canon(t, NAME_FUZZY)
    return (f,) if f else (t,)


def _name_tokens(s: str):
    s = s.lower().replace("&", " and ").replace("+", " and ").replace("'", "").replace("’", "")
    # dotted abbreviations ("A.B.C.", "S.A.R.L.", garbled "Eu..R.L.", "Pvt.Ltd") lose their dots;
    # a dot inside a longer word separates tokens ("Dr.Reddys" -> "dr reddys")
    s = _DOTTED_RE.sub(_undot, s)
    s = s.replace(".", " ")
    toks = []
    for t in _non_alnum.split(s):
        if not t:
            continue
        if any(c.isalpha() for c in t) and any(c.isdigit() for c in t) and not ORDINAL_RE.match(t):
            t = t.translate(LEET)
        toks.extend(_canon_name_tok(t))
    return toks


_VOWELS = str.maketrans("", "", "aeiouy")


def skeleton(tok: str) -> str:
    """Typo / transliteration tolerant token key: first letter + consonants, doubles collapsed
    ("Beeacon" / "Beacon" -> "bcn", "Enetrprises" / "Enterprises" -> "entrprss")."""
    if not tok or tok.isdigit():
        return tok
    t = tok.replace("ph", "f").replace("ck", "k").replace("sh", "s").replace("th", "t").replace("kh", "k")
    t = t[0] + t[1:].translate(_VOWELS)
    out = [t[0]]
    for ch in t[1:]:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


def normalize_name(raw: str):
    """Return (full_tokens, core_tokens, alt_core_tokens, flags)."""
    s = translit_text(raw or "")
    s = strip_accents(s)
    is_domain = 0
    doms = DOMAIN_RE.findall(s)
    if doms:
        is_domain = 1
        # remove the domain part(s) from the string, but keep the label as a token
        s2 = DOMAIN_RE.sub(" ", s).replace("|", " ").strip()
        label = doms[0]
        s = (s2 + " " + label) if s2 else label
    s = s.replace("|", " ")
    s = re.sub(r"^\W+", "", s)
    parts = DBA_RE.split(s)
    is_dba = int(len(parts) > 1)
    if is_dba:
        main, alt = parts[-1], parts[0]
    else:
        main, alt = s, ""
    full = _name_tokens(main)
    core = [t for t in full if t not in NAME_STOP]
    alt_core = [t for t in _name_tokens(alt) if t not in NAME_STOP] if alt else []
    return full, core, alt_core, is_domain, is_dba


NUM_RE = re.compile(r"\d+")


def _postal_code(s: str, country: str) -> str:
    """Postal code: US ZIP, Indian PIN, French code postal, generic 4-6 digits for other countries.
    A number that opens the address ("12345 Main St", "No. 560001") is a house number, not a code."""
    rx = POSTAL_RE.get(country, POSTAL_GENERIC)
    first = NUM_RE.search(s)
    code = ""
    for m in rx.finditer(s):
        if first is not None and m.start() == first.start() and not re.search(r"[a-z]{3,}", s[:m.start()].replace("plot", "")):
            continue
        code = "".join(g for g in m.groups() if g)
    return code


def _state_split(ck: str, smap: dict):
    """(state_code, remaining_text) if the component is a state, optionally with a postal code
    ("il 62704", "karnataka 560001", "tamil nadu - 600001") or preceded by a city ("austin tx 78701")."""
    if not smap or not ck:
        return None, ck
    if ck in smap:
        return smap[ck], ""
    words = ck.split()
    core = [w for w in words if not w.isdigit()]
    had_code = len(core) < len(words)
    if not core:
        return None, ck
    if " ".join(core) in smap and had_code:
        return smap[" ".join(core)], ""
    # trailing state after a city ("springfield il 62704", "pune maharashtra 411001"); only
    # trusted when a postal code follows or the state is a multi-word name ("... tamil nadu")
    for k in (3, 2, 1):
        if len(core) > k:
            cand = " ".join(core[-k:])
            if cand in smap and (had_code or k >= 2):
                return smap[cand], " ".join(core[:-k])
    return None, ck


def normalize_address(raw: str, country: str):
    """Return (tokens without state, numbers, state code, pin, city components, postal code, house number)."""
    s = translit_text(raw or "")
    s = strip_accents(s).lower()
    smap = STATE_MAPS.get(country, {})
    canon = dict(ADDR_CANON_COMMON)
    fuzzy = ADDR_FUZZY_COMMON
    if country == "France":
        canon = {**{k: v for k, v in ADDR_CANON_COMMON.items() if k not in ("st", "ste", "dr", "n", "s", "e", "w")}, **ADDR_CANON_FR}
        fuzzy = {**ADDR_FUZZY_COMMON, **ADDR_FUZZY_FR}
    comps = [c.strip() for c in s.split(",")]
    comps = [c for c in comps if c and c not in ("null", "<null>", "n/a", "na", "none", "nil", "-")]
    pcode = _postal_code(s, country)
    # the state is taken from the right-most component that looks like one, so "new york, ny 10001"
    # keeps "new york" as the city and "washington, dc" keeps "washington"
    state, state_idx, state_rest = "", -1, ""
    for i in range(len(comps) - 1, -1, -1):
        ck = " ".join(re.sub(r"[^a-z0-9& ]+", " ", comps[i]).replace("&", " and ").split())
        st, rest = _state_split(ck, smap)
        if st:
            state, state_idx, state_rest = st, i, rest
            break
    toks = []
    nums = []
    pin = ""
    city_comps = []
    hnum = ""
    for i, c in enumerate(comps):
        if i == state_idx:
            # keep the digits and any city words of a "city st 12345" component, drop the state words
            c = state_rest + " " + " ".join(NUM_RE.findall(c))
            c = c.strip()
            if not c:
                continue
        for n in NUM_RE.findall(c):
            if len(n) == 6 and country == "India":
                pin = n
            n2 = n.lstrip("0") or "0"
            nums.append(n2)
        ctoks = []
        raw_toks = [t for t in _non_alnum.split(c.replace("'", "")) if t]
        for j, t in enumerate(raw_toks):
            if t.isdigit():
                t = t.lstrip("0") or "0"
                # house / plot number = first number that is neither the postal code nor a floor number
                nxt = raw_toks[j + 1] if j + 1 < len(raw_toks) else ""
                if not hnum and t != (pcode.lstrip("0") or "0") and nxt not in FLOOR_WORDS:
                    hnum = t
            if t in canon:
                t = canon[t]
            elif not t.isdigit():
                t = _fuzzy_canon(t, fuzzy) or t
            if t:
                ctoks.append(t)
        toks.extend(ctoks)
        if not any(ch.isdigit() for ch in c) and ctoks:
            city_comps.append(" ".join(ctoks))
    if country == "India" and not pin and len(pcode) == 6:
        pin = pcode
    return toks, nums, state, pin, city_comps, pcode, hnum
