def looks_like_greeting(s: str) -> bool:
    s = s.lower().strip()
    return s in {"hi","hello","hey"} or s.startswith(("hi ","hello "))
