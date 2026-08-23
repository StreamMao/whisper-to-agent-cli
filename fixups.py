import difflib
import re

# File extensions the model often hears as the 'X的YYY' form when a dot is
# spoken (e.g. "readme的md" -> readme.md). This list lets ANY such filename be
# corrected, not just hard-coded examples. Kept lowercase.
KNOWN_EXTENSIONS = {
    "md", "py", "txt", "json", "yaml", "yml", "toml", "ini", "cfg", "log",
    "csv", "tsv", "env", "bat", "cmd", "sh", "ps1", "exe", "dll", "msi",
    "doc", "docx", "xls", "xlsx", "ppt", "pptx", "pdf", "zip", "tar", "gz",
    "7z", "rar", "png", "jpg", "jpeg", "gif", "svg", "webp", "ico", "bmp",
    "mp3", "mp4", "wav", "flac", "ogg", "mkv", "mov", "avi",
    "html", "htm", "css", "scss", "less", "js", "ts", "tsx", "jsx", "vue",
    "php", "rb", "go", "rs", "java", "kt", "swift", "c", "cpp", "cc", "h",
    "hpp", "cs", "sql", "ipynb", "pth", "onnx", "pt", "lock", "crt", "pem",
}


def post_fix(text: str, words: tuple = ()) -> str:
    if not text:
        return text

    # 1) Collapse spaces around dots inside ASCII tokens: "README . md" -> "README.md"
    text = re.sub(r"(?<=[A-Za-z0-9])\s*\.\s*(?=[A-Za-z0-9])", ".", text)

    # 2) The '的' spoken for an extension dot: "readme的md" -> "readme.md",
    #    and "readme的MDFile" -> "readme.md file". Only fires when the token
    #    after 的 is a known extension (optionally followed by the word
    #    "file"), so genuine Mandarin like "Python的API" is left untouched.
    def _dot_for_de(m):
        ext = m.group(2).lower()
        if ext in KNOWN_EXTENSIONS:
            tail = " file" if m.group(3) else ""
            return f"{m.group(1)}.{ext}{tail}"
        return m.group(0)

    text = re.sub(
        r"([A-Za-z0-9_.-]+)的([A-Za-z]+?)([Ff]ile)?(?![A-Za-z0-9])",
        _dot_for_de,
        text,
    )

    # 3) Normalize ASCII extensions to lowercase: "file.PY" -> "file.py"
    text = re.sub(
        r"(\.)([A-Za-z0-9_-]{1,8})(?![A-Za-z0-9])",
        lambda m: m.group(1) + m.group(2).lower(),
        text,
    )

    # 4) Configured hotwords: normalize case-insensitive token matches to the
    #    user's spelling (e.g. "readme" -> "README", "md" -> "md"). Extensible
    #    via whisper.hotwords in user_settings.yaml.
    for word in words:
        if not word:
            continue
        text = re.sub(
            r"(?<![A-Za-z0-9_])" + re.escape(word) + r"(?![A-Za-z0-9_])",
            word,
            text,
            flags=re.IGNORECASE,
        )

    # 5) Hotword-driven filename repair: for each configured filename (one with
    #    a dot), a token with the same basename but a slightly mangled extension
    #    is corrected to the hotword (e.g. hotword "README.md" fixes
    #    "readme.mdf" -> "README.md"). Unrelated extensions are left alone.
    for word in words:
        base, _, ext = word.rpartition(".")
        if not ext:
            continue

        def _repair(mo):
            token_ext = mo.group(0).rpartition(".")[2].lower()
            if difflib.SequenceMatcher(None, token_ext, ext.lower()).ratio() >= 0.75:
                return word
            return mo.group(0)

        text = re.sub(
            r"(?<![A-Za-z0-9_])" + re.escape(base) + r"\.[A-Za-z0-9_-]{1,8}(?![A-Za-z0-9])",
            _repair,
            text,
            flags=re.IGNORECASE,
        )
    return text