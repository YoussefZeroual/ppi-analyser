import re
import logging
import importlib
logger = logging.getLogger(__name__)

TAG_STYLES = {
    "PPI":  ("color: green;",                                                              ""),
    "MD":   ("color: red;",                                                                " (MD)"),
    "EXP":  ("color: brown;",                                                              " (Expansion)"),
    "APP":  ("color: blue;",                                                               " (Appellatif)"),
    "POR":  ("text-decoration: underline; text-decoration-color: black; color: inherit;",  " (Portée)"),
    "MOD":  ("color: orange;",                                                             " (Modifieur)"),
}

POR_STYLE = "text-decoration: underline; text-decoration-color: black;"


# ---------------------------------------------------------------------------
# Shared text-cleaning helpers (mirrors excel.py logic)
# ---------------------------------------------------------------------------

def _normalize_ws(s: str) -> str:
    """Collapse all whitespace including literal \\n strings to a single space."""
    s = s.replace('\\n', ' ').replace('\\t', ' ')
    return re.sub(r'\s+', ' ', s).strip()


def _clean(s: str) -> str:
    """Strip XML-like tags and normalize whitespace (used on target text)."""
    return _normalize_ws(re.sub(r'<[^>]+>', '', s))


def _clean_content(s: str) -> str:
    """Strip XML tags, speaker labels [X], markdown * from justification content."""
    s = re.sub(r'<[^>]+>', '', s)
    s = re.sub(r'\[.*?\]', '', s)
    s = re.sub(r'\*+', '', s)
    return _normalize_ws(s)


def _anchor_pattern(content: str, n: int = 3) -> str:
    """First-n-words .*? last-n-words fuzzy pattern; exact for short content."""
    words = content.split()
    if len(words) <= n * 2:
        escaped = re.escape(content)
    else:
        head = r'\s+'.join(re.escape(w) for w in words[:n])
        tail = r'\s+'.join(re.escape(w) for w in words[-n:])
        escaped = head + r'.*?' + tail
    left_b  = r'\b' if re.match(r'\w', content[0])  else ''
    right_b = r'\b' if re.search(r'\w', content[-1]) else ''
    return left_b + escaped + right_b


def _build_clean_to_raw(text: str):
    """Map positions in clean (tag-stripped, ws-normalized) text back to raw text."""
    clean_chars = []
    inside_tag = False
    for i, ch in enumerate(text):
        if ch == '<':
            inside_tag = True
        if not inside_tag:
            clean_chars.append((i, ch))
        if ch == '>':
            inside_tag = False

    result_chars = []
    prev_was_space = False
    for raw_i, ch in clean_chars:
        if ch in ' \t\n\r':
            if not prev_was_space and result_chars:
                result_chars.append((raw_i, ' '))
            prev_was_space = True
        else:
            result_chars.append((raw_i, ch))
            prev_was_space = False

    if result_chars and result_chars[0][1] == ' ':
        result_chars = result_chars[1:]

    clean_text  = ''.join(ch for _, ch in result_chars)
    clean_to_raw = [raw_i for raw_i, _ in result_chars]
    return clean_text, clean_to_raw


# Known custom tags only — ignore HTML tags like <br>, <strong>, <span>, etc.
_KNOWN_TAGS_RE = r'<(PPI|MD|EXP|APP|POR|MOD)>'
_KNOWN_CLOSE_RE = r'</(PPI|MD|EXP|APP|POR|MOD)>'

def _is_inside_tag(text: str, start: int, end: int) -> bool:
    """Check if position is inside one of our custom XML tags (ignores HTML tags)."""
    before = text[:start]
    return len(re.findall(_KNOWN_TAGS_RE, before)) > len(re.findall(_KNOWN_CLOSE_RE, before))


def _tag_first_untagged_occurrence(text: str, tag_name: str, content: str) -> str:
    """
    Fuzzy-match content in clean version of text and wrap the first occurrence
    with <tag_name>. Expands boundaries to wrap any existing inner tags.
    """
    content = _normalize_ws(content)
    if not content:
        return text

    already_tagged = re.search(
        r'<' + re.escape(tag_name) + r'>.*?</' + re.escape(tag_name) + r'>',
        text, re.DOTALL
    )
    if already_tagged:
        return text

    pattern = _anchor_pattern(content)
    clean_text, clean_to_raw = _build_clean_to_raw(text)

    m = re.search(pattern, clean_text, re.DOTALL)
    if not m or m.end() > len(clean_to_raw) or m.start() >= len(clean_to_raw):
        return text

    raw_start = clean_to_raw[m.start()]
    raw_end   = clean_to_raw[m.end() - 1] + 1

    # Expand start backward past any enclosing custom XML tag (not HTML tags)
    if _is_inside_tag(text, raw_start, raw_end):
        opening = None
        for om in re.finditer(_KNOWN_TAGS_RE, text[:raw_start]):
            opening = om
        if opening is None:
            return text
        raw_start = opening.start()

    # Expand end forward past any cut-off closing tag
    if raw_end < len(text):
        tail    = text[raw_end:]
        close_m = re.match(r'[^<]*</[^>]+>', tail)
        if close_m and '<' not in tail[:close_m.start()]:
            raw_end += close_m.end()

    return (
        text[:raw_start]
        + f'<{tag_name}>'
        + text[raw_start:raw_end]
        + f'</{tag_name}>'
        + text[raw_end:]
    )


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _fix_crossed_tags(text: str) -> str:
    """
    Fix crossed/overlapping tags like <POR><EXP>...</POR></EXP>
    by ensuring inner tags close before outer tags.
    Strategy: for each closing tag, if it closes before its matching opener's
    inner tags have closed, move it to after those inner closers.
    Simple approach: reorder closing tags so inner tags always close first.
    """
    # Find all known tag names in order of appearance
    known = ['PPI', 'MD', 'EXP', 'APP', 'MOD', 'POR']
    stack = []
    result = []
    i = 0
    while i < len(text):
        # Check for opening tag
        m = re.match(r'<(PPI|MD|EXP|APP|POR|MOD)>', text[i:])
        if m:
            stack.append(m.group(1))
            result.append(m.group(0))
            i += len(m.group(0))
            continue
        # Check for closing tag
        m = re.match(r'</(PPI|MD|EXP|APP|POR|MOD)>', text[i:])
        if m:
            tag = m.group(1)
            if stack and stack[-1] != tag and tag in stack:
                # Close all inner tags first, then close this one, then reopen inner tags
                inner = []
                while stack and stack[-1] != tag:
                    inner.append(stack.pop())
                    result.append(f'</{inner[-1]}>')
                # Now close the target tag
                if stack:
                    stack.pop()
                result.append(f'</{tag}>')
                # Reopen inner tags
                for t in reversed(inner):
                    stack.append(t)
                    result.append(f'<{t}>')
            else:
                if stack and stack[-1] == tag:
                    stack.pop()
                result.append(m.group(0))
            i += len(m.group(0))
            continue
        result.append(text[i])
        i += 1
    # Close any unclosed tags
    for tag in reversed(stack):
        result.append(f'</{tag}>')
    return ''.join(result)


def _apply_tag_colors(text: str) -> str:
    """
    Convert <TAG>...</TAG> to HTML spans.
    POR renders as underline-only so inner colored spans are preserved.
    Processes innermost tags first (multi-pass).
    """
    max_passes = 10
    for _ in range(max_passes):
        found = False
        for tag, (style, label) in TAG_STYLES.items():
            if tag == 'POR':
                pattern = r'<POR>(.*?)</POR>'
                def por_replacer(m):
                    return f'<span style="{POR_STYLE}">{m.group(1)}{label}</span>'
                new_text = re.sub(pattern, por_replacer, text, flags=re.DOTALL)
            else:
                pattern  = f'<{tag}>([^<]*)</{tag}>'
                new_text = re.sub(
                    pattern,
                    f'<span style="{style} font-weight: bold;">\\1{label}</span>',
                    text, flags=re.DOTALL
                )
            if new_text != text:
                text  = new_text
                found = True
        if not found:
            break
    return text


def _apply_markdown_inline(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*',     r'<em>\1</em>',         text)
    return text


# ---------------------------------------------------------------------------
# Tag injection into conversation / properties
# ---------------------------------------------------------------------------

def _inject_tags(target: str, df, idx: int) -> str:
    """
    Collect all tagged spans from justification columns and inject them into
    `target` text using fuzzy matching (same logic as excel.py).
    Non-POR tags first, then POR so it can wrap already-injected inner tags.
    """
    tags_found = []
    seen       = set()

    for col in df.columns:
        if 'justification' not in col.lower():
            continue
        val         = str(df.iloc[idx][col])
        raw_matches = re.findall(r'<(PPI|MD|EXP|APP|POR|MOD)>(.*?)</\1>', val, re.DOTALL)

        for tag, content in raw_matches:
            if tag == 'POR':
                # extract inner tags from POR span first
                for itag, icontent in re.findall(r'<(PPI|MD|EXP|APP|MOD)>(.*?)</\1>', content, re.DOTALL):
                    clean = _clean_content(icontent)
                    if clean and clean not in seen:
                        tags_found.append((itag, clean))
                        seen.add(clean)
            clean = _clean_content(content)
            if clean and clean not in seen:
                tags_found.append((tag, clean))
                seen.add(clean)

    non_por = [(t, c) for t, c in tags_found if t != 'POR']
    por     = [(t, c) for t, c in tags_found if t == 'POR']

    for tag, content in non_por + por:
        target = _tag_first_untagged_occurrence(target, tag, content)

    return target


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_conversation(row, state, mode: str, idx: int, df=None) -> str:
    if mode.startswith("écrit") and idx < len(state.full_ecrit_sentence):
        conversation = state.full_ecrit_sentence[idx]
        if mode == "écrit":
            from ppi_analyser.preprocessing.conversation import clean_conv
            conversation = clean_conv(conversation, "écrit")
        conversation = conversation.replace("<dialogue>", "<strong>").replace("</dialogue>", "</strong>")
    else:
        conversation = row.get("Conversation", "")

    conversation = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', str(conversation))
    # Strip <narration> tags (corpus structural tags, not part of our annotation set)
    conversation = re.sub(r'</?narration[^>]*>', ' ', conversation)
    # Normalize <PPI> tags with surrounding spaces e.g. "<PPI> text </PPI>"
    conversation = re.sub(r'<PPI>\s+', '<PPI>', conversation)
    conversation = re.sub(r'\s+</PPI>', '</PPI>', conversation)
    # Strip leftover strong tags from old hardcoded PPI replace
    conversation = re.sub(r'<strong style="color: green;">', '', conversation)
    conversation = re.sub(r'<strong style="color:\s*green;">', '', conversation)
    # Remove orphaned </strong> not closing a speaker label (speaker labels end with ]: </strong>)
    conversation = re.sub(r'(?<!\]\s)\s*</strong>', '', conversation)
    conversation = re.sub(r'\[(.*?)\]', r'<br><strong>[\1]:</strong> ', conversation).strip()

    if df is not None:
        conversation = _inject_tags(conversation, df, idx)

    conversation = _fix_crossed_tags(conversation)
    # Remove empty tags left by crossed-tag repair
    conversation = re.sub(r'<(PPI|MD|EXP|APP|POR|MOD)></\1>', '', conversation)
    conversation = _apply_tag_colors(conversation)
    conversation = _apply_markdown_inline(conversation)

    return f'''<div style="border: 1px solid black; padding: 8px; margin: 8px 0;
                border-radius: 4px; background: #f9f9f9; line-height: 1.8;">
                {conversation}
               </div>'''


def _format_properties(df, row, idx: int, models: list) -> str:
    props = []
    for j in range(len(df.columns) - 3):
        col   = df.columns[j]
        # Inject tags into the cell value, then render colors and markdown
        cell  = str(df.iloc[idx, j])
        cell  = _inject_tags(cell, df, idx)
        val   = _apply_markdown_inline(_apply_tag_colors(cell))
        model = models[j // 2] if j // 2 < len(models) else "auto"
        if model == "no_model":
            model = "auto"

        if "Erreur de traitement" in val:
            prop = f'<p style="color: red;"><strong>{col}:</strong> {val}</p>'
        elif "Justification" not in col:
            prop = f'''<div style="margin-top: 1em;">
                        <h3 style="margin-bottom: 2px;">{col}
                            <span style="color:#888; font-size:0.85em; font-weight:normal;">({model})</span>
                        </h3>
                        <p style="margin-top: 2px;">{val}</p>
                       </div>'''
        else:
            prop = f'''<div style="border: 1px solid #ccc; padding: 8px; margin: 4px 0;
                        border-radius: 4px; background: #fafafa;">
                        <strong>{col}:</strong> {val}
                       </div>'''
        props.append(prop)
    return "\n".join(props)


# ---------------------------------------------------------------------------
# Export entry point
# ---------------------------------------------------------------------------

def export_pdf(df, state, path: str, mode: str) -> None:
    try:
        weasyprint = importlib.import_module("weasyprint")
        HTML = getattr(weasyprint, "HTML")
        CSS  = getattr(weasyprint, "CSS")
    except ImportError as e:
        logger.warning("PDF export unavailable — missing dependency: %s", e)
        return

    html     = _build_html(df, state, mode)
    css      = CSS(string="""
        body { text-align: justify; word-wrap: break-word; hyphens: auto; }
        p, li, div { word-break: break-word; overflow-wrap: anywhere; }
    """)
    HTML(string=html).write_pdf(path, stylesheets=[css])
    html_path = path.replace(".pdf", ".html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    state.html = html
    logger.info("PDF exported to %s", path)


def _build_html(df, state, mode: str) -> str:
    output               = []
    expression           = state.expression
    expression_list      = state.expression_list if state.expression_list else None
    models               = state.submodel_list
    individual_conv_time = state.individual_conv_time

    for idx, row in df.iterrows():
        conversation = _format_conversation(row, state, mode, idx, df=df)
        props        = _format_properties(df, row, idx, models)
        model_names  = "".join(set(models) - {"no_model"})
        model_names  = re.sub(r'(.*?_)', '', model_names)
        conv_time    = individual_conv_time[idx] if idx < len(individual_conv_time) else 0
        sent_idx     = state.sent_index[idx]     if idx < len(state.sent_index)     else idx

        output.append(f"""
<div style="margin-bottom: 2em; border-bottom: 1px solid #ccc; padding-bottom: 1em;">
    <p><strong>No. de la conversation:</strong> {sent_idx}</p>
    <p><strong>PPI:</strong> <em>{expression_list[idx] if expression_list else expression}</em></p>
    <p><strong>Temps de traitement:</strong> {conv_time:.2f}s</p>
    <p><strong>Modèle(s):</strong> {model_names}</p>
    <h2>{idx + 1}. Analyse de la conversation No.{idx + 1}</h2>
    <h3>{idx + 1}.1. Informations générales</h3>
    <p><strong>Locuteur:</strong> {row['Locuteur']}</p>
    <p><strong>Interlocuteur(s):</strong> {row.get('Interlocuteur(s)', '')}</p>
    <p><strong>Conversation:</strong></p>
    {conversation}
    <h3>{idx + 1}.2. Analyse de l'expression</h3>
    {props}
</div>
""")

    body   = "\n".join(output)
    header = f"""
<div>
    <h2>Nombre total de conversations analysées: {len(df)}</h2>
    <p><strong>Temps total:</strong> {state.total_time:.2f}s</p>
    <p><strong>Fichier:</strong> {state.fichier}</p>
</div>
"""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{
    font-family: Arial, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 2em;
    text-align: justify;
    word-wrap: break-word;
}}
h2 {{ color: #333; }}
h3 {{ color: #555; }}
span {{ display: inline; }}
</style>
</head>
<body>
{header}
{body}
</body>
</html>
"""
