# exporters/pdf.py

import re
import logging
import importlib

logger = logging.getLogger(__name__)
# exporters/pdf.py


TAG_STYLES = {
    "PPI":  ("color: green;",  ""),
    "MD":   ("color: red;",    " (MD)"),
    "EXP":  ("color: brown;",  " (Expansion)"),
    "APP":  ("color: blue;",   " (Appellatif)"),
    "POR": ("color: purple;", " (Portée)"),
    "MOD":  ("color: orange;", " (Modifieur)"),
}

def _apply_tag_colors(text: str) -> str:
    """Handle nested and overlapping tags by processing innermost first."""
    
    # keep replacing until no more tags found (handles nesting)
    max_passes = 10
    for _ in range(max_passes):
        found = False
        for tag, (style, label) in TAG_STYLES.items():
            # match innermost tags first (no nested tags inside)
            pattern = f'<{tag}>([^<]*)</{tag}>'
            new_text = re.sub(
                pattern,
                f'<span style="{style}; font-weight: bold;">\\1{label}</span>',
                text,
                flags=re.DOTALL
            )
            if new_text != text:
                text = new_text
                found = True
        if not found:
            break
    
    return text
def _apply_markdown_inline(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    return text
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
    conversation = re.sub(
        r'\[(.*?)\]',
        r'<br><strong>[\1]:</strong> ',
        conversation
    ).strip()
    conversation = conversation.replace(
        "<PPI> ", '<strong style="color: green;">'
    ).replace(
        " </PPI>", "</strong>"
    )

    # inject tags from justifications before applying colors
    if df is not None:
        conversation = _inject_tags_into_conversation(conversation, df, row, idx)

    conversation = _apply_tag_colors(conversation)
    conversation = _apply_markdown_inline(conversation)

    return f'''<div style="border: 1px solid black; padding: 8px; margin: 8px 0;
                border-radius: 4px; background: #f9f9f9; line-height: 1.8;">
                {conversation}
               </div>'''

def _inject_tags_into_conversation(conversation: str, df, row, idx: int) -> str:
    """Find all tagged expressions from justifications and highlight them in the conversation."""
    
    # collect all tagged expressions from justification columns
    tags_found = []
    for col in df.columns:
        if "Justification" in col:
            val = str(df.iloc[idx][col])
            # find all <TAG>content</TAG> patterns
            matches = re.findall(r'<(PPI|MD|EXP|APP|POR|MOD)>(.*?)</\1>', val, re.DOTALL)
            for tag, content in matches:
                content = content.strip()
                if content and content not in [m[1] for m in tags_found]:
                    tags_found.append((tag, content))

    # inject tags into conversation text
    for tag, content in tags_found:
        escaped = re.escape(content)
        # only replace if not already tagged
        already_tagged = re.search(f'<{tag}>{escaped}</{tag}>', conversation)
        if not already_tagged:
            conversation = re.sub(
                escaped,
                f'<{tag}>{content}</{tag}>',
                conversation,
                count=1
            )

    return conversation
def _format_properties(df, row, idx: int, models: list) -> str:
    props = []
    for j in range(len(df.columns) - 3):
        col = df.columns[j]
        val = val = _apply_markdown_inline(_apply_tag_colors(str(df.iloc[idx, j])))
        model = models[j // 2] if j // 2 < len(models) else "auto"
        if model == "no_model":
            model = "auto"

        if "Erreur de traitement" in val:
            prop = f'<p style="color: red;"><strong>{col}:</strong> {val}</p>'

        elif "Justification" not in col:
            # title + value on same block
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

def export_pdf(df, state, path: str, mode: str) -> None:
    try:
        weasyprint = importlib.import_module("weasyprint")
        HTML = getattr(weasyprint, "HTML")
        CSS = getattr(weasyprint, "CSS")
    except ImportError as e:
        logger.warning("PDF export unavailable — missing dependency: %s", e)
        return

    html = _build_html(df, state, mode)
    css = CSS(string="""
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
    markdown_module = importlib.import_module("markdown")
    output = []

    expression = state.expression
    expression_list = state.expression_list if state.expression_list else None
    models = state.submodel_list
    individual_conv_time = state.individual_conv_time

    for idx, row in df.iterrows():
        conversation = _format_conversation(row, state, mode, idx,df=df)
        props = _format_properties(df, row, idx, models)

        model_names = "".join(set(models) - {"no_model"})
        model_names = re.sub(r'(.*?_)', '', model_names)
        conv_time = individual_conv_time[idx] if idx < len(individual_conv_time) else 0
        sent_idx = state.sent_index[idx] if idx < len(state.sent_index) else idx

        # use raw HTML instead of markdown for the properties section
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

    body = "\n".join(output)
    header = f"""
<div>
    <h2>Nombre total de conversations analysées: {len(df)}</h2>
    <p><strong>Temps total:</strong> {state.total_time:.2f}s</p>
    <p><strong>Fichier:</strong> {state.fichier}</p>
</div>
"""
    # wrap in a full HTML document with CSS
    full_html = f"""
<!DOCTYPE html>
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
    return full_html


