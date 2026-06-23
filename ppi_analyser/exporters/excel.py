import logging
import pandas as pd
from ppi_analyser.config import PipelineConfig
logger = logging.getLogger(__name__)

ORIGINAL_COLS = [
    "sentId", "left", "node", "right", "author", "collection",
    "corpusId", "pubdate", "publisher", "pubplace", "puburl",
    "source_language", "sourcefilename", "sub_genre", "title",
    "type", "wordsnumber", "year"
]

TAG_REPLACEMENTS = [
    ("</MD>",  "(MD)</MD>"),
    ("</EXP>", "(Expansion)</EXP>"),
    ("</MOD>", "(Modifieur)</MOD>"),
    ("</POR>", "(Portée)</POR>"),
    ("</APP>", "(Appellatif)</APP>"),
]

import re
import pandas as pd
import math

def format_ppi_bold(df, filename, tag_color_map=None):
    """
    Format Excel with custom tag-based formatting
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to format
    filename : str
        Output filename for the Excel file
    tag_color_map : dict, optional
        Dictionary mapping tag names to colors (e.g., {'MD': 'red', 'EXP': 'green', 'PPI': 'blue'})
        Default: {'PPI': 'blue', 'MD': 'red', 'EXP': 'green'}
    """
    # Default tag-color mapping if none provided
    if tag_color_map is None:
        tag_color_map = {
            'POR': 'yellow',
            'PPI': 'blue',
            'MD':  'red',
            'EXP': 'green',
            'APP': 'pink',
            'MOD': 'orange',
        }
        
    # Validate that all tags are properly formatted (uppercase without brackets)
    # Convert tag names to uppercase for consistency
    tag_color_map = {tag.upper(): color for tag, color in tag_color_map.items()}
    
    # Remove any unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.reset_index(drop=True)
    
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        
        # Create formats dictionary for each color
        color_formats = {
            'blue': workbook.add_format({'bold': True, 'font_color': 'blue'}),
            'red': workbook.add_format({'bold':True,'font_color': 'red'}),
            'green': workbook.add_format({'bold':True,'font_color': 'green'}),
            'black': workbook.add_format({'text_wrap': True, 'valign': 'top'}),
            'purple': workbook.add_format({'bold':True,'font_color': 'purple'}),
            'orange': workbook.add_format({'bold':True,'font_color': 'orange'}),
            'brown': workbook.add_format({'bold':True,'font_color': 'brown'}),
            'pink': workbook.add_format({'bold':True,'font_color': 'pink'}),
            'gray': workbook.add_format({'bold':True,'font_color': 'gray'}),
            'cyan': workbook.add_format({'bold':True,'font_color': 'cyan'}),
            'magenta': workbook.add_format({'bold':True,'font_color': 'magenta'}),
            'yellow': workbook.add_format({'bold':True,'font_color': 'yellow'}),
            'dark_blue': workbook.add_format({'bold':True,'font_color': '#00008B'}),
            'dark_red': workbook.add_format({'bold':True,'font_color': '#8B0000'}),
            'dark_green': workbook.add_format({'bold':True,'font_color': '#006400'}),
            'gold': workbook.add_format({'bold':True,'font_color': '#FFD700'}),
        }
        
        # POR uses underline only (no color)
        por_format = workbook.add_format({'underline': True})

        # Create formats for each tag in the map
        tag_formats = {'POR': por_format}
        for tag, color in tag_color_map.items():
            if tag == 'POR':
                continue  # already set above
            if color in color_formats:
                tag_formats[tag] = color_formats[color]
            else:
                # Try to interpret color as a hex code
                try:
                    tag_formats[tag] = workbook.add_format({'bold': True, 'font_color': color})
                except:
                    # Fallback to black if color format is invalid
                    tag_formats[tag] = color_formats['black']
        
        regular = color_formats['black']
        header_format = workbook.add_format({'bold': True, 'valign': 'top', 'bg_color': '#F2F2F2'})
        
        # Set header row format
        for col_num, col_name in enumerate(df.columns):
            worksheet.write(0, col_num, col_name, header_format)
        
        # First, calculate column widths based on content
        col_widths = {col: len(str(col)) for col in df.columns}  # Initialize with header length
        
        for row_num in range(len(df)):
            for col_num, col_name in enumerate(df.columns):
                cell_value = df.iloc[row_num][col_name]
                if pd.notna(cell_value):
                    text = str(cell_value)
                    # Remove HTML-like tags and markdown for clean text length calculation
                    clean_text = re.sub(r'<[^>]+>', '', text)
                    clean_text = re.sub(r'\*\*', '', clean_text)
                    col_widths[col_name] = max(col_widths[col_name], len(clean_text) + 2)
        
        # Cap maximum width at 100
        for col_name in col_widths:
            col_widths[col_name] = min(col_widths[col_name], 100)
        
        # Now calculate row heights based on the FINAL column widths
        base_height = 15
        row_heights = {}
        
        for row_num in range(len(df)):
            excel_row = row_num + 1
            max_height = base_height
            
            for col_num, col_name in enumerate(df.columns):
                cell_value = df.iloc[row_num][col_name]
                
                if pd.notna(cell_value):
                    text = str(cell_value)
                    # Remove HTML-like tags and markdown for clean text length calculation
                    clean_text = re.sub(r'<[^>]+>', '', text)
                    clean_text = re.sub(r'\*\*', '', clean_text)
                    
                    # Get the final width for this column
                    col_width = col_widths[col_name]
                    
                    # Calculate lines needed based on column width
                    chars_per_line = int(col_width * 1.1)
                    if chars_per_line > 0:
                        lines_needed = max(1, math.ceil(len(clean_text) / chars_per_line))
                    else:
                        lines_needed = 1
                    
                    # Calculate height needed for this cell
                    cell_height = base_height * lines_needed
                    max_height = max(max_height, cell_height)
            
            row_heights[excel_row] = max_height
        
        # Set column widths first
        for col_num, col_name in enumerate(df.columns):
            worksheet.set_column(col_num, col_num, col_widths[col_name], regular)
        
        # Then write content with formatting and set row heights
        for row_num in range(len(df)):
            excel_row = row_num + 1
            
            # Set the pre-calculated row height
            worksheet.set_row(excel_row, row_heights[excel_row])
            
            for col_num, col_name in enumerate(df.columns):
                cell_value = df.iloc[row_num][col_name]
                
                if pd.notna(cell_value):
                    text = str(cell_value)
                    
                    # Replace ** with PPI tags
                    text = re.sub(r'\*\*(.*?)\*\*', r'<PPI>\1</PPI>', text)
                    
                    # Build regex pattern for all tags in the map
                    tags_pattern = '|'.join([f'<{tag}>' for tag in tag_color_map.keys()] + 
                                           [f'</{tag}>' for tag in tag_color_map.keys()])
                    pattern = f'({tags_pattern})'
                    
                    # Check if any special tags exist
                    has_tags = any(f'<{tag}>' in text for tag in tag_color_map.keys())
                    
                    if has_tags:
                        parts = re.split(pattern, text)
                        
                        rich_string = []
                        tag_states = {tag: False for tag in tag_color_map.keys()}
                        # Cache for combined POR+color formats to avoid recreating them
                        por_combined_cache = {}

                        for part in parts:
                            # Check for opening tags
                            tag_opened = False
                            for tag in tag_color_map.keys():
                                if part == f'<{tag}>':
                                    tag_states[tag] = True
                                    tag_opened = True
                                    break

                            if tag_opened:
                                continue

                            # Check for closing tags
                            tag_closed = False
                            for tag in tag_color_map.keys():
                                if part == f'</{tag}>':
                                    tag_states[tag] = False
                                    tag_closed = True
                                    break

                            if tag_closed:
                                continue

                            # Regular text part
                            if part:
                                in_por = tag_states.get('POR', False)

                                # Find the innermost active non-POR tag
                                active_tag = None
                                for tag in tag_color_map.keys():
                                    if tag != 'POR' and tag_states[tag]:
                                        active_tag = tag
                                        break

                                if active_tag and active_tag in tag_formats:
                                    if in_por:
                                        # Combine color format with underline
                                        if active_tag not in por_combined_cache:
                                            base = tag_formats[active_tag]
                                            por_combined_cache[active_tag] = workbook.add_format({
                                                'bold': True,
                                                'font_color': tag_color_map[active_tag],
                                                'underline': True,
                                            })
                                        rich_string.append(por_combined_cache[active_tag])
                                    else:
                                        rich_string.append(tag_formats[active_tag])
                                    rich_string.append(part)
                                elif in_por:
                                    rich_string.append(por_format)
                                    rich_string.append(part)
                                else:
                                    rich_string.append(part)
                        
                        # Write the rich string to Excel
                        if rich_string:
                            rich_string = [item for item in rich_string if item != '']
                            
                            if len(rich_string) > 0:
                                try:
                                    worksheet.write_rich_string(excel_row, col_num, *rich_string)
                                except Exception:
                                    clean_text = re.sub(r'<[^>]+>', '', text)
                                    worksheet.write(excel_row, col_num, clean_text, regular)
                        else:
                            worksheet.write(excel_row, col_num, text, regular)
                    else:
                        worksheet.write(excel_row, col_num, text, regular)
                else:
                    worksheet.write(excel_row, col_num, '', regular)

KNOWN_TAGS = {'PPI', 'MD', 'EXP', 'APP', 'POR', 'MOD'}

def _is_inside_tag(text, match_start, match_end):
    """Return True if the match position is already inside an XML-like tag."""
    text_before = text[:match_start]
    open_tags  = len(re.findall(r'<[^>/]+>', text_before))
    close_tags = len(re.findall(r'</[^>]+>', text_before))
    return open_tags > close_tags


def _normalize_ws(s):
    """Collapse all whitespace (including literal \\n strings and real newlines) to a single space."""
    s = s.replace('\\n', ' ').replace('\\t', ' ')  # literal \n \t strings
    return re.sub(r'\s+', ' ', s).strip()


def _clean(s):
    """Strip XML-like tags and normalize whitespace (used on target text)."""
    return _normalize_ws(re.sub(r'<[^>]+>', '', s))


def _clean_content(s):
    """Strip XML tags, speaker labels, markdown artifacts from justification content."""
    s = re.sub(r'<[^>]+>', '', s)          # remove <TAG> </TAG>
    s = re.sub(r'\[.*?\]', '', s)           # remove [Locuteur 1], [Ours], etc.
    s = re.sub(r'\*+', '', s)               # remove * and ** markdown
    return _normalize_ws(s)


def _anchor_pattern(content, n=3):
    """
    Build a loose pattern that matches the first n and last n words of content,
    with .*? in between. Falls back to exact match for very short content.
    This handles cases where the justification phrasing slightly differs from
    the target, or where punctuation/extra words appear inside the span.
    """
    words = content.split()
    if len(words) <= n * 2:
        # Short enough — match exactly
        escaped = re.escape(content)
    else:
        head = r'\s+'.join(re.escape(w) for w in words[:n])
        tail = r'\s+'.join(re.escape(w) for w in words[-n:])
        escaped = head + r'.*?' + tail
    left_b  = r'\b' if re.match(r'\w', content[0])  else ''
    right_b = r'\b' if re.search(r'\w', content[-1]) else ''
    return left_b + escaped + right_b


def _build_clean_to_raw(text):
    """
    Build a mapping from positions in the clean (tag-stripped, ws-normalized)
    text back to positions in the original raw text.
    Returns (clean_text, clean_to_raw) where clean_to_raw[j] = raw index of clean[j].
    """
    # First pass: strip tags, keeping a char-level map raw→clean
    clean_chars = []   # list of (raw_index, char)
    inside_tag = False
    for i, ch in enumerate(text):
        if ch == '<':
            inside_tag = True
        if not inside_tag:
            clean_chars.append((i, ch))
        if ch == '>':
            inside_tag = False

    # Second pass: normalize whitespace while keeping the mapping
    # We collapse runs of whitespace to a single space
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

    # Strip leading space
    if result_chars and result_chars[0][1] == ' ':
        result_chars = result_chars[1:]

    clean_text = ''.join(ch for _, ch in result_chars)
    clean_to_raw = [raw_i for raw_i, _ in result_chars]
    return clean_text, clean_to_raw


def _tag_first_untagged_occurrence(text, tag_name, content):
    """
    Find the first occurrence of `content` in the clean (tag-stripped,
    whitespace-normalized) version of `text` and wrap it with the tag.

    Uses an anchor-based fuzzy pattern so minor differences between the
    justification phrasing and the target text still match.
    """
    content = _normalize_ws(content)
    if not content:
        return text

    # --- already tagged? check raw text (tags are stripped in _clean so check raw) ---
    already_tagged = re.search(
        r'<' + re.escape(tag_name) + r'>' + r'.*?' + r'</' + re.escape(tag_name) + r'>',
        text,
        re.DOTALL
    )
    if already_tagged:
        return text

    pattern = _anchor_pattern(content)

    # --- build mapping from clean positions back to raw positions ---
    clean_text, clean_to_raw = _build_clean_to_raw(text)

    # --- find the first match in the clean text ---
    m = re.search(pattern, clean_text, re.DOTALL)
    if not m:
        return text

    clean_start = m.start()
    clean_end   = m.end()

    if clean_end > len(clean_to_raw) or clean_start >= len(clean_to_raw):
        return text

    raw_start = clean_to_raw[clean_start]
    raw_end   = clean_to_raw[clean_end - 1] + 1

    # If raw_start falls inside an existing tag, walk back to before that opening tag
    # so the new tag wraps the full span including any inner tags.
    if _is_inside_tag(text, raw_start, raw_end):
        # find the opening tag that contains raw_start and move before it
        opening = None
        for m in re.finditer(r'<[^/][^>]*>', text[:raw_start]):
            opening = m
        if opening is None:
            return text
        raw_start = opening.start()

    # If raw_end falls inside an existing tag, walk forward to after its closing tag
    if raw_end < len(text):
        tail = text[raw_end:]
        close_m = re.match(r'[^<]*</[^>]+>', tail)
        if close_m and '<' not in tail[:close_m.start()]:
            raw_end += close_m.end()

    new_text = (
        text[:raw_start]
        + f'<{tag_name}>'
        + text[raw_start:raw_end]
        + f'</{tag_name}>'
        + text[raw_end:]
    )
    return new_text


def copy_tags_to_columns(df, source_col, target_cols, exact_match=True):
    if isinstance(source_col, list):
        raise ValueError("source_col must be a single column name (string), not a list")

    if source_col not in df.columns:
        print(f"Warning: Source column '{source_col}' not found in dataframe")
        return df

    df = df.copy()

    for idx, row in df.iterrows():
        source_value = row[source_col]
        if pd.isna(source_value):
            continue

        source_text = str(source_value)

        # Collect every tagged span from the source column (known tags only).
        # Two passes: top-level tags, then inner tags inside any POR span
        # (re.findall with .*? won't recurse into nested tags).
        raw_matches = re.findall(
            r'<(PPI|MD|EXP|APP|POR|MOD)>(.*?)</\1>',
            source_text,
            re.DOTALL
        )
        tag_matches = []
        for t, c in raw_matches:
            tag_matches.append((t, c))
            if t == 'POR':
                inner = re.findall(
                    r'<(PPI|MD|EXP|APP|MOD)>(.*?)</\1>',
                    c, re.DOTALL
                )
                tag_matches.extend(inner)

        for tag_name, content in tag_matches:
            # Strip any inner tags from content before matching
            clean_content = _clean_content(content)
            if not exact_match:
                clean_content = clean_content.lower()
            if not clean_content:
                continue

            for target_col in target_cols:
                if target_col not in df.columns:
                    continue

                target_value = row[target_col]
                if pd.isna(target_value):
                    continue

                target_text = str(df.at[idx, target_col])   # always read fresh

                new_text = _tag_first_untagged_occurrence(target_text, tag_name, clean_content)

                if new_text != target_text:
                    df.at[idx, target_col] = new_text

    return df


def copy_tags_from_multiple_sources(df, source_cols, target_cols, exact_match=True):
    """
    Copy tags from multiple source columns to target columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to process
    source_cols : list
        List of source column names
    target_cols : list
        List of target column names
    exact_match : bool, default=True
        If True, match exact content including case
    
    Returns:
    --------
    pandas.DataFrame : The updated dataframe
    """
    result_df = df.copy()
    
    for source_col in source_cols:
        if source_col in result_df.columns:
            result_df = copy_tags_to_columns(result_df, source_col, target_cols, exact_match)
    
    return result_df


def export_excel(df: pd.DataFrame, path: str) -> None:
    format_ppi_bold(df, path)
    logger.info("Excel exported to %s", path)


def export_excel_simple(df: pd.DataFrame, path: str, sentence_file: str = None, config:PipelineConfig = None ) -> None:

    df_simple = df.copy()
    if sentence_file:
        df_full = pd.read_excel(sentence_file)
        found_cols = [c for c in ORIGINAL_COLS if c in df_full.columns]
        start = int(config.start_sent)
        end = config.max_sentences
        if config.max_sentences == "all":
        	end = len(df_full)
        else:
        	end = int(config.max_sentences)
        df_full = df_full.iloc[start:end].reset_index(drop=True) # added to match df with sent file indices
        df_simple = pd.concat([df_simple, df_full[found_cols]], axis=1)

    # copy tags from justification columns into left/node/right
    justification_cols = [c for c in df_simple.columns if "justification" in c.lower()]
    if justification_cols:
        print("DEBUG justif sample  :", df_simple[justification_cols[0]].iloc[0])

    df_simple = copy_tags_from_multiple_sources(
        df_simple,
        source_cols=justification_cols,
        target_cols=["left", "node", "right"]
    )


    # apply tag label suffixes
    for col in ["left", "right", "node"]:
        if col not in df_simple.columns:
            continue
        for old, new in TAG_REPLACEMENTS:
            df_simple[col] = df_simple[col].apply(lambda x: str(x).replace(old, new))

    if "node" in df_simple.columns:
        df_simple["node"] = df_simple["node"].apply(lambda x: f"  <PPI>{x}</PPI>  ")

    drop_cols = [c for c in ["Locuteur", "Interlocuteur(s)"] if c in df_simple.columns]
    df_simple = df_simple.drop(columns=drop_cols)
    
    # drop justification cols from simple df
    
    justification_cols = [c for c in df_simple.columns if c.lower().find("justification")>0 ]
    df_simple = df_simple.drop(justification_cols,axis=1)
    
    # save
    #df_simple.to_csv("/tmp/debug_before_format.csv", index=False)
    format_ppi_bold(df_simple, path)
    logger.info("Simple Excel exported to %s", path)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
