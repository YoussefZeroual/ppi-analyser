# exporters/excel.py

import logging
import pandas as pd


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
        formats = {}
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
        
        # Create special highlight format for PORT tags
        port_highlight_format = workbook.add_format({
            'bold': True,
            'bg_color': '#E6E6FA',  # Light purple background
            'font_color': 'black'     # Keep text black for readability
        })
        
        # Create formats for each tag in the map
        tag_formats = {}
        for tag, color in tag_color_map.items():
            if color in color_formats:
                # For PPI-like tags (can make bold if needed)
                if tag == 'PPI' or any(bold_tag in tag for bold_tag in ['PPI', 'BOLD']):
                    tag_formats[tag] = workbook.add_format({'bold': True, 'font_color': color})
                else:
                    tag_formats[tag] = color_formats[color]
            else:
                # Try to interpret color as a hex code
                try:
                    tag_formats[tag] = workbook.add_format({'font_color': color})
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
                    # Average character width in Excel is roughly 1.1 units per character
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
                    
                    # First, normalize all formatting tags
                    # Replace ** with tag format (assuming PPI style)
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
                        
                        # Track if we're currently inside a PORT tag for highlighting
                        in_port_tag = False
                        port_segment_start = None
                        
                        for i, part in enumerate(parts):
                            # Check for opening tags
                            tag_opened = False
                            for tag in tag_color_map.keys():
                                if part == f'<{tag}>':
                                    tag_states[tag] = True
                                    tag_opened = True
                                    if tag == 'POR':
                                        in_port_tag = True
                                        port_segment_start = len(rich_string)
                                    break
                            
                            if tag_opened:
                                continue
                            
                            # Check for closing tags
                            tag_closed = False
                            for tag in tag_color_map.keys():
                                if part == f'</{tag}>':
                                    tag_states[tag] = False
                                    tag_closed = True
                                    if tag == 'POR':
                                        in_port_tag = False
                                        
                                        # Apply highlight to all segments that were inside the PORT tag
                                        if port_segment_start is not None:
                                            # Replace the segments from port_segment_start to current with highlighted versions
                                            port_segments = rich_string[port_segment_start:]
                                            rich_string = rich_string[:port_segment_start]
                                            
                                            for segment in port_segments:
                                                if isinstance(segment, str):
                                                    # Apply highlight format to text
                                                    rich_string.append(port_highlight_format)
                                                    rich_string.append(segment)
                                                else:
                                                    # Keep existing formats (like colors) but add background
                                                    # Create a new format that combines the existing format with highlight
                                                    if segment == port_highlight_format:
                                                        rich_string.append(segment)
                                                    else:
                                                        # For colored segments inside PORT, preserve their color
                                                        # but add the highlight background
                                                        combined_format = workbook.add_format({
                                                            'bold': segment.bold,
                                                            'font_color': segment.font_color,
                                                            'bg_color': '#E6E6FA'
                                                        })
                                                        rich_string.append(combined_format)
                                                        # The next segment will be the text, so we'll just continue
                                                        # and let it be added normally
                                                        pass
                                                # The text will be added in the next iteration
                                        port_segment_start = None
                                    break
                            
                            if tag_closed:
                                continue
                            
                            # Regular text part
                            if part:  # Non-empty text
                                # Find which tags are active (prioritize tags in order of specificity)
                                active_tag = None
                                for tag in tag_color_map.keys():
                                    if tag_states[tag]:
                                        active_tag = tag
                                        break
                                
                                if active_tag and active_tag in tag_formats:
                                    # Check if we're inside a PORT tag
                                    if in_port_tag and active_tag != 'POR':
                                        # For colored items inside PORT, preserve their color but add highlight background
                                        original_format = tag_formats[active_tag]
                                        combined_format = workbook.add_format({
                                            'bold': original_format.bold,
                                            'font_color': original_format.font_color,
                                            'bg_color': '#E6E6FA'
                                        })
                                        rich_string.append(combined_format)
                                        rich_string.append(part)
                                    elif active_tag == 'POR':
                                        # For PORT tag itself, just use highlight (no color change)
                                        rich_string.append(port_highlight_format)
                                        rich_string.append(part)
                                    else:
                                        # Regular colored tag outside PORT
                                        rich_string.append(tag_formats[active_tag])
                                        rich_string.append(part)
                                else:
                                    # Plain text - check if inside PORT tag
                                    if in_port_tag:
                                        rich_string.append(port_highlight_format)
                                        rich_string.append(part)
                                    else:
                                        rich_string.append(part)
                        
                        # Write the rich string to Excel
                        if rich_string:
                            # Filter out any empty strings
                            rich_string = [item for item in rich_string if item != '']
                            
                            # Write rich string
                            if len(rich_string) > 0:
                                try:
                                    worksheet.write_rich_string(excel_row, col_num, *rich_string)
                                except Exception:
                                    # Fallback if rich string fails
                                    clean_text = re.sub(r'<[^>]+>', '', text)
                                    worksheet.write(excel_row, col_num, clean_text, regular)
                        else:
                            # Fall back to regular text
                            worksheet.write(excel_row, col_num, text, regular)
                    else:
                        # No special tags, write normally with text wrapping
                        worksheet.write(excel_row, col_num, text, regular)
                else:
                    # Write empty cell
                    worksheet.write(excel_row, col_num, '', regular)

KNOWN_TAGS = {'PPI', 'MD', 'EXP', 'APP', 'POR', 'MOD'}

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

        # only match known tags
        tag_matches = re.findall(
            r'<(PPI|MD|EXP|APP|POR|MOD)>(.*?)</\1>',
            source_text,
            re.DOTALL
        )

        for tag_name, content in tag_matches:
            clean_content = content.strip()
            if not clean_content:
                continue

            search_pattern = re.escape(clean_content)
            if not exact_match:
                search_pattern = '(?i)' + search_pattern

            for target_col in target_cols:
                if target_col not in df.columns:
                    continue

                target_value = row[target_col]
                if pd.isna(target_value):
                    continue

                target_text = str(df.at[idx, target_col])  # always read fresh value

                # skip if already tagged
                if re.search(
                    f'<{re.escape(tag_name)}>{re.escape(clean_content)}</{re.escape(tag_name)}>',
                    target_text
                ):
                    continue

                # check content exists in clean version of target
                clean_target = re.sub(r'<[^>]+>', '', target_text)
                pattern = r'\b' + search_pattern + r'\b' if re.search(r'\w', clean_content) else search_pattern

                if not re.search(pattern, clean_target):
                    continue

                # replace only if not already inside a tag
                def make_replacer(tag, current_target):
                    def replace_if_not_tagged(match):
                        start = match.start()
                        text_before = current_target[:start]
                        text_after = current_target[match.end():]
                        open_tags = len(re.findall(r'<[^>/]+>', text_before))
                        close_tags = len(re.findall(r'</[^>]+>', text_before))
                        if open_tags > close_tags:
                            return match.group(0)
                        if text_before.endswith('<') or text_after.startswith('>'):
                            return match.group(0)
                        return f'<{tag}>{match.group(0)}</{tag}>'
                    return replace_if_not_tagged

                new_text = re.sub(
                    pattern,
                    make_replacer(tag_name, target_text),
                    target_text
                )

                if new_text != target_text:
                    df.at[idx, target_col] = new_text

    return df


# Alternative: Function that processes multiple source columns at once
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


def export_excel_simple(df: pd.DataFrame, path: str, sentence_file: str = None) -> None:
    df_simple = df.copy()

    if sentence_file:
        df_full = pd.read_excel(sentence_file)
        df_full = df_full[df_full.index.isin(df_simple.index)]
        found_cols = [c for c in ORIGINAL_COLS if c in df_full.columns]
        df_simple = pd.concat([df_simple, df_full[found_cols]], axis=1)

    # copy tags from justification columns into left/node/right
    justification_cols = [c for c in df_simple.columns if "Justification" in c]
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

    drop_cols = [c for c in ["Conversation", "Locuteur", "Interlocuteur(s)"] if c in df_simple.columns]
    df_simple = df_simple.drop(columns=drop_cols)

    # debug
    print("DEBUG left:", df_simple["left"].iloc[0] if "left" in df_simple.columns else "missing")
    print("DEBUG node:", df_simple["node"].iloc[0] if "node" in df_simple.columns else "missing")
    df_simple.to_csv("/tmp/debug_before_format.csv", index=False)
    format_ppi_bold(df_simple, path)
    logger.info("Simple Excel exported to %s", path)
