"""
Schema-agnostic Excel field matching using semantic embeddings (Sentence-Transformers).
Supports both row-oriented and column-oriented layouts by searching labels against
SYSTEM_FIELD_MAPPING descriptions.
"""
import json
import logging
import re
import os
import pandas as pd
import requests
from google.adk.tools import FunctionTool
from utils.system_field_mapping import SYSTEM_FIELD_MAPPING
from utils.excel_semantic_embedding import match_labels_to_system_fields
from utils.session_context import get_session_id, store_column_mapping, store_excel_header_row, store_rate_card_path, store_field_selection_result

logger = logging.getLogger(__name__)

# Extract field descriptions for backward compatibility
FIELD_DESCRIPTIONS = [desc for _, desc in SYSTEM_FIELD_MAPPING.values()]

DESCRIPTION_TO_FIELD = {}
for field_key, (scale_base, description) in SYSTEM_FIELD_MAPPING.items():
    DESCRIPTION_TO_FIELD[description] = field_key
    if "/" in description:
        for part in description.split("/"):
            part = part.strip()
            if part and part not in DESCRIPTION_TO_FIELD:
                DESCRIPTION_TO_FIELD[part] = field_key


def get_system_field_key_from_description(description: str) -> str | None:
    """
    Helper function to get the system field key from a description.
    Returns the exact field key (e.g., "SOURCE_LOC_ZONE") from SYSTEM_FIELD_MAPPING.
    """
    return DESCRIPTION_TO_FIELD.get(description)


# Weight/slab column pattern (e.g. 30-50, FTL, 100, > 500)
WEIGHT_PATTERN = re.compile(
    r'^(\d+-\d+|\d+\s*-\s*\d+|>\s*\d+|FTL|1/2\s*FTL|\d+(?:\.\d+)*)\s*$',
    re.IGNORECASE,
)

# How many rows/columns to scan for header labels
MAX_HEADER_ROWS = 30
MAX_HEADER_COLS = 300

# Rows to merge for multi-row header templates (e.g. DB Schenker, CEVA)
MAX_MERGED_HEADER_ROWS = 5
MAX_MERGED_HEADER_COLS = 100

# Deep scan to find "header far down" (e.g. Roberts Europe ratesheet with header at row 18)
MAX_SCAN_ROWS_FOR_HEADER = 30
MIN_HEADER_ROW_MATCHES = 2
HEADER_WINDOW_ROWS = 4

# Row index used by downstream when no header row is detected (e.g. sap_upload_tool header=2)
HEADER_ROW_FOR_DOWNSTREAM = 2

# Minimum cosine similarity to accept a label -> system field match
SIMILARITY_THRESHOLD = 0.45

# Minimum matches from merged strategy to prefer it over flat cell matching
MIN_MERGED_MATCHES_TO_PREFER = 2

# Max length for a single header label; longer text is treated as heading/paragraph and excluded
MAX_HEADER_LABEL_LENGTH = 80

# Characters that indicate a bullet or note line (not a column header). Strip and check start of text.
_BULLET_NOTE_PREFIXES = ("■", "•", "▪", "‣", "◦", "►", "●", "○")

def _is_too_long_for_header(text: str) -> bool:
    """True if text is empty or longer than typical column header length (exclude from matching)."""
    if not text or not text.strip():
        return True
    return len(text.strip()) > MAX_HEADER_LABEL_LENGTH


def _is_note_or_bullet_label(text: str) -> bool:
    """True if text looks like a bullet point or note line (e.g. '■ VAT exclusive'), not a column header."""
    if not text or not text.strip():
        return False
    s = text.strip()
    if s.startswith(_BULLET_NOTE_PREFIXES):
        return True
    # Common list-style bullet prefixes (e.g. "- Note text", "* Item")
    if re.match(r"^[\-\*]\s+", s):
        return True
    return False


def _should_exclude_label(text: str) -> bool:
    """True if label should be excluded from column matching (too long, or bullet/note line)."""
    return _is_too_long_for_header(text) or _is_note_or_bullet_label(text)


def _get_exclude_heading_substrings() -> tuple[str, ...]:
    """Load heading-exclude substrings from config or env (comma-separated). Fallback when LLM is not used."""
    try:
        env_val = os.environ.get("EXCLUDE_HEADING_SUBSTRINGS", "").strip()
        if env_val:
            return tuple(s.strip() for s in env_val.split(",") if s.strip())
        from config.dev_config import EXCLUDE_HEADING_SUBSTRINGS
        return tuple(EXCLUDE_HEADING_SUBSTRINGS) if EXCLUDE_HEADING_SUBSTRINGS else ()
    except Exception:
        return ()


def _is_heading_label(text: str, exclude_substrings: tuple[str, ...] | None = None) -> bool:
    """True if text looks like a document/section heading (exclude from column mapping)."""
    if not text or not text.strip():
        return False
    substrings = exclude_substrings if exclude_substrings is not None else _get_exclude_heading_substrings()
    if not substrings:
        return False
    lower = text.strip().lower()
    return any(sub in lower for sub in substrings)


def _llm_detect_heading_labels(matched_fields: list[dict]) -> set[str]:
    """
    Ask the LLM which of the matched column labels are document/section headings (to exclude).
    Returns set of excel_column values to exclude. On failure or missing config, returns empty set.
    """
    if not matched_fields:
        return set()
    try:
        from config.dev_config import AZURE_OPENAI_CHAT_URL, AZURE_OPENAI_API_KEY
    except Exception as e:
        logger.debug("Azure config not available for heading detection: %s", e)
        return set()
    labels_list = [m["excel_column"] for m in matched_fields]
    prompt = (
        "You are classifying Excel labels from a freight rate card. Some labels are real DATA COLUMN HEADERS "
        "(should be kept); others are DOCUMENT/SECTION TEXT and must be excluded (headings, subtitles, legal notes, "
        "explanatory sentences).\n\n"
        "EXCLUDE (return these in your JSON array) – do NOT treat as column headers:\n"
        "- Bullet points or note lines (e.g. starting with ■ or •): '■ This price is inclusive of 60 minutes loading and 60 minutes unloading', "
        "'■ VAT exclusive', '■ Minimum fuel surcharge is 0% which is the valid surcharge at the time of offer', '■ Payment terms: 30 days invoice reception date'.\n"
        "- Sheet titles, subtitles, or section headers (e.g. 'Rate per loaded kilometre', 'Carrier name: X').\n"
        "- Full sentences or explanatory text (e.g. 'Prices exclusive possible additional costs that may occur on a shipment level (see separate sheets)', "
        "'Rates are excluding a fuel surcharge which is indexed on a weekly basis based on the weekly average fuel price', "
        "'Payment terms: 30 days invoice reception date').\n"
        "- Legal/contract wording, disclaimers, or notes that describe conditions rather than naming a single data field.\n"
        "- Anything that reads like a paragraph or clause, contains colons followed by a full phrase, or mentions 'terms', 'excluding', 'see separate', 'based on', 'which is' in a sentence context.\n\n"
        "KEEP (do NOT return these) – these ARE column headers:\n"
        "- Short, specific field names (e.g. 'Origin Country', 'Destination Zone', 'Weight (kg)', 'Service Type', 'Transit Time', 'Lane ID', 'Chargeable Weight').\n"
        "- Weight/slab labels (e.g. '0-50', '151-200', 'FTL', '> 20 001').\n"
        "- Typically one to six words; clearly refer to a single column of data.\n\n"
        "Return ONLY a JSON array of the EXACT label strings that are headings/sentences to EXCLUDE. "
        "Use the exact text as shown below. If none should be excluded, return []. No other output.\n\n"
        "Labels to classify:\n"
    )
    for i, label in enumerate(labels_list, 1):
        prompt += f"{i}. {json.dumps(label)}\n"
    payload = {
        "messages": [
            {"role": "system", "content": "You respond only with a valid JSON array of strings. No markdown, no explanation."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    try:
        resp = requests.post(
            AZURE_OPENAI_CHAT_URL,
            headers={"Content-Type": "application/json", "Accept": "application/json", "api-key": AZURE_OPENAI_API_KEY},
            json=payload,
            timeout=(10, 30),
        )
        if resp.status_code != 200:
            logger.warning("LLM heading detection HTTP %s: %s", resp.status_code, resp.text[:200])
            return set()
        data = resp.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```\w*\n?", "", content).strip()
            content = re.sub(r"\n?```\s*$", "", content).strip()
        arr = json.loads(content)
        if not isinstance(arr, list):
            return set()
        exclude = {str(x).strip() for x in arr if x}
        logger.info("LLM detected %d heading label(s) to exclude: %s", len(exclude), list(exclude)[:5])
        return exclude
    except Exception as e:
        logger.warning("LLM heading detection failed, using config fallback: %s", e)
        return set()


def _read_excel_grid(file_path: str) -> pd.DataFrame:
    """Read first sheet as raw grid (no header)."""
    return pd.read_excel(file_path, header=None, sheet_name=0, engine=None)


def _cell_text(val) -> str:
    """Normalize cell value to string; empty or invalid -> empty string."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if s.lower() == "nan" or s.startswith("Unnamed"):
        return ""
    return s


def _detect_header_row(df: pd.DataFrame) -> int | None:
    """
    Find the row that looks most like a rate table header by semantic match count.
    Scans rows 0..MAX_SCAN_ROWS_FOR_HEADER; returns the row with the most system-field
    matches (>= MIN_HEADER_ROW_MATCHES). On tie, prefers the bottom-most row (header just above data).
    """
    if len(df) == 0 or len(df.columns) == 0:
        return None
    max_scan = min(MAX_SCAN_ROWS_FOR_HEADER, len(df))
    max_c = min(MAX_HEADER_COLS, len(df.columns))
    max_cell_len_for_header = 80

    best_row: int | None = None
    best_count = 0
    for r in range(max_scan):
        texts = []
        for c in range(max_c):
            t = _cell_text(df.iloc[r, c])
            if t and len(t) <= max_cell_len_for_header:
                texts.append(t)
        if len(texts) < 1:
            continue
        matches = match_labels_to_system_fields(
            texts, similarity_threshold=SIMILARITY_THRESHOLD
        )
        num_matches = len(matches)
        if num_matches >= MIN_HEADER_ROW_MATCHES and num_matches >= best_count:
            # Prefer bottom-most row on tie (header usually just above data)
            best_count = num_matches
            best_row = r
    if best_row is not None:
        logger.info("Header row: row %d (%d semantic matches).", best_row, best_count)
    return best_row


def _extract_semantic_units(
    df: pd.DataFrame,
    start_row: int = 0,
    max_rows: int | None = None,
) -> list[tuple[int, int, str]]:
    """
    Extract (row, col, text) for cells that look like metadata.
    If max_rows is set, only include rows in [start_row, start_row + max_rows).
    """
    units = []
    max_r = len(df)
    max_c = len(df.columns) if len(df) > 0 else 0
    end_row = min(max_r, start_row + max_rows) if max_rows is not None else max_r
    for r in range(start_row, end_row):
        for c in range(max_c):
            try:
                val = df.iloc[r, c]
            except Exception:
                continue
            text = _cell_text(val)
            if not text or _should_exclude_label(text):
                continue
            units.append((r, c, text))
    return units


def _extract_merged_column_labels(
    df: pd.DataFrame,
    max_rows: int = MAX_MERGED_HEADER_ROWS,
    start_row: int = 0,
) -> list[tuple[int, str]]:
    """
    Build one label per column by merging non-empty cell text from rows
    [start_row, start_row + max_rows). Handles multi-row headers and
    "header far down" templates (e.g. Roberts Europe at row 18).
    Returns list of (col_index, merged_label).
    """
    if len(df) == 0 or len(df.columns) == 0:
        return []
    out = []
    end_row = min(start_row + max_rows, len(df))
    for c in range(len(df.columns)):
        parts = []
        for r in range(start_row, end_row):
            t = _cell_text(df.iloc[r, c])
            if t:
                parts.append(t)
        merged = " ".join(parts).strip()
        if merged:
            out.append((c, merged))
    return out


def _extract_merged_row_labels(
    df: pd.DataFrame, max_cols: int = MAX_MERGED_HEADER_COLS
) -> list[tuple[int, str]]:
    """
    Build one label per row by merging non-empty cell text from the first max_cols.
    For row-oriented layouts. Returns list of (row_index, merged_label).
    """
    if len(df) == 0 or len(df.columns) == 0:
        return []
    out = []
    for r in range(len(df)):
        parts = []
        for c in range(min(max_cols, len(df.columns))):
            t = _cell_text(df.iloc[r, c])
            if t:
                parts.append(t)
        merged = " ".join(parts).strip()
        if merged:
            out.append((r, merged))
    return out


def _infer_layout(matches: list[tuple[int, int, str, float]]) -> tuple[str, int]:
    """
    matches: list of (row, col, system_field_key, score).
    Returns ("row", header_col) or ("column", header_row).
    """
    if not matches:
        return "column", 0

    row_counts: dict[int, int] = {}
    col_counts: dict[int, int] = {}
    for r, c, _key, _score in matches:
        row_counts[r] = row_counts.get(r, 0) + 1
        col_counts[c] = col_counts.get(c, 0) + 1

    best_row = max(row_counts, key=row_counts.get) if row_counts else 0
    best_col = max(col_counts, key=col_counts.get) if col_counts else 0
    count_row = row_counts.get(best_row, 0)
    count_col = col_counts.get(best_col, 0)

    if count_row >= count_col:
        return "column", best_row
    return "row", best_col


def _column_display_name(
    df: pd.DataFrame, row: int, col: int, fallback: str, header_row_for_name: int | None
) -> str:
    """
    Prefer the cell at header_row_for_name for downstream (e.g. pandas header=2).
    Otherwise return fallback (e.g. merged label).
    """
    if header_row_for_name is not None and 0 <= header_row_for_name < len(df):
        name = _cell_text(df.iloc[header_row_for_name, col])
        if name:
            return name
    return fallback


def _build_matched_and_weights(
        df: pd.DataFrame,
        units: list[tuple[int, int, str]],
        label_matches: list[tuple[int, str, float]],
        layout: str,
        header_index: int,
        header_row_for_name: int | None = None,
) -> tuple[list[dict], list[str]]:
    """
    From semantic units and their matches, build:
    - matched_fields: [{"excel_column": str, "system_field_key": str}, ...] in display order
    - ordered_weight_cols: list of weight/slab labels in order

    label_matches: list of (unit_index, system_field_key, score) from match_labels_to_system_fields.
    If header_row_for_name is set, excel_column is taken from that row for downstream compatibility.
    """
    # (row, col) -> (system_field_key, score)
    matched_cells: dict[tuple[int, int], tuple[str, float]] = {}
    for unit_idx, sys_key, score in label_matches:
        if unit_idx >= len(units):
            continue
        r, c, _ = units[unit_idx]
        key = (r, c)
        if key not in matched_cells or matched_cells[key][1] < score:
            matched_cells[key] = (sys_key, score)

    # Units on the header row (column layout) or header column (row layout), sorted by position
    if layout == "column":
        header_units = [(r, c, t) for r, c, t in units if r == header_index]
        header_units.sort(key=lambda x: x[1])  # by col
    else:
        header_units = [(r, c, t) for r, c, t in units if c == header_index]
        header_units.sort(key=lambda x: x[0])  # by row

    matched_fields = []
    weight_labels = []
    for r, c, text in header_units:
        key = (r, c)
        if key in matched_cells:
            sys_key, _ = matched_cells[key]
            excel_col = _column_display_name(
                df, r, c, text, header_row_for_name
            )
            matched_fields.append({"excel_column": excel_col, "system_field_key": sys_key})
        elif WEIGHT_PATTERN.match(text):
            weight_labels.append(text)

    return matched_fields, weight_labels


def select_rate_card_fields(file_path: str) -> str:
    """Matches Excel labels (rows or columns) with DHL field descriptions via embeddings."""
    logger.info(f"Tool select_rate_card_fields called for: {file_path}")

    try:
        if not file_path or not isinstance(file_path, str):
            return "Error: Invalid file path provided."

        file_path = os.path.abspath(os.path.expanduser(file_path))

        if not os.path.exists(file_path):
            error_msg = f"Error: File not found at path: {file_path}"
            logger.error(error_msg)
            return error_msg

        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in [".xls", ".xlsx"]:
            error_msg = (
                f"Error: Invalid file format. Expected Excel file (.xls or .xlsx), "
                f"but received: {file_ext}. Please upload a valid Excel document."
            )
            logger.error(error_msg)
            return error_msg

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            error_msg = "Error: The uploaded file is empty or corrupted. Please upload a valid Excel file."
            logger.error(error_msg)
            return error_msg

        try:
            df = _read_excel_grid(file_path)
        except Exception as read_error:
            error_type = type(read_error).__name__
            logger.error(f"Failed to read Excel file: {error_type}: {read_error}")
            if "No such file" in str(read_error) or "FileNotFoundError" in error_type:
                error_msg = f"Error: File not found or cannot be accessed: {file_path}"
            elif "Permission" in error_type or "permission" in str(read_error).lower():
                error_msg = "Error: Permission denied. Cannot read file."
            else:
                error_msg = (
                    f"Error: Failed to read Excel file. {str(read_error)}. "
                    "Please verify the file is a valid Excel document (.xls or .xlsx)."
                )
            return error_msg

        # Detect "header far down" (e.g. Roberts Europe ratesheet with header at row 18)
        detected_header = _detect_header_row(df)
        if detected_header is not None:
            header_start = detected_header
            header_window = HEADER_WINDOW_ROWS
            use_header_far_down = True
            logger.info("Detected header row at index %d (header-far-down template).", detected_header)
        else:
            header_start = 0
            header_window = MAX_HEADER_ROWS
            use_header_far_down = False

        units = _extract_semantic_units(df, start_row=header_start, max_rows=header_window)
        if not units:
            error_msg = "Error: No valid header labels found in the Excel file (rows %d–%d, up to %d columns)." % (
                header_start,
                header_start + header_window - 1,
                MAX_HEADER_COLS,
            )
            return error_msg

        # --- Strategy 1: merged column labels (multi-row headers or header far down) ---
        merged_col_labels = _extract_merged_column_labels(
            df, MAX_MERGED_HEADER_ROWS, start_row=header_start
        )
        merged_col_labels = [(c, label) for c, label in merged_col_labels if not _should_exclude_label(label)]
        merged_col_texts = [label for _c, label in merged_col_labels]
        merged_col_matches = match_labels_to_system_fields(
            merged_col_texts, similarity_threshold=SIMILARITY_THRESHOLD
        ) if merged_col_texts else []

        # --- Strategy 2: merged row labels (row-oriented layout) ---
        merged_row_labels = _extract_merged_row_labels(df, MAX_MERGED_HEADER_COLS)
        merged_row_labels = [(r, label) for r, label in merged_row_labels if not _should_exclude_label(label)]
        merged_row_texts = [label for _r, label in merged_row_labels]
        merged_row_matches = match_labels_to_system_fields(
            merged_row_texts, similarity_threshold=SIMILARITY_THRESHOLD
        ) if merged_row_texts else []

        # --- Strategy 3: flat cell-by-cell (original) ---
        label_texts = [t for _r, _c, t in units]
        flat_matches = match_labels_to_system_fields(
            label_texts, similarity_threshold=SIMILARITY_THRESHOLD
        )

        # Choose best strategy: prefer merged column if it gives enough matches (multi-row template)
        use_merged_column = (
            len(merged_col_matches) >= MIN_MERGED_MATCHES_TO_PREFER
            and len(merged_col_matches) >= len(merged_row_matches)
        )
        use_merged_row = (
            not use_merged_column
            and len(merged_row_matches) >= MIN_MERGED_MATCHES_TO_PREFER
            and len(merged_row_matches) > len(merged_col_matches)
        )

        if use_merged_column:
            # One unit per column: (0, col_idx, merged_label) for consistent layout index
            units = [(0, c, label) for c, label in merged_col_labels]
            label_matches = merged_col_matches
            layout, header_index = "column", 0
            header_row_for_name = (
                header_start if use_header_far_down else HEADER_ROW_FOR_DOWNSTREAM
            )
            logger.info(
                "Using merged column header strategy (%d matches).",
                len(merged_col_matches),
            )
        elif use_merged_row:
            # One unit per row: (row_idx, 0, merged_label)
            units = [(r, 0, label) for r, label in merged_row_labels]
            label_matches = merged_row_matches
            layout, header_index = "row", 0
            header_row_for_name = None
            logger.info(
                "Using merged row header strategy (%d matches).",
                len(merged_row_matches),
            )
        else:
            # Fall back to flat cell matching
            label_matches = flat_matches
            matches_with_pos = [
                (units[i][0], units[i][1], sk, sc)
                for i, sk, sc in label_matches
            ]
            layout, header_index = _infer_layout(matches_with_pos)
            header_row_for_name = None
            logger.info(
                "Using flat cell strategy; layout=%s, header index=%s.",
                layout,
                header_index,
            )

        if not label_matches:
            logger.warning("No semantic matches above threshold; returning empty field list.")
            return json.dumps(
                {"fields": [], "column_mapping": [], "weight_columns": []},
                indent=2,
            )

        matched_fields, ordered_weight_cols = _build_matched_and_weights(
            df, units, label_matches, layout, header_index, header_row_for_name
        )

        # Exclude labels that are too long or bullet/note lines (safety net for merged labels)
        matched_fields = [m for m in matched_fields if not _should_exclude_label(m["excel_column"])]

        # Exclude document/section headings: LLM classifies headings vs column headers; fallback to config if LLM unavailable
        llm_exclude = _llm_detect_heading_labels(matched_fields)
        if llm_exclude:
            matched_fields = [m for m in matched_fields if m["excel_column"] not in llm_exclude]
        else:
            config_substrings = _get_exclude_heading_substrings()
            if config_substrings:
                matched_fields = [m for m in matched_fields if not _is_heading_label(m["excel_column"], config_substrings)]

        # Deduplicate by system_field_key (keep first occurrence)
        seen_keys = set()
        unique_matched = []
        for m in matched_fields:
            k = m["system_field_key"]
            if k not in seen_keys:
                seen_keys.add(k)
                unique_matched.append(m)

        system_field_keys = [m["system_field_key"] for m in unique_matched]
        final_fields = system_field_keys + ordered_weight_cols

        # Store mapping for SAP / downstream (system_field_key -> excel_column_name).
        # For origin/destination zone columns, normalize composite merged headers to the
        # first segment (e.g. "Origin country Minimum € / km ..." -> "Origin country")
        # so SAP upload can find the column when Excel is read with a single header row.
        _ORIGIN_DEST_ZONE_KEYS = frozenset({
            "SOURCELOC_ZONE", "SOURCELOC", "SOURCELOC_CNTRY", "SRCLOC_UNLOCD", "SRC_UNLOCD",
            "DESTLOC_ZONE", "DESTLOC", "DESTLOC_CNTRY", "DESTLOC_UNLOCD", "DEST_UNLOCD",
        })
        _COMPOSITE_HEADER_MARKERS = (" Minimum", " € ", " €/km", " €/ km", "/km")

        def _normalize_origin_dest_column(excel_col: str, system_key: str) -> str:
            if system_key not in _ORIGIN_DEST_ZONE_KEYS or not excel_col:
                return excel_col
            for marker in _COMPOSITE_HEADER_MARKERS:
                idx = excel_col.find(marker)
                if idx > 0:
                    return excel_col[:idx].strip()
            return excel_col

        column_mapping = {}
        for m in unique_matched:
            key = m["system_field_key"]
            excel_col = _normalize_origin_dest_column(m["excel_column"], key)
            column_mapping[key] = excel_col
        session_id = get_session_id()
        store_column_mapping(session_id, column_mapping)
        store_rate_card_path(session_id, file_path)
        if use_header_far_down:
            store_excel_header_row(session_id, header_start)

        # Build user-visible mapping: Excel column -> system field (key + description)
        mapping_display = []
        for m in unique_matched:
            sys_key = m["system_field_key"]
            entry = SYSTEM_FIELD_MAPPING.get(sys_key)
            description = entry[1] if entry else sys_key
            mapping_display.append({
                "excel_column": column_mapping.get(sys_key, m["excel_column"]),
                "system_field_key": sys_key,
                "system_field_description": description,
            })
        response = {
            "fields": final_fields,
            "column_mapping": mapping_display,
            "weight_columns": ordered_weight_cols,
        }
        store_field_selection_result(session_id, response)
        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(f"Field selection error: {e}")
        return f"Error matching fields: {str(e)}"


field_selection_tool = FunctionTool(select_rate_card_fields)
 