"""
loader.py
----------
Responsible for loading and validating dataset files (CSV / Excel only)
before they are passed to the rest of the pipeline (profiling, analysis, etc.)
"""

import os

import pandas as pd

# Only these extensions are allowed. Anything else is rejected immediately.
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# Tried in order. "utf-8-sig" reads plain UTF-8 too (and tolerates the BOM that Excel adds).
# "cp1256" is the Arabic Windows encoding (Excel on Arabic Windows saves CSV with it);
# "latin1" accepts any bytes, so it is the last resort.
CSV_ENCODINGS = ["utf-8-sig", "cp1256", "latin1"]

# Separators we recognise (Excel in many locales saves CSV with ";" instead of ",")
CSV_SEPARATORS = [",", ";", "\t", "|"]


class FileValidationError(Exception):
    """Raised when a file fails validation (wrong type, empty, corrupted, etc.)"""
    pass


def load_file(file_path: str, sheet_name=0) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a pandas DataFrame.

    Steps:
    1. Check that the file exists and its extension is allowed.
    2. Try to actually read the file with pandas (this catches files that
       were renamed to look like a CSV/Excel but aren't really).
    3. Light cleanup: strip column names, remove completely empty rows/columns.
    4. Validate the resulting DataFrame (not empty, has columns).

    Args:
        file_path: path to a .csv, .xlsx or .xls file.
        sheet_name: Excel only. Sheet name or position (default: first sheet).
                    The list of all sheets is stored in df.attrs["available_sheets"].

    Raises:
        FileValidationError: if the file is missing, the type is not allowed,
                             the file cannot be parsed, or the content is invalid.
    """
    if not os.path.isfile(file_path):
        raise FileValidationError(f"File not found: '{file_path}'")

    _check_extension(file_path)

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".csv":
            df = _load_csv(file_path)
        else:  # .xlsx or .xls
            df = _load_excel(file_path, sheet_name)
    except FileValidationError:
        raise
    except pd.errors.EmptyDataError:
        raise FileValidationError("The uploaded file is empty.")
    except ImportError as e:
        # Missing optional library (openpyxl for .xlsx, xlrd for .xls): not a corrupted file
        raise FileValidationError(
            f"A required library is missing to read '{ext}' files. Details: {e}"
        )
    except Exception as e:
        # Any parsing failure means the content doesn't actually match
        # the extension (e.g. a .csv that isn't really a CSV).
        raise FileValidationError(
            f"Could not read '{file_path}'. The file seems corrupted "
            f"or does not match its extension. Details: {e}"
        )

    # Light first-pass cleanup: tidy column names and drop empty rows/columns
    # (Excel exports often contain blank trailing columns and rows).
    # Column names are converted to str because Excel headers can be numbers.
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=0, how="all")
    if len(df) > 0:  # on a file with no data rows, every column would look "empty"
        df = df.dropna(axis=1, how="all")

    validate_file(df)

    return df


def _check_extension(file_path: str) -> None:
    """Reject the file immediately if its extension is not allowed."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"File type '{ext}' is not supported. "
            f"Allowed types are: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )


def _detect_separator(file_path: str, encoding: str) -> str:
    """Pick the separator that appears most often in the header line."""
    with open(file_path, "r", encoding=encoding, errors="replace", newline="") as f:
        header = ""
        for line in f:
            if line.strip():
                header = line
                break

    counts = {sep: header.count(sep) for sep in CSV_SEPARATORS}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _load_csv(file_path: str) -> pd.DataFrame:
    """
    Read a CSV trying several encodings and detecting the separator.
    The encoding that worked is stored in df.attrs["encoding"].
    """
    for encoding in CSV_ENCODINGS:
        try:
            sep = _detect_separator(file_path, encoding)
            df = pd.read_csv(file_path, encoding=encoding, sep=sep, low_memory=False)
            df.attrs["encoding"] = encoding
            return df
        except UnicodeDecodeError:
            continue  # try the next encoding

    # Not reachable in practice (latin1 accepts any bytes), kept as a safety net
    raise FileValidationError("Could not decode the CSV file with any supported encoding.")


def _load_excel(file_path: str, sheet_name) -> pd.DataFrame:
    """Read one sheet of an Excel file and remember the names of all sheets."""
    with pd.ExcelFile(file_path) as workbook:
        sheets = workbook.sheet_names
        df = workbook.parse(sheet_name)
    df.attrs["available_sheets"] = sheets
    return df


def validate_file(df: pd.DataFrame) -> bool:
    """
    Validate that the loaded DataFrame is usable.

    Raises:
        FileValidationError: if the DataFrame is empty or has no columns.

    Returns:
        True if validation passes.
    """
    if df is None or len(df.columns) == 0:
        raise FileValidationError("The uploaded file has no columns.")

    if df.empty:
        raise FileValidationError("The uploaded file is empty (0 rows).")

    return True