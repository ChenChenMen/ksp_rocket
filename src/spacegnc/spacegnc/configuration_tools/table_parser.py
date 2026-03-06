from pathlib import Path


def sanitize_content(content_str: str) -> str:
    """Sanitize content string by removing comments and striping whitespaces.

    If line break is included, sanitize comments from each line and
    concatenate the sanitized lines together while remove a pure comment line.

    if line break is not included, simply remove the comment part and return
    the sanitized string.
    """
    if "\n" in content_str:
        lines = content_str.strip().splitlines()
        sanitized_lines = []
        for line in lines:
            sanitized_line = line.split("#", 1)[0].strip()
            if sanitized_line:  # Skip pure comment lines
                sanitized_lines.append(sanitized_line)
        return "\n".join(sanitized_lines)
    else:
        return content_str.split("#", 1)[0].strip()


def parse_table_from_file(file_path: str | Path, transpose: bool = True) -> list[tuple[float, ...]]:
    """Parse a table from a file, ignoring comments and empty lines."""
    with open(file_path, "r") as file:
        table_str = file.read()

    lines = sanitize_content(table_str).splitlines()
    data = []
    for line in lines:
        sanitized_line = sanitize_content(line)
        if sanitized_line:  # Skip empty lines
            row = tuple(float(value) for value in sanitized_line.split())
            data.append(row)
    return data
