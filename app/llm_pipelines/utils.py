import re


def format_header(header: str) -> str:
    """Convert header into slug for source"""
    return header.replace(' ', '').replace('(', '').replace(')', '')


def get_markdown_header_paths(markdown_text: str) -> list[str]:
    """
    Parses a markdown string to find all header lines and returns a list of slash-separated paths for the leaf headers.

    A header is considered a "leaf" if it is not immediately followed by a
    header of a deeper level.
    """

    header_pattern = re.compile(r'^(#{1,6})\s+(.*)', re.MULTILINE)
    matches = header_pattern.finditer(markdown_text)

    headers = []
    for match in matches:
        level = len(match.group(1))
        text = match.group(2).strip().replace(' ', '-')
        headers.append({'level': level, 'text': text})

    if not headers:
        return []

    result_paths = []
    current_path = []

    for i, header in enumerate(headers):
        level = header['level']
        text = header['text']

        while len(current_path) >= level:
            current_path.pop()
        current_path.append(text)

        is_leaf = False
        is_last_header = i == len(headers) - 1

        if is_last_header:
            is_leaf = True
        else:
            next_header_level = headers[i + 1]['level']
            if next_header_level <= level:
                is_leaf = True

        if is_leaf:
            result_paths.append('/'.join(map(format_header, current_path)))

    return result_paths


def get_allowed_sources(material):
    """Get all possible sources from material, both in {filename} and {filename}#{header_path} format"""
    values = list(material.keys())
    for name, content in material.items():
        for header_path in get_markdown_header_paths(content):
            values.append(f'{name}#{header_path}')

    return values
