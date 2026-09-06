import json

def join_segments(*args: str| list[str], seperator: str='\n\n\n') -> str:
    all_segments = []

    for arg in args:
        if isinstance(arg, list):
            all_segments.extend(arg)
        else:
            all_segments.append(strip_string(str(arg)))

    return strip_string(seperator.join(all_segments))

def strip_string(s: str) -> str:
    return s.strip()

def extract_json_from_output(model_output: str) -> dict | None:
    if not isinstance(model_output, str) or not model_output.strip():
        return None

    try:
        value = json.loads(model_output)
    except json.JSONDecodeError:
        pass
    else:
        return value if isinstance(value, dict) else None

    decoder = json.JSONDecoder()
    last_object = None
    position = 0
    while (start := model_output.find('{', position)) != -1:
        try:
            value, end = decoder.raw_decode(model_output, start)
        except json.JSONDecodeError:
            position = start + 1
        else:
            if isinstance(value, dict):
                last_object = value
            position = end
    return last_object

def normalize_label(value: object) -> str | None:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str) and value.strip().lower() in {'true', 'false'}:
        return value.strip().capitalize()
    return None

def to_readable_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
