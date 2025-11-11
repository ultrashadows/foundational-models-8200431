import json

input_path = "./evaluation/test.json"
output_path = "./evaluation/clean.json"
field = "Texto Integral"
strip_whitespace = True
preserve_original_order = True

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

lengths = []
for i, obj in enumerate(data):
    if not isinstance(obj, dict) or field not in obj:
        continue
    val = obj[field]
    if not isinstance(val, str):
        val = "" if val is None else str(val)
    text = val.strip() if strip_whitespace else val
    lengths.append((i, len(text)))

lengths.sort(key=lambda t: (t[1], t[0]))
middle_indices = [idx for (idx, _) in lengths[350:650]]

if preserve_original_order:
    keep = set(middle_indices)
    selected = [data[i] for i in range(len(data)) if i in keep]
else:
    selected = [data[i] for i in middle_indices]

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(selected, f, ensure_ascii=False, indent=2)
