import sys
with open('rewrite_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

start_idx = content.find('"""') + 3
end_idx = content.rfind('"""')

original_app_code = content[start_idx:end_idx].lstrip()

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(original_app_code)
