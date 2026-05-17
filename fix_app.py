with open('app.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('\\"\\"\\"', '"""')

if text.startswith('\\\n'):
    text = text[2:]

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(text)
