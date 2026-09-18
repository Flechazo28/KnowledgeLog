import regex as re

param_regex = [
    r'{([ :_#.\-\w\d]+)}',
    r'{}'
]


def correct_single_template(template, user_strings=None):
    path_delimiters = {
        r'\s', r'\,', r'\!', r'\;', r'\:',
        r'\=', r'\|', r'\"', r'\'',
        r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({
        r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&',
    })

    template = template.strip()
    template = re.sub(r'\s+', ' ', template)
    tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)
    new_tokens = []
    for token in tokens:
        if re.match(r'^\d+$', token):
            token = '<*>'

        if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token):
            if token != '<*>/<*>':
                token = '<*>'
        new_tokens.append(token)

    template = ''.join(new_tokens)

    # Treat a standalone empty parenthesized field as an omitted value.
    template = re.sub(r'(?<= )\(\)(?= )', '(<*>)', template)

    template = re.sub(r'\(\s*<\*>(?:\s+<\*>){3,}\s*\)', '( <*> )', template)

    template = re.sub(r'<\*>(?:##<\*>){2,}', '<*>', template)

    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break

    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        if prev == template:
            break
    while " #<*># " in template:
        template = template.replace(" #<*># ", " <*> ")

    while " #<*> " in template:
        template = template.replace(" #<*> ", " <*> ")

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")

    while "<*>#<*>" in template:
        template = template.replace("<*>#<*>", "<*>")

    while "<*>/<*>" in template:
        template = template.replace("<*>/<*>", "<*>")

    while "<*>@<*>" in template:
        template = template.replace("<*>@<*>", "<*>")

    while "<*>.<*>" in template:
        template = template.replace("<*>.<*>", "<*>")

    while ' "<*>" ' in template:
        template = template.replace(' "<*>" ', ' <*> ')

    while " '<*>' " in template:
        template = template.replace(" '<*>' ", " <*> ")

    while "<*> && <*>" in template:
        template = template.replace("<*> && <*>", "<*>")

    while "<*><*>" in template:
        template = template.replace("<*><*>", "<*>")
    return template
