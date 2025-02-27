with open('../labelled_data/vdi_statements.txt', 'r') as f:
    lines = f.readlines()

with open('../labelled_data/cleaned_vdi_statements.txt', 'w') as f:
    for line in lines:
        line = line.split('.', maxsplit=1)[1]
        line = line.strip()
        if line:
            f.write(line + '\n')
