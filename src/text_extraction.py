import sys
import os
from pypdf import PdfReader


if not os.path.isdir(sys.argv[1]):
    raise ValueError(f"{sys.argv[1]} is not a directory!")

for filename in os.listdir(sys.argv[1]):
    reader = PdfReader(os.path.join(sys.argv[1], filename))
    text = []
    for page in reader.pages:
        text.append(page.extract_text() + "\n")
    with open(os.path.join(sys.argv[2], f"{filename}.txt"), "w", encoding="utf-8") as fp:
        fp.writelines(text)
