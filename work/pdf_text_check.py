import sys

import pymupdf

path = sys.argv[1]
pages = [int(value) for value in sys.argv[2:]]
document = pymupdf.open(path)
for number in pages:
    print(f"===== page {number} =====")
    print(document[number - 1].get_text()[:2500])
