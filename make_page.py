"""Pack the hand-written weights into one offline HTML page for the phone.
All weights are whole numbers, so they are stored as gzipped int16."""
import base64
import gzip

import numpy as np

from build_weights import build

ws = build()
blob = b""
for w in ws:
    assert np.array_equal(w, w.astype("<i2")), "weights must be whole numbers"
    blob += w.astype("<i2").tobytes()

page = open("page_template.html", encoding="utf-8").read()
page = page.replace("__WEIGHTS__", base64.b64encode(gzip.compress(blob, 9)).decode())
open("snake.html", "w", encoding="utf-8").write(page)
print("snake.html", round(len(page) / 1024), "KB")
