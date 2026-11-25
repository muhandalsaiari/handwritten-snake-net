"""Pack the v5 network into one offline HTML page. Layers that repeat (the
flood and trace layers) are stored once. All weights are whole numbers."""
import base64
import gzip
import hashlib
import json
import pickle

import numpy as np

mats = pickle.load(open("v5_net.pkl", "rb"))
unique, order, seen = [], [], {}
for W, b in mats:
    W = W.tocsr()
    W.sort_indices()
    for arr in (W.data, b):
        assert np.array_equal(arr, np.round(arr)) and np.abs(arr).max(initial=0) < 32767
    key = hashlib.sha1(W.indptr.tobytes() + W.indices.tobytes() + W.data.tobytes()
                       + b.tobytes() + str(W.shape).encode()).hexdigest()
    if key not in seen:
        seen[key] = len(unique)
        unique.append((W, b))
    order.append(seen[key])

blob, meta = b"", []
for W, b in unique:
    parts = [W.indptr.astype("<i4"), W.indices.astype("<i2"), W.data.astype("<i2"), b.astype("<i2")]
    meta.append({"rows": W.shape[0], "cols": W.shape[1], "nnz": int(W.nnz)})
    for p in parts:
        blob += p.tobytes()
        if len(blob) % 4:
            blob += b"\0" * (4 - len(blob) % 4)

page = open("page_template_v5.html", encoding="utf-8").read()
page = page.replace("__META__", json.dumps({"layers": meta, "order": order}))
page = page.replace("__WEIGHTS__", base64.b64encode(gzip.compress(blob, 9)).decode())
open("snake.html", "w", encoding="utf-8").write(page)
print(f"{len(mats)} layers, {len(unique)} unique, page {len(page) / 1024:.0f} KB")
