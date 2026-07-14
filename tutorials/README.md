# SIAR tutorials

Worked examples, in order. Each is a Markdown walkthrough; 01 and 02 come with a runnable script
that produces the output quoted in the text. 03 needs no script — it works on the results 02
leaves behind.

**In a hurry?** [**quick_start/**](quick_start/) is the whole product in five commands — build a
sample corpus, train, score, and view it in the browser — on a machine where you have only just
cloned the repo.

| | Tutorial | What you learn |
|---|---|---|
| 01 | [Your first detector](01-first-detector.md) | The whole pipeline, in Python, on synthetic audio where you *know* the answer. Run this first. |
| 02 | [Train, run, and see the results](02-train-run-dashboard.md) | The actual product: three commands, a database, and a dashboard. |
| 03 | [View your results in the browser](03-view-results-in-the-browser.md) | The dashboard on its own: `siar dash`, how to triage, and why it opens empty. |

Run any tutorial's script directly:

```bash
pip install -e .
python tutorials/01_first_detector.py
python tutorials/02_train_run_dashboard.py
```

## Why the first tutorial uses fake audio

Because it is the only way to *check* the thing works.

Unsupervised anomaly detection has an uncomfortable property: it will always give you an answer.
Point it at a folder, and it will confidently draw boxes. If you have no labels — and the whole
premise is that you don't — you have no way to tell a working detector from a broken one. The
boxes look equally plausible either way.

So tutorial 01 builds a corpus where the truth is known by construction: pink noise for
"normal", and one file with a chirp planted at a time and frequency we chose. Then we can ask
the only question that matters — *did it find the thing we hid?* — and get a real answer.

Do the same before you trust SIAR on your own data. Plant something in a copy of your corpus and
confirm it comes back.
