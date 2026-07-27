# Streamlit Annotation Changes — July 20, 2026

## Summary

The SKU annotation tool was updated to make reviewing easier, support multiple
annotators safely, and use pipeline prediction output for OCR suggestions.

## Annotation interface

- The OCR suggestion and expected-SKU controls now appear together at the top.
- Annotation metadata now appears in a separate section underneath.
- The OCR debug dropdown was removed.
- Reviewers can classify a crop as either **Scorable** or **Non-scorable**.
- **Mark X (not visible)** records an unreadable or invisible SKU as `X` and
  excludes it from OCR accuracy.
- Scorable SKUs must contain exactly **6 or 10 digits**.
- Saved values are automatically classified as `6-digit`, `10-digit`, or
  `not-visible`.

## Multiple annotators

- A reviewer now enters only their name; reviewer numbers and slots were removed.
- Names are case-insensitive. For example, `Avinash`, `AVINASH`, and `avinash`
  are treated as the same person.
- New reviewers automatically receive a balanced portion of the remaining work.
- Work is balanced by the number of SKU crops.
- All crops from one source image always stay assigned to the same reviewer.
- Images that already contain saved reviews stay with their current reviewer.
- Returning reviewers enter the same name and resume at their first unfinished crop.
- A **Manage annotators** section allows a reviewer to be removed.
- Removing a reviewer preserves their saved labels and redistributes their
  remaining image groups.

## Safe concurrent reviewing

- Annotation saves now update one row at a time.
- File locking and atomic writes prevent simultaneous reviewers from overwriting
  each other's work.
- Reviewer assignments are also saved with locking.
- Closing the browser loses only the current unsaved edit; saved annotations remain.

## OCR suggestions and `predictions.json`

- Streamlit now reads:

  `/Users/avinash.patel/Downloads/HomeDepotCV/predictions.json`

- If the JSON includes an image filename, predictions are grouped by that filename.
- The current JSON is a bare list without a filename, so the app identifies its
  source image by comparing detection classes and bounding boxes with annotation rows.
- It correctly associated the supplied file with:

  `1770339044281_0244_1031_07-020.jpg`

- Detection boxes are matched to annotation regions using class and bounding-box
  overlap.
- Only valid 6- or 10-digit values from `ocr_words` become OCR suggestions.
- The supplied prediction `1002883543` was matched to the correct RDC annotation.
- For an image represented in `predictions.json`, JSON output takes priority over
  old crop-level OCR suggestions.
- Images not represented in the JSON continue to use their legacy precomputed hints.
- Accuracy metrics use the same effective prediction shown to the reviewer.

### Important OCR note

Streamlit no longer needs to call Google OCR for each displayed crop when prediction
JSON is available. However, the supplied JSON identifies its OCR source as
`seg+google_ocr`, so the upstream pipeline that created this particular file still
used Google OCR internally.

## Sharing with the annotation team

Annotators do not need to log into the 5090 machine. They can use a browser while
connected to the corporate VPN:

- Team application: <http://172.16.20.108:8503>
- Local application: <http://127.0.0.1:8501>

A single centralized application should be used instead of distributing separate
local copies, because separate copies would create conflicting CSV files and
reviewer assignments.

## Local and 5090 deployment

- All changes were applied to the local application.
- The same code and `predictions.json` were deployed to the 5090 host.
- The remote Streamlit service was restarted successfully.
- The remote service returned HTTP 200.
- Deployment used code-only mode, so existing remote annotations were preserved.

## Validation

- 11 focused automated tests passed.
- Tests cover review status handling, `X`, 6/10-digit validation, dynamic reviewer
  allocation, case-insensitive names, annotator deletion, safe row saves, and
  prediction-to-annotation matching.
- The supplied `predictions.json` was verified against the real annotation CSV.
- Local and remote Streamlit health checks passed.

