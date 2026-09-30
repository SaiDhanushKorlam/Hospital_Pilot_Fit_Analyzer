# Hospital Pilot Fit Analyzer

A small Streamlit research prototype that turns manually collected public hospital evidence into a structured assessment for a knee/hip replacement rehabilitation pilot. It uses no patient data and does not scrape websites.

## Install

Use Python 3.10+:

```bash
"C:\\Program Files\\Python311\\python.exe" -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

## Configure the API key

Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. The optional `GEMINI_MODEL` defaults to `gemini-3.1-flash-lite`. Never commit the `.env` file or an API key.

## Run

```bash
.venv\\Scripts\\python.exe -m streamlit run app.py --server.headless true --server.port 8502
```

Open the local URL shown by Streamlit. On the Analyze page, enter a hospital name, website, and evidence copied from a public source. The model receives only that supplied text and must return structured JSON. Missing keys, API failures, invalid JSON, empty evidence, and oversized evidence are shown as errors rather than converted into a fake result.

## Evaluation modes

Open the **Evaluation** page and choose a dataset:

- **Use Demo Dataset** loads the bundled audited `hospital_pilot_dataset_corrected.csv` immediately. The original `hospital_pilot_dataset.csv` remains preserved as the historical pre-audit fixture and source of the original 7/10 baseline.
- **Upload CSV** accepts a custom CSV through Streamlit's uploader. The app shows the uploaded filename and validates the required columns before making any model calls.

Required columns are:

```text
hospital,city,website,evidence_text
```

For measured evaluation, also include:

```text
known_ground_truth
```

When `known_ground_truth` is present, the app runs Evaluation mode. For each row it sends only `hospital`, `website`, and `evidence_text` to the same analyzer used by the Analyze page. It parses and compares `known_ground_truth` only after the model response. It reports case-level correct/incorrect counts, case error rate, and field accuracy for joint replacement, knee replacement, hip replacement, and rehabilitation.

When the four research columns are present but `known_ground_truth` is absent, the app runs **Batch Analysis — no ground truth provided, so accuracy cannot be measured.** It analyzes every row, displays validated structured results, and provides a CSV export. It never invents labels or accuracy.

Use **Download Evaluation CSV Template** for a starter file. The template contains one example row and the required columns but does not contain final ground-truth answers. Accuracy can only be measured when manually verified ground truth is supplied.

The demo Evaluation page includes a separate **Adversarial Robustness Test**. It runs a controlled before-safeguard prompt and then the current safeguarded analyzer for generic joint-replacement wording. If the first test passes, it also tests whether career volume is incorrectly converted into annual hospital-wide volume. These tests are separate from the 10-case accuracy evaluation and are only reported as failures when the model actually makes the unsupported inference.

## Dataset

The original pre-audit fixture is preserved in `hospital_pilot_dataset.csv`. Evaluation uses `hospital_pilot_dataset_corrected.csv`, where every scored positive label is explicitly supported by `evidence_text`; the three failed rows were updated with source-backed explicit wording without changing ground truth. Neither file is required for custom Batch Analysis uploads.

The recorded pre-audit baseline was 7/10 correct with a 30% case error rate. After auditing the evidence and applying the explicit procedure-synonym validation, the corrected 10-case run reached 10/10 correct with a 0% case error rate and 100% field accuracy for joint replacement, knee replacement, hip replacement, and rehabilitation. No genuine model failure remained on the corrected fixture, so the separate adversarial test demonstrates that ambiguous generic wording remains unclear rather than being counted as a 10-case evaluation failure.

The adversarial report includes the failure scenario, before result, after result, safeguard applied, failure-detected status, and final status. In the verified run, no failure was observed in the adversarial tests; they were retained as robustness checks. Adversarial rows are exported separately from the normal hospital rows in the `Adversarial tests` sheet of the evaluation Excel export.

## Known limitations

- Results depend on the selected model and are not a clinical or medical recommendation.
- The prototype does not fetch, verify, or archive source pages.
- The evaluation measures a small, hand-labelled dataset and is not a benchmark of general model quality.
- The output provides a suggested contact, not an assumed decision-maker. A person is described as having authority only when the supplied evidence explicitly establishes authority for pilots, partnerships, procurement, clinical programs, or an equivalent responsibility.
- Always verify model output against the original source before using it in business decisions.
