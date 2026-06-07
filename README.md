# Discharge Medication Review

A Streamlit application for reviewing de-identified CPRS medication transitions across admission reconciliation, inpatient orders, discharge reconciliation, and clinical notes.

## Privacy Warning

Do not enter PHI, patient identifiers, names, dates of birth, medical record numbers, addresses, phone numbers, or other identifiable details.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser, usually at `http://localhost:8501`.

## Optional LLM Sanity Check

The rule-based parser runs locally. The optional LLM check is only enabled when an OpenAI API key is configured and the user confirms the input is de-identified.

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL="gpt-5.2"
streamlit run app.py
```

The LLM receives the structured medication transition output, not a free-form chart review request. Do not enter PHI.

## What It Flags

- New medications at discharge
- Home medications stopped at discharge
- Inpatient medications not continued at discharge
- Dose changes
- Frequency changes
- High-risk medications

## Review Workflow

After analysis, the CPRS paste boxes collapse so the review stays focused. Results are organized into:

- Worklist: grouped clinical action items
- Timeline: visual admission / inpatient / discharge medication status
- Signout: copyable resident-style summary
- LLM Check: optional sanity check and follow-up questions
- Parser Check: parsed counts and medication names that may need a manual glance
- Raw Table: export-ready structured output

This is clinical decision support only. It does not replace formal medication reconciliation or clinical judgment.
