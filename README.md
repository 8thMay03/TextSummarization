# English Text Summarization with T5-small

Fine-tune `t5-small` for English abstractive summarization on CNN/DailyMail.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Project Structure

```text
TextSummarization/
  app/
    streamlit_app.py
  src/
    config.py
    data.py
    evaluate.py
    inference.py
    train.py
  data/
  models/
  outputs/
  requirements.txt
  README.md
```

## Train a Small Test Run

This command trains on a small subset so you can verify the full pipeline first.

```powershell
python -m src.train --max-train-samples 1000 --max-eval-samples 200 --epochs 1
```

The model is saved to:

```text
models/t5-small-cnn-dailymail
```

## Evaluate

```powershell
python -m src.evaluate --model-path models/t5-small-cnn-dailymail --max-test-samples 200
```

## Run Inference

```powershell
python -m src.inference --model-path models/t5-small-cnn-dailymail --text "Long article text here..."
```

## Start Demo App

```powershell
streamlit run app/streamlit_app.py
```

## Suggested Full Training Settings

After the small test run works, increase the sample counts or remove them:

```powershell
python -m src.train --epochs 3 --batch-size 4 --gradient-accumulation-steps 4
```
