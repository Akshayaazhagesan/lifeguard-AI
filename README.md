# LifeGuard AI

This project predicts organ failure in ICU patients using AI.

How to run:

1. Turn on virtual environment: `.\venv\Scripts\activate`
2. Install packages: `pip install -r requirements.txt`
3. Train the model: `python train.py`
4. Start the API server: `uvicorn main:app --reload`
5. Use the API to get predictions.
