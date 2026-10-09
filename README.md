# Encontrar — faculty attendance dashboard

## Run
1. Terminal 1: `npm run dev`
2. Terminal 2: `python -m pip install -r requirements.txt` then `npm run api`
3. Open the Vite URL, normally http://127.0.0.1:5173.

## Trained model
The notebook saves `student_tracking_model/attendance_model.joblib` and `student_tracking_model/feature_columns.json`. Neither file was present in the inspected project folder. Run the notebook's model-training/save cell, copy both files into `student_tracking_model/`, then restart the API. The API reports `model_available: false` rather than inventing a prediction when the model is missing.

## Gate data and limitations
`data/gate_events.json` is sample data, not a live reader. Replace it with an adapter for the actual RFID reader/export/API. RFID entry only confirms campus entry; classroom presence needs a trusted classroom attendance source. The notebook model was trained on synthetic data and includes camera-derived features; validate/retrain on representative, consented real data before operational use. Location shows last known zone only. Calling/SMS are not connected. Points are demo-only and require faculty confirmation.
