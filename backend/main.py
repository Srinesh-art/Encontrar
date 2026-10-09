from pathlib import Path
import json
from datetime import datetime
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
ROOT=Path(__file__).resolve().parent.parent
ART=ROOT/'student_tracking_model'; MODEL=ART/'attendance_model.joblib'; FEATURES=ART/'feature_columns.json'; GATES=ROOT/'data'/'gate_events.json'
app=FastAPI(title='Encontrar Attendance API',version='0.1.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://127.0.0.1:5173','http://localhost:5173'],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
model=None; feature_columns=[]; model_error=None
def load_model():
 global model,feature_columns,model_error
 if not MODEL.exists(): model_error='Trained model file is missing: '+str(MODEL); return
 try:
  model=joblib.load(MODEL)
  feature_columns=json.loads(FEATURES.read_text(encoding='utf-8')) if FEATURES.exists() else []
  model_error=None
 except Exception as exc: model=None;model_error=str(exc)
load_model()
class Check(BaseModel):
 student_id:str
 department:str='AIML'
 year_of_study:int=2
 attendance_status:str='Absent'
class Points(BaseModel):
 student_id:str
 points:int=1
 reason:str='Faculty-reviewed policy violation'
def gate_for(sid):
 try:
  rows=json.loads(GATES.read_text(encoding='utf-8'))
  matches=[r for r in rows if str(r.get('student_id'))==sid]
  if matches:return sorted(matches,key=lambda r:str(r.get('timestamp','')))[-1]
 except Exception:pass
 return {'student_id':sid,'gate_id':'Unknown','entry_verified':0,'campus_present':False,'class_present':False,'zone':'Unknown','last_seen':'Not available','source':'No gate event found'}
@app.get('/api/health')
def health(): return {'status':'ok','model_available':model is not None,'model_path':str(MODEL),'model_error':model_error,'gate_events_available':GATES.exists()}
@app.post('/api/attendance/check')
def check(body:Check):
 gate=gate_for(body.student_id)
 if model is None:
  return {'student_id':body.student_id,'status':'model_unavailable','model_available':False,'campus_present':bool(gate.get('campus_present',False)),'class_present':bool(gate.get('class_present',False)),'zone':gate.get('zone','Unknown'),'last_seen':gate.get('last_seen','Not available'),'gate_id':gate.get('gate_id','Unknown'),'source':gate.get('source','Gate event'),'reason':'No trained model loaded. Gate data only; no ML prediction was made.'}
 stamp=str(gate.get('timestamp',gate.get('entry_time','00:00:00')))
 try: hour=int(stamp.split('T')[-1].split(' ')[-1].split(':')[0])
 except Exception: hour=0
 elapsed=max(0,int(gate.get('time_since_last_observation_minutes',0) or 0)); count=max(0,int(gate.get('camera_observation_count',0) or 0))
 row={'year_of_study':body.year_of_study,'entry_verified':int(gate.get('entry_verified',0) or 0),'time_since_last_observation_minutes':elapsed,'camera_observation_count':count,'identity_verified':int(gate.get('identity_verified',0) or 0),'entry_hour':hour,'observation_rate':count/(elapsed+1.0)}
 for dept in gate.get('known_departments',['Computer Science','Data Science','Electrical Eng','Mechanical Eng','Business Admin','AIML','CSE','ECE','IT']):row['dept_'+dept]=int(dept==body.department)
 try:
  X=pd.DataFrame([{f:row.get(f,0) for f in feature_columns}]); pred=int(model.predict(X)[0]); prob=float(model.predict_proba(X)[0][1]) if hasattr(model,'predict_proba') else None
 except Exception as exc: raise HTTPException(status_code=422,detail='Model feature mismatch: '+str(exc))
 campus=bool(gate.get('campus_present',int(gate.get('entry_verified',0))==1)); classroom=bool(gate.get('class_present',False))
 return {'student_id':body.student_id,'status':'exception_flagged' if pred==1 else 'no_exception_flagged','model_available':True,'prediction':pred,'confidence':prob,'campus_present':campus,'class_present':classroom,'zone':gate.get('zone','Unknown'),'last_seen':gate.get('last_seen','Not available'),'gate_id':gate.get('gate_id','Unknown'),'source':gate.get('source','Gate event'),'reason':'ML flagged an attendance exception; faculty review required.' if pred==1 else 'ML did not flag an exception.'}
@app.post('/api/points')
def add_points(body:Points): return {'status':'recorded_demo_only','student_id':body.student_id,'points':max(1,min(10,body.points)),'reason':body.reason,'timestamp':datetime.now().isoformat()}
