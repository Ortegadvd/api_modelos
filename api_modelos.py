import subprocess

# Ejecuta el script para descargar el modelo si no existe
subprocess.run(["python", "download_model.py"])

from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Carga los modelos y el vectorizador
vectorizer = joblib.load("vectorizer.pkl")
clf_concepto = joblib.load("modelo_concepto.pkl")
clf_subconcepto = joblib.load("modelo_subconcepto.pkl")
clf_descripcion = joblib.load("modelo_descripcion.pkl")

app = FastAPI()

# Define el esquema de entrada
class Movimiento(BaseModel):
    FECHA: str
    INGRESOS: float
    EGRESOS: float
    CONCEPTO_BANCO: str
    CUENTA: str
    TIPO_CUENTA: str

def make_input_concepto(mov):
    return f"FECHA: {mov.FECHA} | INGRESOS: {mov.INGRESOS} | EGRESOS: {mov.EGRESOS} | CONCEPTO BANCO: {mov.CONCEPTO_BANCO} | CUENTA: {mov.CUENTA} | TIPO CUENTA: {mov.TIPO_CUENTA}"

def make_input_subconcepto(mov, concepto):
    return f"FECHA: {mov.FECHA} | INGRESOS: {mov.INGRESOS} | EGRESOS: {mov.EGRESOS} | CONCEPTO BANCO: {mov.CONCEPTO_BANCO} | CUENTA: {mov.CUENTA} | TIPO CUENTA: {mov.TIPO_CUENTA} | CONCEPTO: {concepto}"

def make_input_descripcion(mov, concepto, subconcepto):
    return f"FECHA: {mov.FECHA} | INGRESOS: {mov.INGRESOS} | EGRESOS: {mov.EGRESOS} | CONCEPTO BANCO: {mov.CONCEPTO_BANCO} | CUENTA: {mov.CUENTA} | TIPO CUENTA: {mov.TIPO_CUENTA} | CONCEPTO: {concepto} | SUBCONCEPTO: {subconcepto}"

@app.post("/predict")
def predict(mov: Movimiento):
    # 1. Predice CONCEPTO
    input_concepto = make_input_concepto(mov)
    X_concepto = vectorizer.transform([input_concepto])
    concepto_pred = clf_concepto.predict(X_concepto)[0]

    # 2. Predice SUBCONCEPTO usando el CONCEPTO predicho
    input_subconcepto = make_input_subconcepto(mov, concepto_pred)
    X_subconcepto = vectorizer.transform([input_subconcepto])
    subconcepto_pred = clf_subconcepto.predict(X_subconcepto)[0]

    # 3. Predice DESCRIPCION usando CONCEPTO y SUBCONCEPTO predichos
    input_descripcion = make_input_descripcion(mov, concepto_pred, subconcepto_pred)
    X_descripcion = vectorizer.transform([input_descripcion])
    descripcion_pred = clf_descripcion.predict(X_descripcion)[0]

    return {
        "CONCEPTO": concepto_pred,
        "SUBCONCEPTO": subconcepto_pred,
        "DESCRIPCION": descripcion_pred
    }