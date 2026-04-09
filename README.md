# Riesgo Crediticio: Scoring IA con XGBoost + SHAP

Modelo de machine learning para estimar la probabilidad de impago de prestamos. Predice con **AUC 77.6%** y explica cada prediccion usando SHAP (SHapley Additive exPlanations) para interpretabilidad real.

**Stack:** XGBoost + SHAP + FastAPI. **Costo:** $0 para entrenar, usar y deployar.

---

## El Problema

Predecir si un cliente va a entrar en default es crítico para:
- Reducir pérdidas por incobrabilidad
- Tomar decisiones de crédito rápido y justo
- Entender **por qué** rechazamos/aprobamos un cliente (regulación + confianza)

El modelo traduce esto en:
1. **Probabilidad de impago** (0-100%)
2. **Top factores** que influyeron (interpretabilidad SHAP)
3. **API REST lista para producción**

---

## Métrica

| Métrica | Valor |
|---------|-------|
| AUC (test) | 0.776 |
| Dataset | 30,000 clientes (CCF Taiwan) |
| Features | 23 variables |
| Umbral default | 50% |

---

## Arquitectura

```
┌─────────────────────────────────────────────┐
│ Cliente (JSON con atributos)                │
└──────────────┬──────────────────────────────┘
               │
         ┌─────▼────────┐
         │  API FastAPI │
         │   /predict   │
         │   /explain   │
         └─────┬────────┘
               │
    ┌──────────┴──────────┐
    │                     │
    ▼                     ▼
┌─────────────┐  ┌──────────────┐
│ XGBoost     │  │ SHAP         │
│ Pipeline    │  │ TreeExplainer│
│ 500 árboles │  │ Top-8 vars   │
└─────────────┘  └──────────────┘
    │                     │
    └──────────┬──────────┘
               │
         ┌─────▼──────────────┐
         │ Response JSON      │
         │ - probability      │
         │ - risk_level       │
         │ - top_contributors │
         └────────────────────┘
```

---

## Interpretabilidad SHAP

SHAP (SHapley Additive exPlanations) es un método basado en game theory que responde:
"¿Cuánto cambió cada variable la predicción del modelo?"

Ejemplo de output:
```json
{
  "base_value": -1.27,
  "top_contributors": [
    {
      "feature": "PAY_0",
      "shap_value": -0.456,
      "direction": "decreases_risk"
    },
    {
      "feature": "PAY_AMT3",
      "shap_value": 0.195,
      "direction": "increases_risk"
    }
  ]
}
```

**Interpretación:** PAY_0 (antecedentes de pago) bajó mucho el riesgo. PAY_AMT3 (pago reciente) lo subió poco.

---

## Instalación Local

### 1. Clonar y entrar

```bash
git clone https://github.com/tu_usuario/Riesgo_Credito.git
cd Riesgo_Credito
```

**Primero:** Lee [WORKFLOW.md](WORKFLOW.md) para entender el flujo completo.

### 2. Virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate  # En Windows
# source .venv/bin/activate  # En macOS/Linux
```

### 3. Dependencias

```bash
pip install -r requirements.txt
```

### 4. Entrenar modelo

```bash
python -m src.train
```

**Salida esperada:**
```
Model saved to: models/xgb_credit_pipeline.joblib
Metrics saved to: models/metrics.json
AUC test: 0.7756
```

---

## Uso de la API

### Levantar servidor

```bash
uvicorn app.main:app --reload
```

Swagger UI (documentación interactiva):
- http://127.0.0.1:8000/docs

### Demo visual

```
http://127.0.0.1:8000/demo
```

Interfaz de usuario para scoring + visualización SHAP en tiempo real.

---

## Endpoints

### `GET /` — Health check

```bash
curl http://127.0.0.1:8000/
```

**Response:**
```json
{
  "status": "ok",
  "message": "Credit risk API running",
  "auc_test": 0.776,
  "demo_url": "/demo",
  "docs_url": "/docs"
}
```

### `GET /model-info` — Metadatos del modelo

```bash
curl http://127.0.0.1:8000/model-info
```

**Response:**
```json
{
  "dataset": "default-of-credit-card-clients",
  "auc": 0.776,
  "features": ["LIMIT_BAL", "SEX", "EDUCATION", ...],
  "top_shap_features": [...]
}
```

### `POST /predict` — Predicción

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/payload.json
```

**Request:** (23 variables financieras)
```json
{
  "LIMIT_BAL": 200000,
  "SEX": 2,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 29,
  "PAY_0": 0,
  "BILL_AMT1": 3913,
  "PAY_AMT1": 0,
  ...
}
```

**Response:**
```json
{
  "probability_default": 0.1717,
  "predicted_label": 0,
  "risk_level": "low",
  "threshold": 0.5
}
```

### `POST /explain?top_k=8` — Explicación SHAP

```bash
curl -X POST "http://127.0.0.1:8000/explain?top_k=8" \
  -H "Content-Type: application/json" \
  -d @examples/payload.json
```

**Response:**
```json
{
  "base_value": -1.2676,
  "top_contributors": [
    {
      "feature": "numeric__PAY_0",
      "shap_value": -0.4554,
      "direction": "decreases_risk"
    },
    {
      "feature": "numeric__PAY_AMT3",
      "shap_value": 0.1953,
      "direction": "increases_risk"
    }
  ]
}
```

---

## Jugar con el modelo

### Desde PowerShell

```powershell
# Predicción
$body = Get-Content .\examples\payload.json -Raw
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post `
  -ContentType "application/json" -Body $body

# Explicación
Invoke-RestMethod -Uri "http://127.0.0.1:8000/explain?top_k=8" -Method Post `
  -ContentType "application/json" -Body $body
```

### Análisis exploratorio

Abre el Notebook:

```bash
jupyter notebook notebooks/analysis.ipynb
```

Incluye:
- Exploración de datos
- Entrenamiento + validación
- Visualización SHAP
- Feature importance

---

## Deploy Gratis

### Opción 1: Render (recomendado)

1. Fork este repo en GitHub
2. Ve a https://render.com
3. Conecta tu repo GitHub
4. Crea nuevo "Web Service"
5. Selecciona `render.yaml` como blueprint
6. Deploy automático (entrena modelo + levanta API)

**URL en vivo:** `https://credit-risk-api-[tu-id].onrender.com`

### Opción 2: Railway

Similar a Render:
1. https://railway.app
2. "New Project" → Connect GitHub
3. Selecciona repo
4. Auto-deploy

---

## Estructura del proyecto

```
Riesgo_Credito/
├── app/
│   ├── main.py              # API endpoints
│   ├── static/
│   │   └── demo.html        # UI visual
│   └── __init__.py
├── notebooks/
│   └── analysis.ipynb       # EDA + training analysis
├── examples/
│   └── payload.json         # Request de ejemplo
├── models/                  # Artefactos (generados al entrenar)
│   ├── xgb_credit_pipeline.joblib
│   └── metrics.json
├── src/
│   ├── train.py             # Pipeline de entrenamiento
│   └── __init__.py
├── Dockerfile               # Para deploy en contenedores
├── render.yaml              # Config para Render deploy
├── requirements.txt         # Dependencias
├── .gitignore
├── README.md
└── LICENSE
```

---

## Dataset

**Fuente:** [UCI ML - Default of Credit Card Clients](https://www.openml.org/d/1461)

- **30,000 clientes** de Taiwan
- **23 features:** límite de crédito, sexo, educación, matrimonio, edad, historial de pagos (6 meses), montos facturados (6 meses), montos pagados (6 meses)
- **Target:** default en siguiente mes (sí/no)
- **Balance:** ~22% positivos (default)

Automáticamente descargado desde OpenML al entrenar.

---

## Tecnologías

| Componente | Tech |
|-----------|------|
| ML Model | XGBoost 2.1+ |
| Interpretabilidad | SHAP 0.46+ |
| API | FastAPI 0.115+ |
| Server | Uvicorn 0.30+ |
| Data | Pandas, Scikit-learn |

**Ventaja:** Todas open source. Sin costos de licencia.

---

## Notas de Producción

- **Modelo entrenado en CPU** (sin GPU necesaria)
- **AUC ~77.6%** varía ligeramente con versiones de librerías
- **Threshold 50%** ajustable según tolerancia de riesgo
- **SHAP se calcula en tiempo real** (~100ms por predicción en CPU)

Para optimizar SHAP en producción:
- Pre-calcular explicaciones top global (modelo-wide)
- Usar background dataset reducido (default: 100 muestras)

---

## Licencia

MIT – Libre para usar, modificar y distribuir.

---

## Contacto

Preguntas? Issues en [GitHub Discussions](#).

---

**Built with ❤️ for interpretable AI. No cloud fees. No bullshit.**
