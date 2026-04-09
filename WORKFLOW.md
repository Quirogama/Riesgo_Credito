# Workflow del Proyecto: Riesgo Crediticio

## Escenario 1: Desarrollador Local (Setup completo)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PREPARAR AMBIENTE LOCAL                                  │
└─────────────────────────────────────────────────────────────┘
   $ python -m venv .venv
   $ .venv\Scripts\activate
   $ pip install -r requirements.txt
   
   ✓ Ambiente aislado listo

┌─────────────────────────────────────────────────────────────┐
│ 2. ENTRENAR MODELO (PRIMERA VEZ O REENTRENAMIENTO)          │
└─────────────────────────────────────────────────────────────┘
   $ python -m src.train
   
   ↓ ¿Qué pasa internamente?
   
   src/train.py:
   ├─ fetch_openml() → Descarga 30k registros desde OpenML
   ├─ train_test_split() → 80/20
   ├─ ColumnTransformer() → Preprocesamiento (OneHot + passthrough)
   ├─ XGBClassifier(500 árboles) → Entrena
   ├─ roc_auc_score() → Calcula AUC (~0.776)
   ├─ compute_global_shap() → Top 20 features SHAP
   └─ joblib.dump() → Guarda:
      ├─ models/xgb_credit_pipeline.joblib (modelo)
      └─ models/metrics.json (metadatos + AUC)
   
   ⏱️ Tiempo: ~2-3 min (primera vez descarga datos)
       ~30 seg (reentrenamiento)
   
   ✓ Output:
     Model saved to: models/xgb_credit_pipeline.joblib
     Metrics saved to: models/metrics.json
     AUC test: 0.7756

┌─────────────────────────────────────────────────────────────┐
│ 3. LEVANTAR API (DESARROLLO)                                │
└─────────────────────────────────────────────────────────────┘
   $ uvicorn app.main:app --reload
   
   ↓ Servidor inicia
   
   app/main.py:
   ├─ @app.on_event("startup")
   │  ├─ joblib.load() ← carga modelo
   │  ├─ json.load() ← carga métricas
   │  └─ shap.TreeExplainer() ← inicializa SHAP
   ├─ FastAPI uvicorn en 127.0.0.1:8000
   └─ Hot reload activado

   ✓ URLs disponibles:
     http://127.0.0.1:8000/docs (Swagger)
     http://127.0.0.1:8000/demo (UI visual)

┌─────────────────────────────────────────────────────────────┐
│ 4. PROBAR ENDPOINTS                                         │
└─────────────────────────────────────────────────────────────┘

   a) SWAGGER UI (interactivo)
      → http://127.0.0.1:8000/docs
      → Click en "/predict" → "Try it out"
      → Pega JSON del cliente
      → "Execute"
      → Ve probabilidad

   b) DEMO VISUAL (bonito)
      → http://127.0.0.1:8000/demo
      → "Cargar Ejemplo"
      → "Analizar Riesgo"
      → Ve KPIs + barras SHAP en vivo

   c) TERMINAL / POSTMAN
      $ $body = Get-Content .\examples\payload.json -Raw
      $ Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
          -Method Post -ContentType "application/json" -Body $body

      Response:
      {
        "probability_default": 0.1717,
        "predicted_label": 0,
        "risk_level": "low",
        "threshold": 0.5
      }

   d) EXPLAIN ENDPOINT
      $ Invoke-RestMethod -Uri "http://127.0.0.1:8000/explain?top_k=8" `
          -Method Post -ContentType "application/json" -Body $body

      Response:
      {
        "base_value": -1.2676,
        "top_contributors": [
          {"feature": "PAY_0", "shap_value": -0.4554, "direction": "decreases_risk"},
          {"feature": "PAY_AMT3", "shap_value": 0.1953, "direction": "increases_risk"}
        ]
      }

```

---

## Escenario 2: Usuario Final (Scoring masivo de clientes)

```
┌─────────────────────────────────────────────────────────────┐
│ CLIENTE ENVÍA SCORE REQUEST                                 │
└─────────────────────────────────────────────────────────────┘

   POST /predict + JSON
   {
     "LIMIT_BAL": 200000,
     "SEX": 2,
     "EDUCATION": 2,
     "MARRIAGE": 1,
     "AGE": 29,
     "PAY_0": 0,  ← Historial pagos mes 0
     "PAY_2": 0,  ← Historial pagos mes 2
     ...
     "BILL_AMT1": 3913,
     "PAY_AMT1": 0
   }

         ↓ API recibe

┌─────────────────────────────────────────────────────────────┐
│ API: /predict endpoint                                      │
└─────────────────────────────────────────────────────────────┘

   1. Validar input (Pydantic LoanRequest)
      ✓ 23 campos presentes
      ✓ SEX ∈ [1,2] ✓ AGE ∈ [18,100] etc.

   2. Transformar DataFrame
      X_sample = pd.DataFrame([payload.dict()])

   3. Pasar por pipeline.predict_proba()
      ├─ Preprocesamiento
      │  ├─ OneHot(SEX, EDUCATION, MARRIAGE)
      │  └─ Passthrough(resto numéricos)
      ├─ XGBoost (500 árboles)
      └─ Retorna probabilidad clase 1

   4. Comparar con threshold (0.5)
      predicted_label = 1 if prob >= 0.5 else 0

   5. Retornar JSON
      {
        "probability_default": 0.1717,  ← 17.17% riesgo de default
        "predicted_label": 0,            ← No default (por threshold)
        "risk_level": "low",             ← Label amigable
        "threshold": 0.5
      }

         ↓ Usuario sabe: 17% de riesgo → APRUEBA el crédito

┌─────────────────────────────────────────────────────────────┐
│ (OPCIONAL) Usuario quiere saber POR QUÉ bajo riesgo         │
└─────────────────────────────────────────────────────────────┘

   POST /explain?top_k=8 + MISMO JSON

         ↓ API recibe

   1. Transformar input de igual forma

   2. Calcular SHAP TreeExplainer
      shap_values = explainer.shap_values(X_transformed)

   3. Encontrar top-8 contribuyentes
      indices = argsort(abs(shap_values))[::-1][:8]

   4. Para cada contribuyente:
      {
        "feature": "numeric__PAY_0",
        "shap_value": -0.4554,                ← Negativo = reduce riesgo
        "direction": "decreases_risk"         ← Label amigable
      }

   5. Retornar explicación
      {
        "base_value": -1.2676,  ← Predicción base del modelo
        "top_contributors": [...]
      }

         ↓ Usuario entiende:
               • PAY_0 (historial pagos) DISMINUYÓ riesgo 0.45 pts
               • PAY_AMT3 (pago reciente) AUMENTÓ riesgo 0.19 pts
               • Balance final: BAJO

```

---

## Escenario 3: Deployment en Render (CLOUD)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PUSH TO GITHUB                                           │
└─────────────────────────────────────────────────────────────┘

   $ git add .
   $ git commit -m "Ready for production"
   $ git push origin main

   ✓ GitHub ahora tiene:
     ├─ src/train.py
     ├─ app/main.py
     ├─ requirements.txt
     ├─ render.yaml ← ¡Esto es la magia!
     └─ Dockerfile

┌─────────────────────────────────────────────────────────────┐
│ 2. RENDER DETECTA CAMBIOS (webhook GitHub)                  │
└─────────────────────────────────────────────────────────────┘

   Render leyó render.yaml, dispara build:

   a) COMPILE STAGE
      ○ Docker: python:3.12-slim
      ○ pip install -r requirements.txt
      ○ python -m src.train ← ENTRENA modelo en Render
         (Lee dataset OpenML gratis)
      ○ Guarda modelo en /app/models/
      
      ⏱️ ~3 min (primera vez, después cache)

   b) RUNTIME
      ○ uvicorn app.main:app --host 0.0.0.0 --port $PORT
      ○ Carga modelo desde JobLib
      ○ API lista en https://credit-risk-api-[id].onrender.com

   ✓ URL VIVA:
     https://credit-risk-api-[id].onrender.com/docs

┌─────────────────────────────────────────────────────────────┐
│ 3. USUARIO ACCEDE DESDE PRODUCCIÓN                          │
└─────────────────────────────────────────────────────────────┘

   Usuario desde Sezzle:
   $ curl https://credit-risk-api-[id].onrender.com/predict \
       -X POST \
       -H "Content-Type: application/json" \
       -d @cliente.json

   ↓ Request viaja a Render en la nube

   ├─ Load balancer Render
   ├─ Container con FastAPI
   ├─ Carga modelo desde memoria
   ├─ XGBoost + SHAP calculan
   └─ Response JSON (50-100ms)

   ✓ Respuesta:
     {
       "probability_default": 0.1717,
       "predicted_label": 0,
       "risk_level": "low",
       "threshold": 0.5
     }

┌─────────────────────────────────────────────────────────────┐
│ 4. REENTRENAR EN PRODUCCIÓN                                 │
└─────────────────────────────────────────────────────────────┘

   Si quieres actualizar modelo:
   $ git push origin main
   └─ Render redeploy automático
      └─ Re-entrena modelo con datos nuevos
         └─ Tiempo de inactividad: ~3 min

   O trigger manual en Render dashboard

```

---

## Diagrama completo: End-to-End

```
                         DESARROLLO LOCAL
                         ─────────────────

              ┌──────────────────────────────┐
              │ git clone + venv + pip       │
              └──────────────┬───────────────┘
                             │
                    ┌────────▼─────────┐
                    │ python -m        │
                    │ src.train        │
                    │ (5 min)          │
                    └────────┬─────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼────┐         ┌─────▼──────┐      ┌─────▼──────┐
   │Notebook │         │Model + Metr│      │API Ready   │
   │Analysis │         │ics trained │      │            │
   │.ipynb   │         │ .joblib    │      │            │
   └─────────┘         └─────┬──────┘      └─────┬──────┘
                             │                    │
                             │         ┌──────────▼────────┐
                             │         │ uvicorn dev       │
                             │         │ :8000/docs        │
                             │         │ :8000/demo        │
                             │         └──────┬────────────┘
                             │                │
                             │         ┌──────▼─────────┐
                             │         │ Test endpoints │
                             │         │ PowerShell     │
                             │         └──────┬─────────┘
                             │                │
        ┌────────────────────┴────────────────┘
        │
        ▼
    ┌─────────────────────────────────┐
    │ git push origin main            │  GITHUB
    │ ├─ Dockerfile                   │
    │ ├─ render.yaml                  │
    │ ├─ requirements.txt             │
    │ └─ src/ app/                    │
    └────────────┬────────────────────┘
                 │
                 │ (webhook)
                 ▼
    ┌────────────────────────────────┐
    │ Render detects render.yaml     │  RENDER DEPLOY
    ├─ Builds Docker image           │
    ├─ python -m src.train (compile) │
    ├─ uvicorn startup               │
    └────────────┬────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │ https://credit-risk-api-xyz    │  PRODUCCIÓN
    │ /predict (POST)                │
    │ /explain (POST)                │
    │ /demo (UI)                     │
    │ /docs (Swagger)                │
    └────────────────────────────────┘
                 │
                 │ (requests desde Sezzle)
                 ▼
    ┌────────────────────────────────┐
    │ 23 variables cliente JSON      │
    │ ↓ XGBoost.predict_proba        │
    │ ↓ SHAP.shap_values             │
    │ ↓ Risk Score + Explicación     │
    └────────────────────────────────┘

```

---

## Flujos de Datos Internos

### Flujo: /predict endpoint

```
JSON Input
   │
   ├─ Pydantic validation (LoanRequest)
   │
   ├─ DataFrame([payload])
   │
   ├─ Pipeline.predict_proba()
   │  ├─ ColumnTransformer
   │  │  ├─ OneHotEncoder(SEX, EDUCATION, MARRIAGE)
   │  │  └─ Passthrough(LIMIT_BAL, AGE, PAY_*, BILL_*, ...)
   │  │
   │  └─ XGBClassifier([500 árboles])
   │     └─ predict_proba() → [prob_class_0, prob_class_1]
   │
   ├─ np.array [0, 1] → take [1] → 0-1 scalar
   │
   ├─ Compare threshold (0.5)
   │  └─ label = 1 if prob >= 0.5 else 0
   │
   └─ JSON Response
      ├─ probability_default: 0.1717
      ├─ predicted_label: 0
      ├─ risk_level: "low"
      └─ threshold: 0.5
```

### Flujo: /explain endpoint

```
JSON Input
   │
   ├─ Pipeline.predict_proba() [igual que /predict]
   │
   ├─ Pipeline.named_steps["preprocessor"].transform()
   │  └─ X_transformed (después de OneHot + scaling)
   │
   ├─ SHAP TreeExplainer.shap_values(X_transformed)
   │  └─ Calcula impacto de cada feature
   │     (negativo = disminuye default, positivo = aumenta)
   │
   ├─ np.argsort(abs(shap_values))[::-1][:top_k]
   │  └─ Ordena por importancia (abs value)
   │
   └─ JSON Response
      ├─ base_value: -1.2676 (predicción sin features)
      └─ top_contributors: [
         ├─ feature: "numeric__PAY_0"
         ├─ shap_value: -0.4554
         ├─ direction: "decreases_risk"
         └─ ...
      ]
```

---

## Ejemplo Real Paso a Paso

**Cliente:** Juan García, solicitante de crédito de $50,000

```
ENTRADA (23 campos):
├─ LIMIT_BAL: 50000 (solicita ese límite)
├─ SEX: 1 (hombre)
├─ EDUCATION: 2 (escuela superior)
├─ MARRIAGE: 1 (casado)
├─ AGE: 35
├─ PAY_0: -1 (pagó a tiempo hace 1 mes)
├─ PAY_2: -1 (pagó a tiempo hace 3 meses)
├─ PAY_3: 0 (pagó exacto hace 4 meses)
├─ PAY_4: -1 (pagó a tiempo hace 5 meses)
├─ PAY_5: -1
├─ PAY_6: -1
├─ BILL_AMT1: 4500 (deuda reciente)
├─ BILL_AMT2: 4100
├─ BILL_AMT3: 3800
├─ BILL_AMT4: 3600
├─ BILL_AMT5: 3400
├─ BILL_AMT6: 3200
├─ PAY_AMT1: 3000 (pagó reciente)
├─ PAY_AMT2: 3000
├─ PAY_AMT3: 2800
├─ PAY_AMT4: 2600
├─ PAY_AMT5: 2400
└─ PAY_AMT6: 2200

PROCESAMIENTO:
├─ OneHotEncoder: SEX, EDUCATION, MARRIAGE → one-hot vectors
├─ XGBoost 500 árboles: "Este tipo paga bien..."
└─ Probabilidad: 0.082 (8.2% default risk)

SALIDA:
├─ probability_default: 0.082 ✓
├─ predicted_label: 0 (no default)
├─ risk_level: "low" ✓
└─ threshold: 0.5 ✓

EXPLICACIÓN SHAP:
├─ PAY_0 (-0.45): Excelente historial pagos
├─ LIMIT_BAL (-0.14): Límite solicitado es razonable
├─ BILL_AMT1 (-0.12): Deuda reciente manejable
├─ PAY_AMT1 (-0.10): Pagos al día reduce riesgo
├─ PAY_2 (-0.09): Historial consistente
├─ AGE (+0.05): Edad 35 es normal (datos)
├─ MARRIAGE: Sin impacto relevante
└─ EDUCATION: Sin impacto relevante

DECISIÓN:
✓ APROBAR crédito
├─ Riesgo bajo (8.2%)
├─ Cliente pagador
├─ Comportamiento predecible
└─ Razones: Historia de pagos a tiempo
```

---

## Resumiendo el workflow:

| Fase | Usuario | Herramientas | Tiempo | Output |
|------|---------|--------------|--------|--------|
| 1. Setup | Dev | Python + venv | 2 min | Env activado |
| 2. Entrenar | Dev | src.train | 2-3 min | Modelo + métricas |
| 3. Develop | Dev | uvicorn + Swagger | ∞ | Endpoints testados |
| 4. Test | Dev/QA | /predict + /explain | 50ms/call | JSON probas |
| 5. Deploy | DevOps | git push + Render | 3-5 min | URL en vivo |
| 6. Score | Usuario final | POST /predict | 50ms | Risk score + explicación |

**Es un ciclo:** Código → Entrena → API → Recibe requests → Devuelve scores + SHAP → ¡Listo!
