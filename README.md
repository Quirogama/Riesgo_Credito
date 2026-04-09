# Riesgo Crediticio: Scoring IA con XGBoost + SHAP

Modelo de machine learning para estimar la probabilidad de impago de prestamos. Predice con **AUC 77.6%** y explica cada prediccion usando SHAP (SHapley Additive exPlanations) para interpretabilidad real.

**Stack:** XGBoost + SHAP + FastAPI. **Costo:** $0 para entrenar, usar y desplegar.

**Demo en vivo:** https://riesgo-credito.onrender.com/demo

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

## Quickstart (2 minutos)

```bash
pip install -r requirements.txt
python -m src.train
uvicorn app.main:app --reload
```

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

SHAP (SHapley Additive exPlanations) es un método basado en teoría de juegos que responde:
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
git clone https://github.com/Quirogama/Riesgo_Credito.git
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

Rutas útiles en producción:
- Demo: https://riesgo-credito.onrender.com/demo
- Docs: https://riesgo-credito.onrender.com/docs
- Health JSON: https://riesgo-credito.onrender.com/api/health

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

---

## Licencia

MIT – Libre para usar, modificar y distribuir.

---