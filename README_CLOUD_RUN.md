# MLB-Pred V7 en Google Cloud Run

La interfaz web nueva reemplaza la dependencia de Streamlit para producción, pero conserva los motores V7 (`PredictorMLMLB`, Monte Carlo, scanner, Kelly y Big Data) como fuente única de verdad.

## Arquitectura

`static/index.html -> FastAPI web_app.py -> modules/web_service.py -> ML + Monte Carlo + Big Data + scanner`

Streamlit (`app_mlb.py`) se conserva temporalmente como fallback y para comparar resultados durante la migración.

## Endpoints

- `GET /` dashboard web móvil.
- `GET /api/health` estado del modelo, Big Data y configuración.
- `GET /api/slate` cartelera oficial + cuotas cuando existe `ODDS_API_KEY`.
- `POST /api/scan` scanner completo V7 y persistencia del top 3.
- `POST /api/reload` recarga CSV/modelo sin reiniciar la instancia.

## Cloud Run

Cloud Run inyecta `PORT`; el contenedor escucha siempre `${PORT}`. El Dockerfile usa un solo worker porque el modelo ML y DuckDB son objetos pesados y no conviene duplicarlos por proceso.

Configuración propuesta: 2 CPU, 2 GiB RAM, concurrency 4, timeout 900 s, max 3 instancias. El scanner ejecuta 50,000 simulaciones por partido, igual que producción Streamlit.

## Primera vinculación segura de GitHub con Google Cloud

El workflow `.github/workflows/deploy-cloud-run.yml` usa Workload Identity Federation (OIDC), no una llave JSON estática. Configura en GitHub:

- Secret `GCP_PROJECT_ID`: ID de tu proyecto.
- Secret `GCP_WIF_PROVIDER`: recurso completo del provider de Workload Identity.
- Secret `GCP_WIF_SERVICE_ACCOUNT`: service account autorizada a desplegar Cloud Run.
- Variable opcional `GCP_REGION`; por defecto `us-central1`.

La service account de despliegue necesita permisos para Cloud Run y source builds de Cloud Build/Artifact Registry según la configuración del proyecto.

## Secretos de ejecución

No guardes claves en el repositorio. Después del primer deploy agrega en Cloud Run, preferentemente desde Secret Manager:

- `ODDS_API_KEY` para mercado real. Sin ella la cartelera funciona, pero el scanner no fabrica apuestas.
- Las variables del backend persistente del ledger si deseas conservar picks entre revisiones (`GITHUB_TOKEN`, `LEDGER_GITHUB_REPO`) o la configuración de Google Sheets ya usada por el proyecto.

Ejemplo con Secret Manager si ya creaste `ODDS_API_KEY`:

```bash
gcloud run services update mlb-pred-v7 \
  --region us-central1 \
  --set-secrets ODDS_API_KEY=ODDS_API_KEY:latest
```

## Pruebas antes de desplegar

```bash
pip install -r requirements.txt
python test_cloudrun_web.py
docker build -t mlb-pred-cloudrun .
docker run --rm -p 8080:8080 -e PORT=8080 mlb-pred-cloudrun
```

Abre `http://localhost:8080` y prueba `http://localhost:8080/api/health`.

## Despliegue

En GitHub Actions ejecuta manualmente **Deploy MLB to Cloud Run**. Al final imprime la URL pública del servicio.

## Regla de seguridad predictiva

La migración no cambia umbrales ni fórmulas de selección. `modules/web_service.py` llama a los mismos motores V7 de ML, Monte Carlo y `scanner_engine`; las señales avanzadas conservan sus gates de cobertura/validación. Si no hay cuotas completas de dos vías, el resultado es `NO BET`.
