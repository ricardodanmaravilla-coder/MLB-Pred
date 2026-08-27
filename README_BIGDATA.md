# MLB-Pred Big Data

La capa Big Data es aditiva y mantiene compatibilidad con el pipeline CSV existente.

## Flujo de producción

1. `PredictorMLMLB` intenta abrir `data/bigdata/mlb.duckdb`.
2. Si el warehouse no existe, se construye automáticamente desde `data/mlb_games.csv`.
3. Las features pregame se generan cronológicamente y se exportan a Parquet.
4. El modelo entrena con el contrato histórico del predictor actual desde DuckDB/Parquet.
5. Si DuckDB/Parquet falla, el modelo vuelve al constructor cronológico CSV existente.
6. El scanner guarda sus recomendaciones en el ledger CSV/GitHub/Google Sheets y, además, en DuckDB.
7. `settle_picks.py` liquida WIN/LOSS/PUSH y replica el resultado en DuckDB.

## Garantía temporal

Para un partido de fecha D, las features de forma y H2H se construyen exclusivamente con partidos completados antes de D. Las métricas estacionales que usa el contrato ML heredado se unen desde `Season - 1`. Los targets se almacenan para entrenamiento, pero nunca participan en la construcción de features anteriores.

## Componentes

- `modules/bigdata_mlb.py`: warehouse, ingesta, feature store, training frames y métricas de tracking.
- `modules/bigdata_tracking.py`: puente idempotente entre el ledger del scanner y DuckDB.
- `build_bigdata.py`: construcción/actualización manual.
- `test_bigdata_mlb.py`: pruebas de fuga temporal.
- `test_bigdata_integration.py`: prueba end-to-end warehouse + ML + tracking + settlement.
- `.github/workflows/bigdata-ci.yml`: validación automática.

## Artefactos locales

`data/bigdata/*.duckdb` y `data/bigdata/*.parquet` no se versionan. Se reconstruyen desde las fuentes CSV canónicas para evitar binarios obsoletos en Git.

## Comandos

```bash
pip install -r requirements.txt
python build_bigdata.py
python test_bigdata_mlb.py
python test_bigdata_integration.py
```
