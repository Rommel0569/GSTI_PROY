# Sistema Predictivo de No-Show Médico

MVP funcional del proyecto de investigación *"Sistema predictivo de inasistencia
a citas médicas con recomendación automatizada de intervención en consulta
externa"* (Escuela Profesional de Ingeniería de Sistemas, UNSA).

Predice la probabilidad de inasistencia de un paciente a una cita, la explica
con SHAP, recomienda la intervención de menor costo suficiente según el canal
de contacto disponible, y gestiona la liberación/reasignación automática de
cupos a una lista de espera.

## Stack tecnológico (100% gratuito / open-source)

| Componente | Tecnología | Costo |
|---|---|---|
| Modelo predictivo | scikit-learn, XGBoost | Gratis (open-source) |
| Explicabilidad | SHAP (TreeExplainer) | Gratis (open-source) |
| Persistencia | SQLite (librería estándar de Python) | Gratis, sin servidor |
| Backend | FastAPI + Uvicorn | Gratis (open-source) |
| Frontend | Streamlit + Plotly | Gratis (open-source) |
| Ingesta de datos | Kaggle API / descarga HTTP directa | Gratis |
| Notificaciones | Twilio, en **modo simulación offline por defecto** (`MOCK_NOTIFICATIONS=true`) | S/ 0 en el MVP académico |

No se requiere ninguna tarjeta de crédito, cuenta paga ni infraestructura en la
nube para ejecutar el proyecto completo. Twilio (único componente con costo
real en producción) está desactivado por defecto: todas las llamadas, SMS y
WhatsApp se simulan en consola y en la interfaz.

## Estructura del proyecto

```
gsti_proy/
├── data/{raw,processed}/     # Dataset y SQLite (hospital.db)
├── models/                   # Modelo, preprocesador y explicador serializados
├── src/                      # Pipeline, entrenamiento, motor de reglas, BD
├── app/                      # API FastAPI + Dashboard Streamlit
├── tests/                    # Pruebas unitarias (pytest)
└── requirements.txt
```

## Instalación y ejecución (one-command setup)

```bash
# 0. (Recomendado) crear entorno virtual
python -m venv venv
venv\Scripts\activate           # Windows (cmd.exe / PowerShell)
# source venv/Scripts/activate  # Windows (Git Bash / MINGW64)
# source venv/bin/activate      # Linux/Mac

# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Descargar datos, limpiar, generar variables y poblar SQLite
python src/data_pipeline.py

# 3. Entrenar y exportar modelos con SHAP
python src/train.py

# 4. Iniciar el dashboard (funciona de forma autónoma, sin necesidad del API)
streamlit run app/main_dashboard.py

# (Opcional) Levantar también el backend REST con Swagger UI en /docs
uvicorn app.api:app --reload
```

## Estrategia de ingesta de datos (3 niveles)

`src/data_pipeline.py` implementa `load_dataset()` con failover automático:

1. **Nivel 1 — Archivo local:** busca `data/raw/KaggleV2-May-2016.csv` o
   `data/raw/noshowappointments.csv`.
2. **Nivel 2 — Kaggle API:** si hay `KAGGLE_USERNAME`/`KAGGLE_KEY` (env vars) o
   `~/.kaggle/kaggle.json`, descarga el dataset oficial `joniarroba/noshowappointments`.
3. **Nivel 3 — Espejo HTTP:** si no hay credenciales, descarga el mismo CSV
   (110,527 filas) desde un mirror público en GitHub, con barra de progreso.

Si los tres niveles fallan (por ejemplo, sin conexión a internet), el script
indica cómo colocar el archivo manualmente.

## Simplificaciones documentadas (alcance académico de un semestre)

- El dataset de Kaggle no incluye el canal de contacto real del paciente ni
  la especialidad de la cita: ambas se generan de forma sintética y
  configurable (`contact_channel_available`: 70% WhatsApp / 20% SMS / 10% sin
  celular; `specialty`: 5 especialidades de ejemplo), tal como se declara en
  la Sección 3.5 del informe de avance.
- Para que la "Consola de Gestión" tenga citas dentro de la ventana de 48-24h
  sin importar cuándo se ejecute el proyecto, `data_pipeline.py` genera además
  ~60 citas "demo" con fechas relativas a la fecha de ejecución, muestreando
  perfiles reales del histórico.
- El costo de la visita domiciliaria (`src/decision_engine.py`) es una
  estimación propia (personal + transporte); el informe declara explícitamente
  que el costeo real de llamada y visita para un hospital peruano es una tarea
  de la siguiente fase.

## Pruebas

```bash
pytest -v
```

Las pruebas no requieren el dataset descargado ni el modelo entrenado: usan
DataFrames sintéticos y una base SQLite en memoria.

## Notas sobre Twilio (Componente E)

Por defecto `MOCK_NOTIFICATIONS=true`: todo SMS/WhatsApp/llamada se imprime en
consola y se registra en la auditoría sin costo. Para usar Twilio real,
configure las variables de entorno `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`,
`TWILIO_FROM_NUMBER` y fije `MOCK_NOTIFICATIONS=false`.
