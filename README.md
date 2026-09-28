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

## Puesta en producción / entrega a un hospital

El código está listo para producción (autenticación, automatización sin
clics manuales, contenedores, validación de firma de Twilio). Lo que **no**
se puede resolver desde el código, porque depende de la identidad y los
pagos del hospital, son estos pasos manuales:

1. **Cuenta Twilio real** — crear una cuenta en [twilio.com](https://www.twilio.com),
   comprar un número con capacidad de voz/SMS, y copiar `Account SID` /
   `Auth Token` al archivo `.env` (basado en `.env.example`).
2. **WhatsApp Business aprobado por Meta** — el sandbox de Twilio sirve solo
   para pruebas; para enviar recordatorios reales a cualquier paciente, Meta
   exige verificar el negocio y aprobar una plantilla de mensaje. Esto aplica
   sin importar el proveedor (Twilio, la API oficial de Meta, o cualquier
   revendedor como NeuroChat u otros BSP de WhatsApp): la aprobación es de
   Meta, no del proveedor. El sistema soporta dos rutas, elegibles con
   `WHATSAPP_PROVIDER` en `.env`:
   - `twilio` (por defecto): el SID de la plantilla va en `TWILIO_WHATSAPP_CONTENT_SID`.
   - `meta_cloud`: usa la API oficial de Meta directamente (sin margen de
     intermediario por mensaje), con `META_WHATSAPP_TOKEN`,
     `META_WHATSAPP_PHONE_NUMBER_ID` y `META_WHATSAPP_TEMPLATE_NAME`.
   Si el hospital ya tiene cuenta con otro proveedor (por ejemplo NeuroChat)
   y su documentación técnica de API, se puede agregar como una tercera rama
   en `src/notification_service.py` siguiendo el mismo patrón — pero requiere
   esa documentación real; no se integra a ciegas un servicio sin API pública
   verificable.
3. **Hosting con HTTPS público** — Twilio necesita poder llamar de vuelta al
   webhook `/api/v1/webhook/twilio-dtmf` durante una llamada real, lo que
   requiere una URL pública con TLS (no `localhost`). Despliegue
   `docker-compose.yml` en un servidor/VM del hospital o en un proveedor
   cloud, y ponga esa URL en `PUBLIC_BASE_URL`.
4. **Secretos de producción** — genere `API_KEY` (protege `app/api.py`) y
   `DASHBOARD_PASSWORD` (protege `app/main_dashboard.py`) con valores
   aleatorios reales; sin ellos, el sistema queda abierto y lo advierte en
   `/` y en la barra lateral del dashboard.
5. **Autorización institucional y legal** — acceso al sistema HIS del
   hospital para reemplazar el dataset sintético por pacientes reales, y
   cumplimiento de la normativa peruana de protección de datos personales
   (los datos de salud son datos sensibles).

### Ejecutar con Docker (recomendado para entrega)

```bash
cp .env.example .env      # completar credenciales reales
docker compose up --build
```

Esto levanta tres servicios: `api` (puerto 8000), `dashboard` (puerto 8501)
y `scheduler` (proceso en segundo plano, sin puerto), compartiendo la misma
base de datos vía volumen. El `scheduler` es lo que hace que el sistema
contacte pacientes y reasigne cupos **solo**, sin que un operador presione
botones — es la pieza que falta para llamarlo "automatizado" de verdad.
