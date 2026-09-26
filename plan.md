# ParkWatch — Plan del proyecto

**Objetivo:** pipeline MLOps end-to-end (captura → almacenamiento → etiquetado → entrenamiento → reentreno periódico → despliegue) para clasificar la ocupación de plazas de parking a partir de una cámara ESP32-CAM.

**Contexto:** segundo proyecto de portfolio MLOps, complementario a `tinyConvNeXt` (Azure ML + MLflow self-hosted). Cámara ESP32-CAM con sensor OV5640, colocada en interior, enfocando a través de una ventana hacia un parking exterior.

**Estado:** planificación — aún sin empezar la implementación.

---

## Decisiones tomadas

- **Nombre del proyecto:** ParkWatch
- **Hardware de captura:** ESP32-CAM (OV5640), requiere Arduino IDE para flashear firmware y probablemente adaptador USB-TTL
- **Ruta de subida:** ESP32 → Raspberry Pi (HTTP local, sin TLS) → Pi sube a Azure Blob Storage
  - Motivo: evitar fragilidad de TLS/certificados en el ESP32 (memoria limitada); la Pi permite logging, reintentos, resize/compress y metadata antes de subir
- **Cadencia de captura:** 1 foto cada 3 minutos (fase de portfolio/dataset, no producto en vivo)
- **Horario de captura:** el ESP32 no sabe nada de horarios — captura cada 3 min de forma continua (24h) y manda todo a la Pi. La Pi decide, foto a foto, si cae dentro de la ventana de luz usando `astral` (crepúsculo civil `dawn`/`dusk`, con margen de ~15 min antes/después, calculado día a día — se adapta a las estaciones sin tocar código). Si está fuera de ventana, se descarta y no se sube a Blob. La comprobación ocurre en el mismo servicio de la Pi que recibe la foto del ESP32, antes de subir a Blob.
- **Etiquetado:** manual por el usuario, con Label Studio, usando las predicciones del modelo como pre-anotación (revisión, no etiquetado desde cero)
- **Estrategia de inferencia/despliegue (v1):**
  - **NO** inferencia en directo desde el principio — no tiene sentido servir rápido un modelo de baja calidad
  - Proceso diario: **Batch Endpoint** procesa todas las fotos nuevas del día → predicciones
  - Revisión/corrección manual en Label Studio
  - Etiquetas corregidas alimentan el pool de reentreno
  - **Reentreno diario** (batch job, patrón ya validado en tinyConvNeXt Paso 7): reentreno completo desde cero, evaluación contra holdout fijo, promoción a `champion` solo si mejora
  - Batch Endpoint también reservado para evaluación de reentreno y reprocesado histórico
- **Inferencia en directo (v2, futuro):** solo cuando el modelo tenga accuracy suficiente — arquitectura event-driven (Azure Function con Blob trigger) + API de consulta ligera que lee un estado cacheado (Table Storage), no Managed Online Endpoint siempre activo
- **ROIs:** fijas por plaza (estilo PKLot/CNRPark), definidas una vez sobre una imagen de referencia tras fijar la posición de la cámara
- **Privacidad:** blurring de matrículas obligatorio antes de compartir el dataset públicamente

## Puntos pendientes de definir

- Prueba de campo con la cámara ya montada: nº de plazas en encuadre, ángulo/perspectiva, nitidez a través del cristal, reflejos/contraluz en distintas franjas horarias
- Coordenadas exactas a usar en `astral` (Zubiri vs. Pamplona — confirmar ubicación definitiva de la cámara)
- Cuántos días/fotos de dataset semilla antes del primer entrenamiento
- Formato exacto de las ROIs (coordenadas, config JSON) y cómo se cargan en Label Studio (clasificación por recorte de plaza vs. anotación completa)
- Estructura de contenedores/carpetas en Blob (raw/, labeled/, manifest)
- Compute a usar para el Batch Endpoint (cluster Azure ML vs. homelab)

---

## Fases

### Fase 0 — Puesta a punto física
- Fijar posición definitiva de la cámara con soporte casero, en la ventana ya elegida (no debe moverse una vez fijadas las ROIs)
- Prueba de campo: firmware mínimo de captura + fotos sueltas en distintos momentos del día para validar nº de plazas en encuadre, ángulo/perspectiva, nitidez y reflejos/contraluz
- Horario de captura: **decidido** — ESP32 sin lógica de horario (capta 24h cada 3 min), la Pi filtra con `astral` (crepúsculo civil `dawn`/`dusk` + margen ~15 min) antes de subir a Blob
- Capturar imagen de referencia y definir ROIs fijas por plaza (config JSON con coordenadas; rectángulo simple o polígono de 4 puntos según el ángulo que muestre la prueba de campo)
- Validar encuadre: cobertura de todas las plazas objetivo, sin obstrucciones (reflejos, ramas, etc.), en varias franjas horarias

### Fase 1 — Captura (ESP32 → Pi → Blob)
- Firmware ESP32-CAM: captura periódica (cada 3 min), POST a endpoint local de la Pi
- Servidor en la Pi (Flask/FastAPI): recibe la foto, logging, reintentos, resize/compress, sube a Azure Blob con metadata (timestamp, nombre consistente)
- Estructura en Blob: contenedor `raw/`

### Fase 2 — Etiquetado inicial + dataset
- Bootstrapping: etiquetado manual completo desde el día 0 en que hay fotos reales en Blob (después de Fase 0)
- Definición de train/holdout (holdout fijo, nunca se re-etiqueta, como en tinyConvNeXt)
- Herramienta: Label Studio, clasificación libre/ocupado por ROI/plaza

### Fase 3 — Entrenamiento inicial
- Modelo: clasificador binario por parche de ROI (no detección) — arquitectura ligera, posible reutilización de patrones de tinyConvNeXt adaptados a tarea binaria
- Tracking en el MLflow self-hosted ya desplegado (Azure Container App)

### Fase 4 — Pipeline diario de reentreno (recurrente, régimen permanente)
1. **Batch Endpoint**: inferencia sobre todas las fotos nuevas del día
2. **Label Studio**: revisión/corrección manual de las predicciones (active learning / human-in-the-loop)
3. Correcciones entran al pool de reentreno (ingesta incremental, patrón similar a `ingest_batch.py` de tinyConvNeXt)
4. **Reentreno diario**: entrenamiento completo desde cero sobre el pool acumulado
5. **Evaluación**: contra holdout fijo, F1 macro (u otra métrica) frente al `champion` actual
6. **Promoción**: alias `champion` se reasigna solo si el nuevo modelo supera al actual

### Fase 5 — Inferencia en directo (futuro, no v1)
- Solo activar cuando el modelo alcance accuracy suficiente en el proceso de Fase 4
- Arquitectura event-driven: Azure Function con Blob trigger ejecuta inferencia por foto nueva, escribe último estado en Table Storage
- API de consulta pública ligera y siempre disponible (Function HTTP, consumption plan) que solo lee el estado cacheado — no ejecuta el modelo por petición
- Alternativa/comparativa de portfolio: Managed Online Endpoint con scale-to-zero, si se quiere demostrar también ese patrón

---

## Notas de arquitectura / decisiones descartadas

- **ESP32 → Blob directo:** descartado por fragilidad de TLS/memoria en el ESP32; se prefiere pasar por la Pi
- **Managed Online Endpoint siempre activo como pieza principal:** descartado para v1 — coste por hora sin tráfico real, no encaja con cadencia de 3 min y con la decisión de no servir inferencia en directo todavía
- **Batch Endpoint como única pieza de inferencia "en vivo":** descartado inicialmemte por overhead de arranque de compute por evento individual; pero recuperado como pieza correcta para el proceso diario de lote (no por evento unitario)
