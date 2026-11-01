# PLIEGO — ATISBO-SPEC (contrato de experimentos)

Meta Global AI Developer Hackathon, pista Indie. **Todo el código de este repositorio se escribe durante el evento** (27 oct – 4 nov 2026); no se reutiliza código previo de Atisbo. Repo público. El vídeo de 3 min muestra solo lo construido aquí. Este archivo es el contrato: las órdenes a Muse lo citan por sección.

## 0. Tesis falsable

Ejecución especulativa de acciones: **Atisbo** (modelo pequeño) propone las próximas 3 acciones con una confianza; el lazo las ejecuta sin consultar al LLM mientras `conf ≥ τ` y no hay **sorpresa**; **Muse Spark** decide solo en los pasos restantes. `τ` sale de **control conforme del riesgo** (Angelopoulos, Bates, Fisch, Lei, Schuster, 2022, *Conformal Risk Control*) con cota `α` sobre la tasa de aceptación de acciones malas.

- **H1:** con `α = 0.05`, llamadas LLM por episodio ≤ 30 % de E1 y éxito ≥ E1 − 5 pp.
- **H0:** no se cumple. Un resultado negativo se publica tal cual.
- **Regla de oro:** los tests verifican la **maquinaria** (determinismo, contratos, esquemas, casos límite con predictores de control); la **hipótesis** se mide y se escribe en `results/`, nunca se asserta.

## 1. Mundo (`atisbo/env/`)

- Cuadrícula `W = 12 × H = 12`, celdas discretas. Objetos `n_obj ∈ [3, 6]`, tipo ∈ {rojo, verde, azul}; un **cubo** por tipo en posiciones fijas (esquinas); un bot.
- Acciones `A = {ARRIBA, ABAJO, IZQ, DER, TOMAR, SOLTAR, ESPERAR}` (7). `TOMAR` solo si el bot está sobre un objeto libre y no carga nada; `SOLTAR` deposita en el cubo si el bot está sobre él, si no deja el objeto en la celda. Moverse contra el borde o contra un bloqueo = quedarse.
- Episodio: **éxito** cuando todos los objetos están en su cubo; presupuesto 160 pasos. Los estados iniciales se generan resolubles por la rutina en ≤ 120 pasos sin perturbaciones (si no, se descartan y se vuelven a sacar con el mismo `random.Random`).
- **Política de rutina** (genera las trazas y define "acción buena"): objeto libre más cercano (Manhattan; desempate por `id` menor) → `TOMAR` → su cubo → `SOLTAR`. Movimiento: primero reduce `|dx|`, luego `|dy|`. Sin objetos libres y sin carga → `ESPERAR`.
- **Perturbaciones** (probabilidad `p_perturb` por paso ∈ {0, 0.02, 0.05}): (i) mover un objeto libre a una celda vacía aleatoria; (ii) añadir un objeto (hasta `n_max = 6`); (iii) bloquear 5 pasos la celda frente al bot. La rutina reacciona en el paso siguiente al nuevo estado.
- **Determinismo:** cada generador usa su `random.Random(seed)`; `generar(seed, n_episodios, p_perturb) -> list[Episodio]` con `Episodio = list[tuple[EstadoSim, Accion]]`; dos llamadas iguales → `pickle` byte a byte igual.
- **Render:** `render(estado) -> np.ndarray[H·c, W·c, 3] uint8`, `c = 32 px`; colores fijos por tipo; bot gris oscuro; cubos como marcos del color; objeto cargado dibujado sobre el bot. `a_mp4(frames, ruta, fps=10)` H.264 (`imageio-ffmpeg` o `ffmpeg` de sistema).

## 2. Estado (`atisbo/estado.py`)

- `EstadoSim` (dataclass **congelada**): `bot: (x, y)`, `carga: tipo | None`, `objetos: tuple[(id, tipo, x, y, en_cubo)]`, `cubos: {tipo: (x, y)}`, `bloqueo: (x, y) | None`, `paso: int`.
- `vector(estado) -> np.float32[D]`, `D` fijo (objetos rellenados hasta `n_max = 6` con máscara de presencia): bot normalizado, carga one-hot(4), por objeto (presencia, pos normalizada, tipo one-hot(3), en_cubo), bloqueo (presencia + pos).
- `EstadoSam`: el mismo `EstadoSim` construido desde máscaras: `estado_desde_mascaras(mascaras_por_id: dict[int, np.bool_[H·c, W·c]], frame) -> EstadoSim`. Centroide → celda; tipo por color medio del recorte (distancia mínima a los colores fijos); `en_cubo` por pertenencia a la celda del cubo; bot = máscara del color del bot. Sin bot, `bot = None`. La carga sale del color de la zona del parche en la celda del bot; el bloqueo, de su color; los cubos son los fijos. Si SAM da el parche de la carga pero no el bot, el bot sale del parche. `SeguidorEstado` (`atisbo/sam.py`) mantiene entre frames lo que la imagen tapa (objeto bajo el bot, pilas del mismo tipo en un cubo) y los ids; sin bot, estima el estado con la acción ejecutada y la dinámica conocida; un objeto no visto dura un frame más.

## 3. Atisbo (`atisbo/modelo.py`)

- Entrada: ventana de `k = 8` vectores de estado `s_{t−7..t}` (relleno con el primero si el episodio es más corto). Salida: logits de las acciones `a_t, a_{t+1}, a_{t+2}` (`3 × 7`; **`a_t` es la que se ejecuta en el estado actual `s_t`**), vectores de los estados `s_{t+1}, s_{t+2}, s_{t+3}` (`3 × D`), **confianza** = producto de las probabilidades máximas de las 3 acciones. Etiquetas tomadas del episodio `[(s_t, a_t)]`; los últimos 3 pasos de cada episodio no generan muestra.
- Arquitectura: `GRU(hidden = 128)` + cabezas lineales; **≤ 1 M parámetros** (`n_parametros()`). Pérdida: CE(acciones) + `0.1 · MSE(estados)`. Entrenamiento ≤ 5 min en CPU. `torch.manual_seed(seed)`; `torch.use_deterministic_algorithms(True)`.
- API: `Atisbo.entrenar(episodios, seed, epocas) -> Atisbo`; `predecir(ventana) -> Prediccion(acciones: tuple[3], estados: np.ndarray[3, D], confianza: float)`; `guardar(ruta)`, `cargar(ruta)`.
- Predictores de control para tests (`atisbo/control.py`), misma API que `Atisbo`: `PredictorOraculo` (simula la rutina 3 pasos con la transición determinista **sin perturbaciones** y devuelve esas acciones y estados con `conf = 1.0`; así una perturbación real produce sorpresa), `PredictorNulo` (`conf = 0.0`), `PredictorRuido(p)` (rutina con prob. `1−p`, acción uniforme con prob. `p`, `conf = 1−p`).

## 4. Baselines (`atisbo/baselines.py`)

`repetir_ultima` (repite la última acción 3 veces), `ngram3` (frecuencias de la acción siguiente dadas las 2 últimas; desempate por orden de `A`), `azar`. Plan B: `velocidad_constante` y `kalman` sobre posiciones del bot.

## 5. Lazo especulativo (`atisbo/lazo.py`)

```
mientras no fin:
    pred = predictor.predecir(ventana)
    si pred.confianza < τ: accion = llm(estado); fuente = "llm"; ejecutar; continuar
    para j en 0..2:
        si j > 0 y sorpresa(estado_obs, pred.estados[j-1]) > ε: romper   # replanificar
        ejecutar pred.acciones[j]; fuente = "atisbo"
```
- `sorpresa(obs, pred) = ‖vector(obs) − pred‖₂`; `ε` = percentil 95 del error de predicción a 1 paso en trazas **sin** perturbación (calibrado en E2 y guardado en `results/E2.json`).
- Registro por paso: `fuente`, `conf`, `sorpresa`, `latencia_ms`, `accion`, `accion_rutina`.
- Métricas por episodio: `llamadas_llm`, `pasos`, `exito`, `racha_media` (longitud media de secuencias consecutivas con fuente `atisbo`), `malas_aceptadas` (pasos con fuente `atisbo` y `accion ≠ accion_rutina`), `tasa_malas_aceptadas = malas_aceptadas / pasos`.

## 6. LLM y SAM (`atisbo/meta_api.py`, `atisbo/politica_llm.py`, `atisbo/sam.py`)

- Modelo `muse-spark-1.3` vía Meta Model API; **firma exacta copiada de `dev.meta.ai/docs`** (quickstart, reasoning), no de memoria. Clave en `META_API_KEY`; nunca en el repo; `.env` en `.gitignore`.
- Política: prompt = reglas de §1 + estado en JSON → respuesta JSON `{"accion": "<A>"}` (salida estructurada si la API la ofrece; si no, parseo estricto + 1 reintento; si falla, `ESPERAR` y contador `llm_invalidas`). Esfuerzo de razonamiento mínimo disponible.
- **Registro/reproducción:** cada petición se guarda en `cache/llm/<sha256(payload)>.json`. `ATISBO_LLM=api|replay|simulada`: `replay` nunca toca la red (falta en caché → `CacheMiss`); `simulada` = `PredictorRuido(0.1)` como política, rotulada `"modo": "simulada"` en todo resultado. **Los tests corren siempre en `replay` o `simulada`.** Cap `--max-llm-calls` por corrida (por defecto 1500); al alcanzarlo, la corrida termina con `"truncada": true`.
- SAM 3.1: `model = "sam-3.1"`, Responses API con `stream: true` y `metadata.include_confidence = "true"`, un solo prompt, `"colored square"` (sondas del 31-oct: `robot` y `red cube` no encuentran nada en este render). El lazo no conoce los frames por adelantado, así que percibe cada paso con una imagen PNG en línea (`input_image`, 2.50 $ por 1000 imágenes); los episodios enteros van como vídeo MP4 H.264 en línea (`input_video`, 0.20 $ por 1000 frames, mínimo 50 por petición). Salida en texto con tokens especiales, un renglón por frame; parser `meta-sam-parser` (Python). Misma política de registro/reproducción en `cache/sam/`. Límite 16 objetos por frame.

## 7. Control conforme (`atisbo/conformal.py`)

- Score `s = 1 − confianza`. Aceptar si `s ≤ λ`. Pérdida por paso `L_i(λ) = 1[s_i ≤ λ ∧ accion_i ≠ accion_rutina_i] ∈ {0, 1}`, **monótona no decreciente en λ**, `B = 1`.
- Sobre calibración de `n` pasos: `R̂_n(λ) = (1/n) Σ L_i(λ)`; `λ̂ = sup{λ : (n/(n+1)) · R̂_n(λ) + 1/(n+1) ≤ α}` (búsqueda sobre los `s_i` ordenados). `τ = 1 − λ̂`. Garantía: `E[L(λ̂)] ≤ α` = tasa de pasos con acción mala aceptada ≤ α.
- Reportar en test separado: `tasa_malas_aceptadas` (la cantidad controlada) y, como secundaria, `malas_aceptadas / aceptadas` (condicional, no garantizada). `α ∈ {0.01, 0.05, 0.10}`. Si `α` no admite ningún `λ` → `λ̂ = −∞` (no se acepta nada).

## 8. Resultados (`results/<EXP>.json`, `atisbo/exp/schema_results.json`)

```json
{"experimento": "E2", "fecha_utc": "...", "commit": "...", "seeds": [0,1,2,3,4],
 "config": {"p_perturb": 0.02, "alpha": 0.05, "tau": 0.83, "epsilon": 0.19, "n_episodios": 50},
 "metricas": {"llamadas_llm": {"media": 7.1, "ic95": [5.9, 8.4], "por_seed": [...]}, "...": {}},
 "llm": {"modo": "api|replay|simulada", "modelo": "muse-spark-1.3", "llamadas": 355},
 "notas": ["..."]}
```
Validación con JSON Schema draft 7 (validador propio sin dependencias). `ic95` por bootstrap (1000 remuestreos, semilla fija) sobre medias por episodio. Dos corridas con la misma semilla → JSON idéntico salvo `fecha_utc`.

## 9. Experimentos

| Id | Pregunta | Diseño | Lectura |
|---|---|---|---|
| E0 | ¿Hay estructura que predecir? | Atisbo vs `repetir_ultima` vs `ngram3` vs `azar`; `acierto@1` y `acierto@3`; `p_perturb` ∈ {0, 0.02}; 5 semillas | **Go** si Atisbo supera al mejor baseline en `acierto@3` por encima del IC95; si no, **Plan B** |
| E1 | Baseline caro | `PredictorNulo` (LLM en cada paso) con `muse-spark-1.3`; 20 episodios × `p_perturb = 0.02` | referencia de llamadas, latencia, éxito |
| E2 | Lazo cerrado | Atisbo + sorpresa + fallback LLM, `τ` fijo 0.5 y `ε` calibrado | llamadas ≤ 0.3·E1 y éxito ≥ E1 − 5 pp → H1 |
| E3 | Calibración conforme | barrido `α`; `τ(α)`; riesgo empírico en test | riesgo empírico ≤ `α` + tolerancia IC |
| E4 | Ablaciones | sin conforme (`τ` fijo) · sin sorpresa · `E_sam` vs `E_sim` · horizonte 1 vs 3 | cada componente aporta o sale del relato |
| E5 | Robustez | 5 semillas × `p_perturb` ∈ {0, 0.02, 0.05}; IC95 bootstrap | se publica tal cual |

**Plan B** (si E0 = no-go): Atisbo como tracker predictivo — SAM 3.1 segmenta y sigue al bot en vídeo; Atisbo predice su posición a 3 pasos vs `velocidad_constante` y `kalman`; mismas semillas e IC.

## 10. Reglas del evento que afectan al código

Sin código previo al 27 oct. Repo público antes de la entrega. `README.md` con sección "Built during the hackathon" (todo) y "How Meta models are used" (Muse Spark en la política; SAM 3.1 en `E_sam`). Secretos solo por entorno. Reclamar los $150 de créditos antes del 4 nov 9:00 PM ET.
