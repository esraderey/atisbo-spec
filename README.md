# ATISBO-SPEC

## Built during the hackathon

Todo el código de este repositorio se escribe durante el hackathon de Meta (27 oct – 4 nov 2026). Sin código previo a esa fecha.

## How Meta models are used

- Muse Spark en la política (decide en los pasos restantes del lazo especulativo).
- SAM 3.1 en `E_sam` (segmentación del vídeo para el estado desde máscaras).

## Uso de Meta Model API (E1–E3)

- `ATISBO_LLM=api|replay|simulada` (`--modo` tiene prioridad sobre la variable).
- Clave solo por entorno: `META_API_KEY`. Nunca en el repo.
- Caché `cache/llm/<sha256(payload)>.json` (solo payload, cuerpo y latencia).
- `ATISBO_LLM=api .venv/Scripts/python.exe -m atisbo.exp.e1 --seed 0 --n 20 --p 0.02 --max-llm-calls 3500 --cache cache/llm --out results/E1.json`
- Con 1500 la corrida se trunca, porque solo la rutina ya necesita 1763 pasos para los 20 episodios.
- `ATISBO_LLM=replay` reproduce sin red (falta en caché → `CacheMiss`).
- Fixtures actuales en `tests/fixtures/llm`: los dos primeros episodios de E1 sacados de `cache/llm` con un replay (`--cache DIR` permite grabar episodios cortos directo en `tests/fixtures/llm`).
- `.venv/Scripts/python.exe -m atisbo.exp.e2 --modo api --max-llm-calls 3500 --out results/E2.json`: 50 episodios (5 semillas × 10). Con 1500 se trunca, porque hacen falta unas 2500 llamadas.
- `.venv/Scripts/python.exe -m atisbo.exp.e3 --modo api --max-llm-calls 9000 --out results/E3.json`: 3 valores de α × 50 episodios, 7064 llamadas.

## Coste de la evaluación

| Experimento | Episodios | Llamadas al LLM | Por episodio | Éxito | Coste estimado | Sin Atisbo (estimado) |
|---|---:|---:|---:|---:|---:|---:|
| E1: Muse Spark decide cada paso | 20 | 1841 | 92.05 | 1.00 | 2.36 $ | — |
| E2: Atisbo con sorpresa, τ = 0.5 | 50 | 2463 | 49.26 | 0.96 | 3.15 $ | ≈ 4600 |
| E3: CRC, α = 0.01 | 50 | 2955 | 59.10 | 1.00 | — | ≈ 4600 |
| E3: CRC, α = 0.05 | 50 | 2203 | 44.06 | 0.64 | — | ≈ 4600 |
| E3: CRC, α = 0.10 | 50 | 1906 | 38.12 | 0.18 | — | ≈ 4600 |
| E3 en total | 150 | 7064 | | | 9.31 $ | ≈ 13 800 |

- En total, E1–E3 hicieron 11 368 llamadas y costaron 14.83 $ estimados.
- «Sin Atisbo» son los episodios × 92.05, la media de E1, donde Muse Spark decide todos los pasos. Es una estimación, porque E1 corre en mundos distintos de los de E2 y E3: semilla 0 frente a Mundo(1000 + s). Sin Atisbo, los 200 episodios de E2 y E3 habrían pedido unas 18 400 llamadas; con Atisbo fueron 9527.
- El ahorro no es gratis: cuanto más acepta Atisbo, más baja el éxito. Por eso la columna Éxito va al lado.
- El coste estimado multiplica los tokens por el precio publicado de muse-spark-1.3 (https://dev.meta.ai/docs/pricing-rate-limits). En E3 cuenta también las llamadas servidas desde la caché: de las 7064, 5530 fueron a la red.
- Las llamadas van en serie, una tras otra. Con `ATISBO_LLM=replay` todo se reproduce sin red desde la caché local (`cache/` no se sube al repo).
