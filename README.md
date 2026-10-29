# ATISBO-SPEC

## Built during the hackathon

Todo el código de este repositorio se escribe durante el hackathon de Meta (27 oct – 4 nov 2026). Sin código previo a esa fecha.

## How Meta models are used

- Muse Spark en la política (decide en los pasos restantes del lazo especulativo).
- SAM 3.1 en `E_sam` (segmentación del vídeo para el estado desde máscaras).

## Uso de Meta Model API (E1)

- `ATISBO_LLM=api|replay|simulada` (`--modo` tiene prioridad sobre la variable).
- Clave solo por entorno: `META_API_KEY`. Nunca en el repo.
- Caché `cache/llm/<sha256(payload)>.json` (solo payload, cuerpo y latencia).
- `ATISBO_LLM=api .venv/Scripts/python.exe -m atisbo.exp.e1 --seed 0 --n 20 --p 0.02 --max-llm-calls 3500 --cache cache/llm --out results/E1.json`
- Con 1500 la corrida se trunca, porque solo la rutina ya necesita 1763 pasos para los 20 episodios.
- `ATISBO_LLM=replay` reproduce sin red (falta en caché → `CacheMiss`).
- Fixtures actuales en `tests/fixtures/llm`: los dos primeros episodios de E1 sacados de `cache/llm` con un replay (`--cache DIR` permite grabar episodios cortos directo en `tests/fixtures/llm`).
