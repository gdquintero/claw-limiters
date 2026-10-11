# Exploración (cómo se llegó al esquema)

Programas de las etapas previas al borrador. Usan los módulos de la carpeta de arriba (`../`) y los datos `.npy` que genera `../run_all.sh`.

| Archivo | Qué hace | Etapa |
|---|---|---|
| `check.py` | verificación simbólica (sympy) de TDPUS-C3 (tesis de Caro Candezano) y de los esquemas κ: solo κ=1/3 es de tercer orden para promedios de celda | estudio del material |
| `pubick.py` | verificación del esquema PUBICK (artículo de Bézier, Vetor 2023) | estudio del material |
| `minbee_equivalencia.py` | MINBEE = ½(Superbee+Minmod) = minmod generalizado θ=1.5 = MC con θ=¾ (diferencia exactamente 0) | pregunta "¿dónde está hecho MINBEE?" |
| `fig_paso1.py` | paso 1: tu híbrido original frente a los cambios TOPUS→Arora–Roe y MINBEE→CFL-Superbee | `img/paso1.png` |
| `fig_paso2.py` | paso 2: tu sensor con umbral δ frente al mejor limitador de Kemm | `img/paso2.png` |
| `fig_paso3.py` | paso 3: Euler 1D en CLAWPACK | `img/paso3.png` |
| `fig_paso4.py` | paso 4: Euler 2D en CLAWPACK | `img/paso4.png` |
| `conv.py` | estudios de convergencia de variantes exploratorias (mezclas con el interruptor de Harten) | exploración descartada |
| `notas_de_trabajo.md` | bitácora completa (en inglés): análisis de cada documento, resultados numéricos de cada etapa, erratas encontradas (Kemm ec. 58, fallos en limitadores de PyClaw) | todas |

Algunas comprobaciones rápidas se hicieron directamente en la terminal y no quedaron como archivo (por ejemplo, el diagnóstico de la presión negativa en la onda explosiva o la medición de tiempos); sus resultados están en `notas_de_trabajo.md`.
