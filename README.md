# 🍳 Recetario Inteligente — RAG Multimodal con Entrada y Salida por Voz

Sistema de recomendación de recetas que combina **búsqueda semántica**, **normalización de consultas con LLM** y un **pipeline de voz completo** (habla → texto → búsqueda → texto → habla). El usuario puede pedir una receta hablando ("tengo tomate, huevo y cebolla, qué puedo cocinar") y recibe tanto los resultados en pantalla como una respuesta hablada generada dinámicamente.

> Proyecto de la asignatura de Procesamiento del Lenguaje Natural — enfocado en comparar técnicas de *retrieval* (léxico vs. semántico), re-ranking, y construir un sistema RAG end-to-end con interfaz real.

## 🎥 Demo

- 🎙️ [`audio_query.wav`](./audio_query.wav) — ejemplo de consulta hablada por el usuario.
- 🔊 [`respuesta_recetas.wav`](./respuesta_recetas.wav) — respuesta generada por el sistema, sintetizada con Kokoro TTS.

## 🧠 Arquitectura del sistema

```
🎙️ Voz del usuario
     │  Whisper (small) — transcripción
     ▼
📝 Texto de la consulta
     │  LLM local (Ollama) — normalización y corrección
     ▼
🔍 Query limpia
     │  multilingual-e5-large — embeddings + ChromaDB (similitud coseno)
     ▼
📋 Recetas recuperadas
     │  LLM local — genera una respuesta hablada natural
     ▼
🔊 Kokoro-ONNX — síntesis de voz
     ▼
🗣️ Respuesta hablada al usuario
```

## 🔬 Del notebook a la app: proceso de investigación

El notebook (`proyecto_nlp.ipynb`) documenta el proceso completo de experimentación antes de construir la app final:

1. **EDA** sobre ~22.000 recetas: distribución por categoría, dificultad, tiempo de preparación e ingredientes.
2. **Feature engineering**: parseo de tiempos de texto libre a minutos, categorización de dificultad y tiempo, y construcción de un texto semántico por receta.
3. **Retrieval léxico (BM25)** vs. **retrieval semántico (embeddings + ChromaDB)** — comparación de varios modelos de embeddings, quedándonos con `multilingual-e5-large` por ser el que mejor capturaba sinónimos e ingredientes relacionados.
4. **Re-ranking** con cross-encoders sobre los candidatos recuperados, para evaluar si aportaban mejora sobre E5 puro.
5. **Visualización UMAP** del espacio semántico de las recetas.
6. **Normalización de consultas con LLM**: un modelo pequeño en local (vía Ollama) corrige errores ortográficos, interpreta consultas abstractas ("algo dulce para el postre") y descarta preguntas no relacionadas con cocina.
7. **Pipeline de audio**: transcripción con Whisper, y generación de una respuesta hablada natural (no solo lectura de resultados) sintetizada con Kokoro-ONNX.
8. **Interfaz final en Streamlit** (`app.py`), que integra todo el pipeline en una aplicación usable.

## 🛠️ Tecnologías

| Componente | Tecnología |
|---|---|
| Interfaz | Streamlit |
| Embeddings semánticos | `intfloat/multilingual-e5-large` |
| Base vectorial | ChromaDB (similitud coseno) |
| Normalización de queries | LLM local vía Ollama |
| Transcripción de voz | Whisper (small) |
| Síntesis de voz | Kokoro-ONNX |
| Retrieval léxico (comparativa) | BM25 |


## 💡 Aprendizajes clave

- El retrieval semántico (E5 + ChromaDB) supera claramente al léxico (BM25) en consultas con sinónimos o ingredientes relacionados, aunque BM25 sigue siendo más rápido y predecible con coincidencias exactas.
- Los cross-encoders de re-ranking no aportaron una mejora significativa sobre E5 en este dominio y volumen de datos.
- Un LLM pequeño en local es suficiente para tareas de normalización de texto (corrección ortográfica, interpretación de consultas abstractas), sin necesidad de un modelo grande ni de APIs externas.
- Generar la respuesta hablada con un LLM (en vez de leer literalmente los resultados) hace que la interacción por voz se sienta natural en lugar de mecánica.

## 📄 Licencia

Proyecto con fines educativos. El dataset de recetas se usa con fines de investigación académica.
