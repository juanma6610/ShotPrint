**Propuesta de Trabajo de Fin de Máster (TFM)**

Tutor: Rafa Galvez

Estudiante: Juan Manuel Oliver

Set de Datos Principal: NBA 2015/16 (Datos de Tracking de SportVU) (Se podrían añadir los de la temporada 2013/14 y 2014/15)

Datos de PBP de nbastats o basketballreference.

---

**Objetivos del TFM**

1. **Métrica y Modelo EPV:** Creación de una nueva métrica de "gravedad" ($G_s$) que mide la atención defensiva que atrae un jugador y la manera en la desvirtúa la posición lógica de una defensa. Esta métrica se añadirá al trabajo de desarrollado de EPV, al que también se incorporará un factor de "fatiga" ($F_t$) inferido de los datos de tracking.
2. **Modelo de Simulación:** Crear un entorno para generar o aislar posesiones de entrenamiento (2v2, 2v3, 3v1) con restricciones (bote, tiempo) y evaluar su rendimiento con el modelo EPV desarrollado. Se podría ver la progresión de los jugadores a lo largo de la temporada y posibles mejoras en su juego.

---

**Alomejor hacer las dos cosas es demasiado alcance, puede que sea mejor centrarse en el primer objetivo y ver como se va avanzando**

---

**Retro-Planning del TFM (Entrega: Mayo 2026)**

**Fase 0: Fundamentos y Procesamiento de Datos (Ahora – 31 de diciembre, 2025)**

_(Duración: ~8 semanas)_

Este es el trabajo más pesado y crítico del proyecto.

- **[Ahora] Revisión de Literatura y uso de avances de otros proyectos:**

- Sintetizar artículos clave sobre EPV (trabajo en progreso)
- Investigar modelos de "control de cancha" (gravedad) y métricas de fatiga en deportes.
- Intentar usar todos los repos open-source relacionados y hacerlos funcionar en local para aprovecharlos para el trabajo. Ver como se pueden incorporar y aprovechar

- **[Mediados Nov.] Adquisición e Ingesta de Datos:**

- Cargar los datos de SportVU 2015-16.
- Cargar los datos de Play-by-Play (PBP) correspondientes. Intentar usar trabajo ya realizado por otros en este aspecto.

- **[Dic.] Limpieza y Sincronización:**

- **Tarea Crítica:** Sincronizar las marcas de tiempo de los datos de tracking (SportVU) con los eventos de PBP.
- Desarrollar _scripts_ para segmentar los datos de tracking en posesiones individuales.
- Manejar datos faltantes, jugadores fuera de cámara y errores de tracking.

- **Hito 1 (31/DIC):** Tener un conjunto de datos limpio y unificado de "posesiones", donde cada fila contiene el PBP y una referencia a los datos de tracking (trayectorias).

**Esta fase se podria acortar dependiendo de si es posible usar el trabajo de preprocesamiento de otros investigadores para acortar trabajo.**

---

**Fase 1: Feature engineering (Enero, 2026)**

_(Duración: ~4-5 semanas)_

- **[Semana 1-2] Modelo EPV Base:**

- Replicar un modelo EPV base (usando características simples) para tener un _benchmark_ que mejorar.

- **[Semana 3] Métrica de Gravedad ($G_s$):**

- Diseñar y programar las características que definen la métrica de "gravedad" (ej. distancia del defensor a atacante _off-ball_, orientación defensiva, "abandono" del tirador).

- **[Semana 4] Métrica de Fatiga ($F_t$):**

- Diseñar y programar las características de "fatiga" (ej. velocidad media en los últimos 2 min, nº de aceleraciones, tiempo de recuperación defensiva).
- Esto ya esta implementado en este repo pero puede que este deprecado hay que probar [https://github.com/christopherjenness/NBA-player-movement](https://github.com/christopherjenness/NBA-player-movement)

- **Hito 2 (31/ENE):** Tener un script que pueda tomar cualquier posesión y generar el vector de características completo: $X = [\text{estado}, G_s, F_t]$.

---

**Fase 2: Modelado y Análisis de Resultados (Febrero – 15 de marzo, 2026)**

_(Duración: ~6 semanas)_

- **[Feb.] Entrenamiento del Modelo Expandido:**

- Entrenar el modelo final: $EPV_{\text{aug}} = f(\text{estado}, G_s, F_t)$.
- Experimentar con arquitecturas de modelo (XGBoost, LSTM, etc.).

- **[Inicios Mar.] Validación y Análisis de Coeficientes:**

- Comparar $EPV_{\text{aug}}$ con $EPV_{\text{base}}$.
- **Tarea Crítica:** Analizar la importancia de las características. ¿Son $G_s$ y $F_t$ predictores estadísticamente significativos?

- **Hito 3 (15/MAR):** Tener los resultados finales, tablas y gráficos que demuestren el valor de $G_s$ y $F_t$.

---

**Fase 3: Redacción del TFM (16 de marzo – 31 de mayo, 2026)**

_(Duración: ~10-11 semanas)_

- **[Resto de Mar.] Redacción: Metodología:**

- Escribir la sección de "Datos" (Fase 0) y "Metodología" (Fase 1 y 2).

- **[Abril] Redacción: Resultados y Análisis:**

- Crear todas las visualizaciones y tablas.
- Escribir el capítulo de "Resultados", explicando el impacto de $G_s$ y $F_t$.

- **[1-15 Mayo] Redacción: Introducción y Conclusión:**

- Escribir la Introducción, Revisión de Literatura y Conclusiones (mencionando el TFM 2 como "Trabajo Futuro").

- **[16-29 Mayo] Revisión y Formato:**

- Buffer para correcciones, formato, bibliografía y revisión final.

- **Hito 4 (29/MAY):** **Entrega del TFM.**

---