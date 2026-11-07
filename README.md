# Laboratorio pequeño de kernels SVM

**Interactive demo:** [SVM Kernel Atlas](https://rubeng1009.github.io/svm-kernel-atlas/) ·
**Dashboard source:** [`dashboard/`](dashboard/) ·
**Reproducible experiment:** [`experiment.py`](experiment.py)

An interactive study of six SVM kernels on two synthetic datasets. Rotate the
decision surfaces, inspect predictions, and compare kernels on the same fixed
train/test split. The figures are educational results from one split, not a
general performance benchmark.

Experimento original con dos conjuntos sintéticos bidimensionales (`make_moons` y
`make_circles`). Compara `SVC` con seis kernels: `linear`, `rbf`, `poly`,
`laplacian`, `rational_quadratic` y `cosine`, usando los
mismos ejemplos de entrenamiento y prueba para cada kernel. No usa código,
figuras ni contenido del libro.

## Instalación y ejecución

Desde esta carpeta, con `uv` y Python 3.13:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python experiment.py --vary gamma --values 0.1 1 10
.venv/bin/python -m pytest -q tests
```

Si Python 3.13 ya está disponible como `python3.13`, también sirve
`python3.13 -m venv .venv` seguido de
`.venv/bin/python -m pip install -r requirements.txt`. `experiment.py` sin
argumentos genera solo la comparación base. `--output DIR` cambia el
directorio de salida.

Las salidas de esta ejecución están en [`output/`](output/):

- [`kernel_comparison.png`](output/kernel_comparison.png): los seis kernels en ambos datasets.
- [`sweep_gamma_rbf.png`](output/sweep_gamma_rbf.png) y
  [`sweep_gamma_poly.png`](output/sweep_gamma_poly.png),
  [`sweep_gamma_laplacian.png`](output/sweep_gamma_laplacian.png) y
  [`sweep_gamma_rational_quadratic.png`](output/sweep_gamma_rational_quadratic.png): cambian solo `gamma`.
- [`metrics.json`](output/metrics.json): valores completos, incluidos accuracy
  de entrenamiento/prueba y número total de vectores de soporte.

## Vídeo vertical para explicar el experimento

`animate.py` genera un MP4 4:5 de 26 segundos, sin audio y con texto integrado,
para revisar antes de compartir. Usa los modelos y el split del experimento:
primero compara la frontera lineal y RBF sobre los mismos círculos; después
muestra tres valores de `gamma` para RBF sobre las mismas lunas. Los cambios
entre fronteras son transiciones visuales entre modelos entrenados; los puntos
no se mueven. Los porcentajes son *accuracy* de un único conjunto de prueba
de 150 ejemplos, no una estimación general. En círculos, RBF clasifica bien
58 ejemplos de prueba más que el modelo lineal (145 frente a 87).

```bash
.venv/bin/python animate.py
```

Archivos: [`kernel_story.mp4`](output/kernel_story.mp4),
[`kernel_story_cover.png`](output/kernel_story_cover.png) y
[`kernel_story_preview.png`](output/kernel_story_preview.png). El vídeo es
un recurso local; el script no publica nada en LinkedIn.

### Versión 3D

```bash
.venv/bin/python animate_3d.py
```

[`kernel_story_3d.mp4`](output/kernel_story_3d.mp4) muestra la función de
decisión de cada SVM como una superficie tridimensional en lenta rotación.
Los puntos de entrenamiento y prueba y la frontera de puntuación cero están
proyectados sobre el suelo. Los valores de la función de decisión se recortan
visualmente a ±2,2 para mantener una escala común; la frontera se calcula
con los valores originales. Las cifras provienen del mismo split con semilla
42: 87/150 frente a 145/150 en círculos y 131/150, 145/150 y 143/150 en
las tres configuraciones de `gamma` para lunas. Es una representación
explicativa, no un dataset tridimensional ni una segunda evaluación.

El diseño aplica el principio de la skill [Anatomy](https://github.com/wheresryan22/anatomy)
de que la figura y sus lecturas procedan del mismo modelo. La superficie
matemática se renderiza con Matplotlib porque es el objeto que explica este
experimento; Anatomy está orientada a figuras isométricas de mecanismos
físicos. También se generan
[`kernel_story_3d_cover.png`](output/kernel_story_3d_cover.png) y
[`kernel_story_3d_preview.png`](output/kernel_story_3d_preview.png).

## Dashboard interactivo

Abre [`dashboard/index.html`](dashboard/index.html) en un navegador moderno.
Funciona localmente y sin conexión: Plotly.js está incluido en
`dashboard/vendor/`. Arrastra para girar la superficie 3D, usa la rueda para
acercarte o alejarte, selecciona un punto o cambia a la vista 2D.
La portada dibuja una muestra de puntos y la frontera
real del modelo RBF con `gamma = 1` en círculos. Los controles
permiten elegir dataset, kernel y `gamma`, ocultar capas, destacar errores de
prueba o vectores de soporte y comparar dos modelos con la misma cámara.
Se pueden abrir configuraciones concretas con parámetros de URL; por ejemplo,
[`cosine` frente a RBF en círculos](dashboard/index.html?dataset=circles&kernel=cosine&compare=rbf:1&view=2d#explore).
El panel muestra los aciertos en los 150 ejemplos de prueba, ofrece accuracy de
entrenamiento, vectores de soporte y matriz de confusión en «More model metrics»,
y permite inspeccionar la predicción y el score de cada punto. «Save PNG» guarda
el gráfico principal. «Copy view link» copia la configuración del explorador;
si se abre desde el archivo local o localhost, el enlace solo sirve en ese equipo.

El explorador presenta cuatro capas: puntos, frontera, superficie y vectores de
soporte. Permite elegir entre los dos conjuntos, los seis kernels y los valores de
gamma, comparar dos modelos, elegir un punto de prueba con el teclado o pulsar un
punto en la escena, y reproducir o detener el recorrido. Los puntos se proyectan sobre el
plano de score cero; su altura en la escena no representa un atributo de entrada.
La superficie usa el score precomputado y se recorta solo para la representación.

`dashboard/build_data.py` genera `data.json` y `data.js` directamente de
`experiment.py`, con las predicciones y la función de decisión de los modelos.
`data.js` permite abrir el HTML sin un servidor. Para regenerar los datos:

```bash
.venv/bin/python dashboard/build_data.py
```

Las opciones de `gamma` son 0,1, 1 y 10 para RBF, polinómico, Laplaciano y
racional cuadrático. El kernel Laplaciano usa distancia Manhattan
`exp(-gamma · ||x-y||₁)`; el racional cuadrático usa
`(1 + gamma · ||x-y||₂²)⁻¹` (`alpha = 1`); el de coseno usa el producto
normalizado y no depende de `gamma`. El valor numérico de `gamma` no tiene
el mismo significado geométrico en todos los kernels. El dashboard no calcula valores
arbitrarios ni ajusta modelos en el navegador. No representa datos originales
en tres dimensiones: la altura es el puntaje de decisión y se recorta a ±2,2
solo para dibujarla. La frontera de puntuación cero, las predicciones y las
métricas usan los valores sin recortar. La demo interactiva está disponible en
[GitHub Pages](https://rubeng1009.github.io/svm-kernel-atlas/).

### Capturas para compartir

- [Vista general del atlas](dashboard/shots/01-atlas-overview.png).
- [Coseno frente a RBF en círculos, vista 2D](dashboard/shots/02-circles-cosine-vs-rbf.png): 73/150 frente a 145/150 aciertos en prueba.
- [Laplaciano en lunas, vista 3D](dashboard/shots/03-moons-laplacian.png): 143/150 aciertos en prueba.
- [Racional cuadrático en círculos, vista 3D](dashboard/shots/04-circles-rational-quadratic.png): 145/150 aciertos en prueba.

Son capturas del dashboard local con los mismos datos exportados; se recortó
la página para centrar el instrumento, sin alterar las gráficas ni las cifras.

En las figuras, los puntos rellenos son de entrenamiento, los aros son de
prueba y la línea negra es la frontera de decisión. Azul y naranja indican
las dos clases. Las figuras se generan localmente a 180 dpi.

## Cambiar un parámetro cada vez

`--c`, `--gamma` y `--degree` fijan la configuración base. El barrido cambia
solamente el parámetro nombrado en `--vary`; los demás quedan fijos. Acepta
entre dos y cuatro valores positivos. Por ejemplo:

```bash
.venv/bin/python experiment.py --c 1 --gamma 1 --degree 3 --vary c --values 0.1 1 10
.venv/bin/python experiment.py --c 1 --gamma 1 --degree 3 --vary degree --values 2 3 4
```

El barrido de `C` muestra los seis kernels, el de `gamma` muestra RBF,
polinómico, Laplaciano y racional cuadrático, y el de grado muestra el
polinómico. Los kernels lineal y coseno no usan `gamma` ni grado.
Usar otro `--output` por barrido evita
reemplazar las figuras de una ejecución anterior.

## Reproducibilidad y comparación

Semilla predeterminada: **42** (`--seed` permite cambiarla). Cada dataset
contiene 500 ejemplos; 350 van a entrenamiento y 150 a prueba, con split
estratificado 70/30. Dentro de cada dataset, los seis kernels y todos los
valores de un barrido reutilizan exactamente los mismos índices. Cada modelo
es un `Pipeline(StandardScaler, SVC)`: el escalador aprende media y varianza
solo de entrenamiento. Las pruebas comprueban los índices, el balance de
clases, las estadísticas del escalador y la repetición de datos, predicciones
y métricas. Las versiones directas usadas están fijadas en `requirements.txt`.

## Cuatro observaciones de la ejecución incluida

Con semilla 42, `C=1`, `gamma=1` y grado 3:

1. En *moons*, RBF obtuvo accuracy de prueba **0,967** frente a **0,853**
   del lineal. La frontera curva se ajusta mejor a estas dos lunas en esta
   partición.
2. En *circles*, RBF obtuvo **0,967** frente a **0,580** del lineal y
   **0,627** del polinómico de grado 3. Su frontera cerrada captura la
   estructura circular de estos datos.
3. Al subir `gamma` de 1 a 10 para RBF en *moons*, el accuracy de
   entrenamiento pasó de **0,966** a **0,977**, pero el de prueba bajó de
   **0,967** a **0,953** y los vectores de soporte subieron de **74** a
   **155**. Aquí, una frontera más flexible no mejoró la prueba.
4. En *circles*, el kernel de coseno consiguió **0,487** en prueba mientras
   RBF consiguió **0,967**. El coseno compara dirección tras el escalado y
   descarta la distancia radial que distingue las dos clases concéntricas.

Son observaciones de una sola partición, no estimaciones de rendimiento
general. Los datasets son sintéticos y bidimensionales; esto no es un
benchmark real ni una recomendación universal de hiperparámetros. Ningún
kernel gana siempre. Para sacar conclusiones sobre datos reales harían falta
datos del problema, validación repetida y ajuste dentro de entrenamiento.

## Typography

The dashboard uses Atkinson Hyperlegible Next, self-hosted in [`dashboard/fonts/`](dashboard/fonts/) under the SIL Open Font License 1.1 ([`OFL.txt`](dashboard/fonts/OFL.txt)). It was chosen because its letterforms and numerals are built to be told apart, and the page is read through its figures.
