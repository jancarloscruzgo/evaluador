# Evaluador de audio

Script en Python que **evalúa ejercicios de síntesis de audio**. Compara el
audio de un alumno con el audio de referencia del profesor y le pone una
**nota del 0 al 10** según lo parecidos que son.

La idea es que el profesor decida, desde la línea de comandos, **en qué
características fijarse** para cada ejercicio (timbre, energía, ruido...), y el
script calcula la nota directamente. No hace falta calibrar ni darle las notas
de antemano.

---

## Requisitos e instalación

- Python 3.10 o superior
- Las librerías de `requirements.txt`

```bash
# 1. Crear un entorno virtual (recomendado)
python -m venv venv

# 2. Activarlo
#    Windows (PowerShell):
venv\Scripts\activate
#    Linux / Mac:
source venv/bin/activate

# 3. Instalar las dependencias
pip install -r requirements.txt
```

---

## Uso

Indica tu audio de referencia (`-r`) y el del alumno (`-t`), o una carpeta
entera de alumnos (`--carpeta`). Usa las rutas a tus propios archivos.

### Modo individual: evaluar un solo alumno
```bash
python evaluador.py -r referencia.wav -t alumno.wav
```
Muestra la nota de **cada característica** y la **nota final**:
```
--- PUNTUACIÓN POR CARACTERÍSTICA ---
mel_spectrogram                     ->  9.48 / 10
chroma                              ->  9.84 / 10
rms                                 ->  9.48 / 10
...
--- NOTA FINAL ---
Nota: 9.64 / 10
```

### Modo carpeta: evaluar a todos los alumnos de golpe
```bash
python evaluador.py -r referencia.wav --carpeta carpeta_alumnos
```
Compara cada `.wav` de la carpeta contra la referencia (la propia referencia se
ignora si está dentro, igual que los archivos basura `._*` de macOS) y muestra
una **tabla ordenada de mayor a menor nota**. Si algún archivo falla, lo lista
al final sin parar el resto:
```
Alumno                  Nota
-----------------------------
ajr44.wav              9.76
jtct1.wav              9.70
...
mgf80.wav              0.00
```

### Ver la ayuda completa
```bash
python evaluador.py -h
```

---

## Elegir qué características mirar

Por defecto se usan todas las del archivo `caracteristicas.json`. Hay tres
formas de decidir cuáles entran en la nota:

### 1. Filtros rápidos por consola
```bash
# Usar SOLO estas
python evaluador.py -r referencia.wav --carpeta carpeta_alumnos --only rms spectral_bandwidth mfcc_std

# Usar TODAS menos estas
python evaluador.py -r referencia.wav --carpeta carpeta_alumnos --exclude chroma tonnetz
```

### 2. El archivo `caracteristicas.json`
Es la lista de características. Cada una puede valer:
- `true` → se usa (peso 1)
- `false` → no se usa
- un **número** → se usa con ese **peso** (cuenta más en la nota final)

```jsonc
{
  "rms": 3,          // pesa el triple que una normal
  "spectral_bandwidth": true,   // peso 1
  "chroma": false    // no se usa
}
```
Así puedes dar más importancia a unas características que a otras.

### 3. Otro JSON para otro ejercicio
```bash
python evaluador.py -r referencia.wav --carpeta carpeta_alumnos --config otro_ejercicio.json
```

---

## Qué características usar en cada ejercicio

Las que mejor funcionan **dependen del tipo de ejercicio**. Sugerencias probadas:

- **Ejercicio tonal** (importa el timbre): `spectral_contrast`,
  `spectral_contrast_std`, `spectral_bandwidth`, `mfcc_std`, `rms`.
- **Ejercicio de ruido** (importa la textura): `mel_spectrogram`, `chroma`,
  `tonnetz_std`, `rms`, `spectral_bandwidth`.

```bash
# Ejemplo para un ejercicio tonal
python evaluador.py -r referencia.wav --carpeta carpeta_alumnos \
  --only spectral_contrast spectral_contrast_std spectral_bandwidth mfcc_std rms

# Ejemplo para un ejercicio de ruido
python evaluador.py -r referencia.wav --carpeta carpeta_alumnos \
  --only mel_spectrogram chroma tonnetz_std rms spectral_bandwidth
```

> Nota: en un ejercicio tonal `chroma` y `tonnetz` no sirven (dependen de la
> nota concreta), pero en uno de ruido sí, porque ahí miden la textura.

---

## Extras del modo individual

Estas opciones solo funcionan al evaluar **un** alumno (`-t`), no en carpeta.

### Dibujar gráficas (`-d` / `--display`)
```bash
python evaluador.py -r referencia.wav -t alumno.wav -d wave        # ondas
python evaluador.py -r referencia.wav -t alumno.wav -d spec        # espectrogramas
python evaluador.py -r referencia.wav -t alumno.wav -d wave spec   # las dos
```
- `wave` → dibuja la **forma de onda** de la referencia y del alumno.
- `spec` → dibuja el **espectrograma** (cómo se reparte la energía por frecuencias).

### Ver vectores por consola (`-f` / `--features`)
```bash
python evaluador.py -r referencia.wav -t alumno.wav -f temp   # vectores MFCC
python evaluador.py -r referencia.wav -t alumno.wav -f frec   # frecuencias de muestreo
python evaluador.py -r referencia.wav -t alumno.wav -f all    # las dos cosas
```

---

## Cómo se calcula la nota (por dentro)

1. **Carga y prepara** los dos audios: los pone a la misma frecuencia de
   muestreo y al mismo volumen, para comparar solo lo que importa.
2. **Extrae las características** de cada uno. Cada característica nace como una
   serie temporal (un valor por cada trocito del audio) y se resume a su
   **media** (valor típico) y su **desviación** `_std` (cuánto varía en el tiempo).
3. **Pone nota a cada característica** comparando alumno vs referencia. La regla
   general es "0 de diferencia = 10 puntos", pero según el tipo de característica
   la comparación cambia (ver la tabla de abajo).
4. **Nota final** = media de todas las características activas, teniendo en cuenta
   sus pesos.
5. **Audio vacío → 0:** si el alumno entrega silencio o un archivo sin señal
   (sin energía), la nota es 0 directamente, sin comparar.

### Cómo se compara cada tipo de característica

| Tipo | Cómo se compara | Características |
|---|---|---|
| **Escalar** (1 número) | diferencia relativa al mayor | `rms`, `bandwidth`, `rolloff`, `centroid`, `zcr`, `flatness`, `duracion_activa`... |
| **Vector** (varios números) | distancia entre vectores, relativa a su tamaño | `mfcc`, `chroma`, `spectral_contrast`, `tonnetz` |
| **Mel** (decibelios) | diferencia media en dB (a 20 dB la nota es 0) | `mel_spectrogram` |
| **Evolución temporal (DTW)** | compara cómo evoluciona el timbre en el tiempo | `mfcc_dtw` |
| **Curva de volumen** | correlación de la energía frame a frame | `correlacion_rms` |
| **Desvío relativo** | solo el % de desviación, no el valor absoluto | `spectral_centroid_relativo`, `tempo` |

Las dos de "evolución temporal" (`mfcc_dtw`, `correlacion_rms`) son útiles en
ejercicios donde importa **cómo cambia** el sonido (melodías, ritmos); en
sonidos sostenidos aportan poco.

---

## Opciones (resumen)

| Opción | Qué hace |
|---|---|
| `-r`, `--ref` | Audio de referencia (el del profesor). **Obligatorio.** |
| `-t`, `--test` | Audio de un alumno (modo individual) |
| `--carpeta` | Carpeta con audios de alumnos (modo carpeta) |
| `-c`, `--config` | JSON con las características (por defecto `caracteristicas.json`) |
| `--only` | Usar solo estas características |
| `--exclude` | Usar todas menos estas |
| `-d`, `--display` | Dibujar `wave` y/o `spec` (solo individual) |
| `-f`, `--features` | Mostrar vectores por consola: `temp`, `frec` o `all` (solo individual) |
| `-s`, `--suelo` | Nota mínima por característica (por defecto 0). Súbelo (ej. 2) para que un solo 0 no hunda la nota |

---

## Características disponibles

```
mfcc, mfcc_std, mfcc_dtw
spectral_centroid, spectral_centroid_std, spectral_centroid_relativo
spectral_bandwidth, spectral_bandwidth_std
spectral_rolloff, spectral_rolloff_std
chroma, chroma_std
rms, rms_std, duracion_activa, correlacion_rms
zero_crossing_rate, zero_crossing_rate_std
tempo
spectral_contrast, spectral_contrast_std
spectral_flatness, spectral_flatness_std
tonnetz, tonnetz_std
mel_spectrogram
```

Qué mide cada una (resumen):

| Característica | Qué mide |
|---|---|
| `mfcc` / `mfcc_std` | la "huella" del timbre |
| `mfcc_dtw` | evolución del timbre en el tiempo (DTW) |
| `spectral_centroid` | el brillo (centro de gravedad del sonido) |
| `spectral_centroid_relativo` | desvío de brillo respecto a la referencia |
| `spectral_bandwidth` | lo ancho que es el sonido |
| `spectral_rolloff` | si el sonido es más grave o agudo |
| `chroma` | energía en las 12 notas musicales (textura en ruido) |
| `rms` / `rms_std` | la energía / volumen |
| `correlacion_rms` | si el volumen sube y baja igual en el tiempo |
| `duracion_activa` | segundos con sonido de verdad |
| `zero_crossing_rate` | cuánto cruza el cero la onda (agudos, ruido) |
| `tempo` | velocidad estimada (pulsos por minuto) |
| `spectral_contrast` | diferencia entre zonas fuertes y flojas del espectro |
| `spectral_flatness` | 0 = notas claras, 1 = ruido plano |
| `tonnetz` | relaciones armónicas entre notas |
| `mel_spectrogram` | la "foto" del sonido en decibelios |

El sufijo `_std` de cualquier característica mide **cuánto varía en el tiempo**
(su evolución), en vez de su valor medio.

---

## Notas y limitaciones

- La nota es **el parecido con la referencia**: no necesita notas previas.
- **No existe una calibración universal**: cada ejercicio tiene su propio reparto
  de notas, por eso se eligen las características desde la consola.
- Las características de tipo **duración** crecen con la longitud del audio, así
  que solo tienen sentido comparando audios del mismo ejercicio.
- Si los audios tienen distinta frecuencia de muestreo, el programa los iguala
  automáticamente antes de comparar.
