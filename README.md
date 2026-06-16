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

### Evaluar un solo alumno
```bash
python evaluador.py -r referencia.wav -t alumno.wav
```

### Evaluar una carpeta entera de alumnos
```bash
python evaluador.py -r referencia.wav --carpeta carpeta_alumnos
```
En modo carpeta, el script compara cada `.wav` de la carpeta contra la
referencia (la propia referencia se ignora si está dentro) y muestra una tabla
de notas ordenada.

### Elegir qué características mirar
Por defecto se usan todas las del archivo `caracteristicas.json`. Puedes filtrar:

```bash
# Usar SOLO estas
python evaluador.py -r referencia.wav --carpeta carpeta_alumnos --only rms spectral_bandwidth mfcc_std

# Usar TODAS menos estas
python evaluador.py -r referencia.wav --carpeta carpeta_alumnos --exclude chroma tonnetz
```

### Ver la ayuda completa
```bash
python evaluador.py -h
```

---

## Qué características usar en cada ejercicio

Las características que mejor funcionan **dependen del tipo de ejercicio**.
Sugerencias probadas:

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

## Cómo funciona por dentro

1. **Carga y prepara** los dos audios (misma frecuencia de muestreo, mismo volumen).
2. **Extrae las características** de cada uno. Cada característica nace como una
   serie temporal y se resume a su **media** (valor típico) y su **desviación**
   (cuánto varía en el tiempo).
3. **Pone nota a cada característica** comparando alumno vs referencia
   (0 de diferencia = 10 puntos).
4. **Nota final** = media de todas las características activas.
5. **Audio vacío → 0:** si el alumno entrega silencio o un archivo sin señal,
   la nota es 0 directamente.

---

## Opciones

| Opción | Qué hace |
|---|---|
| `-r`, `--ref` | Audio de referencia (el del profesor) |
| `-t`, `--test` | Audio de un alumno (modo individual) |
| `--carpeta` | Carpeta con audios de alumnos (modo carpeta) |
| `-c`, `--config` | JSON con las características (por defecto `caracteristicas.json`) |
| `--only` | Usar solo estas características |
| `--exclude` | Usar todas menos estas |
| `-d`, `--display` | Dibujar `wave` y/o `spec` (solo individual) |
| `-f`, `--features` | Mostrar vectores por consola (solo individual) |
| `-s`, `--suelo` | Nota mínima por característica (por defecto 0) |

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

---

## Notas y limitaciones

- La nota es **el parecido con la referencia**: no necesita notas previas.
- **No existe una calibración universal**: cada ejercicio tiene su propio reparto
  de notas, por eso se eligen las características desde la consola.
- Las características de tipo **duración** crecen con la longitud del audio, así
  que solo tienen sentido comparando audios del mismo ejercicio.
