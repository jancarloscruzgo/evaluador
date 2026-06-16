# ==================================================================
# EVALUADOR DE AUDIO
# ==================================================================
# ¿Qué hace este programa?
#   Compara el audio de un alumno con el audio del profesor (la "referencia")
#   y le pone una nota del 0 al 10 según lo parecidos que son.
#
# ¿Cómo decide en qué fijarse?
#   Hay una lista de "características" (volumen, brillo, timbre...). En el
#   archivo caracteristicas.json están TODAS puestas en true. Desde la consola
#   tú decides cuáles usar para cada ejercicio con --only o --exclude.
#   (Cada ejercicio es un mundo: en uno importa el timbre, en otro el ruido...)
#
# Idea importante que aprendí haciendo esto:
#   librosa no te da un solo número por característica. En realidad parte el
#   audio en trocitos pequeños ("frames") y calcula la característica en cada
#   trocito -> sale una serie de muchos números. Aquí la resumimos a:
#       · su MEDIA  (el valor típico)          -> ej. "spectral_centroid"
#       · su DESVIACIÓN std (cuánto varió)     -> ej. "spectral_centroid_std"
#   La media dice "cómo suena de media" y la std dice "cómo de estable es".
#
# Dos formas de usarlo:
#   1) Un alumno:   python evaluador.py -r TEST/0.wav -t TEST/agn54.wav
#   2) Carpeta:     python evaluador.py -r TEST/0.wav --carpeta TEST
# ==================================================================

import argparse
import librosa
import librosa.sequence
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import json


# ==================================================================
# 1. CARGAR Y PREPARAR EL AUDIO
# ==================================================================
# Antes de comparar, los dos audios tienen que estar "en igualdad de
# condiciones": misma frecuencia de muestreo y mismo volumen. Si no, el
# programa compararía cosas que no tienen que ver con la calidad.

def cargar_audio(ruta):
    # sr=None -> respeta la frecuencia de muestreo original del archivo
    y, sr = librosa.load(ruta, sr=None)
    return y, sr


def igualar_sample_rate(y1, sr1, y2, sr2):
    # Si los dos audios van a distinta frecuencia, "reescalamos" el segundo
    # para que ambos vayan a la misma. Así los frames coinciden en duración.
    if sr1 != sr2:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
        sr2 = sr1
    return y1, sr1, y2, sr2


def normalizar_audios(y1, y2):
    # Pone los dos audios al mismo volumen (el pico más alto pasa a valer 1).
    # Así uno grabado más fuerte no parece "mejor" solo por ser más alto.
    y1 = librosa.util.normalize(y1)
    y2 = librosa.util.normalize(y2)
    return y1, y2


# ==================================================================
# 2. SACAR LAS CARACTERÍSTICAS DE UN AUDIO
# ==================================================================
# Devuelve un diccionario {nombre_caracteristica: valor}.
# Algunas características son UN número (escalar) y otras son una LISTA de
# números (vector). Lo aclaro al lado de cada una.

def extraer_caracteristicas(y, sr):
    c = {}

    # MFCCs: la "huella del timbre". Quitamos el primer coeficiente porque
    # solo mide el volumen general y eso ya lo miramos con el rms.
    # Es un VECTOR (un valor por cada coeficiente).
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[1:, :]
    c["mfcc"]     = np.mean(mfccs, axis=1)   # vector: media de cada coeficiente
    c["mfcc_std"] = np.std(mfccs, axis=1)    # vector: cuánto varió cada uno
    c["mfcc_dtw"] = mfccs                    # matriz completa (para DTW, ver abajo)

    # Spectral centroid: el "centro de gravedad" del sonido. Alto = más brillante.
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    c["spectral_centroid"]     = float(np.mean(centroid))   # escalar
    c["spectral_centroid_std"] = float(np.std(centroid))    # escalar

    # Spectral bandwidth: lo "ancho" que es el sonido alrededor del centro.
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    c["spectral_bandwidth"]     = float(np.mean(bandwidth))  # escalar
    c["spectral_bandwidth_std"] = float(np.std(bandwidth))   # escalar

    # Spectral rolloff: la frecuencia por debajo de la cual está casi toda
    # la energía. Otra forma de medir si el sonido es grave o agudo.
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    c["spectral_rolloff"]     = float(np.mean(rolloff))      # escalar
    c["spectral_rolloff_std"] = float(np.std(rolloff))       # escalar

    # Chroma: cuánta energía hay en cada una de las 12 notas musicales.
    # OJO: depende de la frecuencia absoluta. En ejercicios de RUIDO sí sirve
    # (mide la textura), en ejercicios de tono igual no. Es un VECTOR de 12.
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    c["chroma"]     = np.mean(chroma, axis=1)
    c["chroma_std"] = np.std(chroma, axis=1)

    # RMS: la energía (el volumen) del sonido. Escalar.
    rms = librosa.feature.rms(y=y)
    c["rms"]        = float(np.mean(rms))
    c["rms_std"]    = float(np.std(rms))
    c["rms_frames"] = rms[0]   # la curva completa de volumen (para correlacion_rms)

    # Duración activa: cuántos segundos hay sonido de verdad (rms por encima
    # del 10% del máximo). Sirve para detectar silencios o audios cortados.
    # OJO: esto CRECE con la duración del audio, así que solo tiene sentido
    # comparar audios del mismo ejercicio (de duración parecida).
    umbral_rms = float(np.max(rms)) * 0.1
    frames_activos = np.sum(rms > umbral_rms)
    c["duracion_activa"] = float(frames_activos * 512 / sr)

    # Zero crossing rate: cuántas veces la onda cruza el cero. Sube con los
    # sonidos agudos y el ruido. Escalar.
    zcr = librosa.feature.zero_crossing_rate(y=y)
    c["zero_crossing_rate"]     = float(np.mean(zcr))
    c["zero_crossing_rate_std"] = float(np.std(zcr))

    # Tempo: velocidad estimada en pulsos por minuto. Escalar.
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    c["tempo"] = float(np.atleast_1d(tempo)[0])

    # Spectral contrast: diferencia entre las zonas fuertes y flojas del
    # espectro. Bueno para el timbre. VECTOR.
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    c["spectral_contrast"]     = np.mean(contrast, axis=1)
    c["spectral_contrast_std"] = np.std(contrast, axis=1)

    # Spectral flatness: 0 = sonido con notas claras, 1 = ruido plano.
    # Truco: un audio vacío/silencio suele dar flatness = 1. Escalar.
    flatness = librosa.feature.spectral_flatness(y=y)
    c["spectral_flatness"]     = float(np.mean(flatness))
    c["spectral_flatness_std"] = float(np.std(flatness))

    # Tonnetz: relaciones armónicas entre notas. Depende de la frecuencia
    # absoluta (como chroma). VECTOR.
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    c["tonnetz"]     = np.mean(tonnetz, axis=1)
    c["tonnetz_std"] = np.std(tonnetz, axis=1)

    # Mel spectrogram: la "foto" del sonido en escala de decibelios. VECTOR.
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    c["mel_spectrogram"] = np.mean(librosa.amplitude_to_db(mel), axis=1)

    return c


# ==================================================================
# 3. PONER NOTA A CADA CARACTERÍSTICA POR SEPARADO (0 a 10)
# ==================================================================
# Cada tipo de característica se compara de una forma distinta. La regla
# general que sigo es: "0 de diferencia = 10 puntos; cuanta más diferencia,
# menos puntos". Lo que cambia es CÓMO mido esa diferencia.

def similitud_vector_magnitud(v1, v2):
    # Para VECTORES (mfcc, chroma, contrast...). Mido la distancia entre los
    # dos vectores y la divido por su tamaño medio, para que la penalización
    # sea relativa (no es lo mismo fallar 5 sobre 10 que 5 sobre 1000).
    v1 = np.atleast_1d(v1).flatten()
    v2 = np.atleast_1d(v2).flatten()
    n = min(len(v1), len(v2))
    v1, v2 = v1[:n], v2[:n]
    distancia = np.linalg.norm(v1 - v2)
    tamano = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2
    return float(max(0.0, (1.0 - distancia / (tamano + 1e-8))) * 10)


def similitud_escalar_relativa(a, b):
    # Para ESCALARES (bandwidth, rolloff, rms...). Diferencia entre los dos
    # números dividida por el mayor de ellos -> "qué porcentaje se desvía".
    a, b = float(a), float(b)
    mayor = max(abs(a), abs(b))
    if mayor < 1e-8:        # los dos son ~0: idénticos
        return 10.0
    return float(max(0.0, (1.0 - abs(a - b) / mayor)) * 10)


def similitud_mel_db(m1, m2, umbral_db=20.0):
    # Para el mel spectrogram, que está en decibelios. Uso la diferencia media
    # en dB. Si se diferencian 20 dB o más, la nota cae a 0.
    m1 = np.atleast_1d(m1).flatten()
    m2 = np.atleast_1d(m2).flatten()
    n = min(len(m1), len(m2))
    diff_db = float(np.mean(np.abs(m1[:n] - m2[:n])))
    return float(max(0.0, (1.0 - diff_db / umbral_db)) * 10)


def similitud_dtw(mat1, mat2):
    # DTW (Dynamic Time Warping) compara cómo EVOLUCIONA el timbre en el tiempo,
    # aunque los audios no estén perfectamente alineados. Útil en ejercicios
    # donde importa la evolución (melodías, ritmos), no tanto en sonidos fijos.
    D, wp = librosa.sequence.dtw(mat1, mat2, metric='euclidean')
    coste_por_frame = float(D[-1, -1]) / len(wp)
    return float(max(0.0, (1.0 - coste_por_frame / (coste_por_frame + 50))) * 10)


def similitud_centroid_relativo(c1, c2):
    # Mira solo cuánto se desvía el "brillo" del alumno respecto al del profe,
    # en porcentaje. Así no penalizamos el valor absoluto sino la desviación.
    diff_rel = abs(c1["spectral_centroid"] - c2["spectral_centroid"]) \
               / (c1["spectral_centroid"] + 1e-8)
    return float(max(0.0, (1.0 - diff_rel)) * 10)


def similitud_tempo(c1, c2):
    # Diferencia de tempo, en porcentaje sobre el mayor.
    t1, t2 = c1["tempo"], c2["tempo"]
    mayor = max(t1, t2)
    if mayor < 1e-8:
        return 10.0
    return float(max(0.0, (1.0 - abs(t1 - t2) / mayor)) * 10)


def similitud_correlacion_rms(c1, c2):
    # Compara la CURVA de volumen frame a frame (¿sube y baja igual?).
    # Sirve para ejercicios rítmicos. En sonidos sostenidos no aporta mucho
    # porque la curva es casi plana.
    r1 = np.atleast_1d(c1["rms_frames"]).flatten()
    r2 = np.atleast_1d(c2["rms_frames"]).flatten()
    n = min(len(r1), len(r2))
    r1, r2 = r1[:n], r2[:n]
    if np.std(r1) < 1e-8 or np.std(r2) < 1e-8:
        return 10.0
    corr = float(np.corrcoef(r1, r2)[0, 1])
    return float(max(0.0, corr) * 10)


# Para saber qué función usar con cada característica, las agrupo por tipo.
VECTORES = {"mfcc", "mfcc_std", "chroma", "chroma_std",
            "spectral_contrast", "spectral_contrast_std",
            "tonnetz", "tonnetz_std"}
ESCALARES = {"spectral_bandwidth", "spectral_bandwidth_std",
             "spectral_rolloff", "spectral_rolloff_std",
             "rms", "rms_std", "duracion_activa",
             "zero_crossing_rate", "zero_crossing_rate_std",
             "spectral_centroid", "spectral_centroid_std",
             "spectral_flatness", "spectral_flatness_std"}
MEL = {"mel_spectrogram"}
# Las "especiales" tienen su propia función porque no encajan en lo de arriba.


def nota_de_caracteristica(key, c1, c2):
    """Devuelve la nota (0-10) de UNA característica comparando profe vs alumno."""
    if key == "spectral_centroid_relativo":
        return similitud_centroid_relativo(c1, c2)
    if key == "mfcc_dtw":
        return similitud_dtw(c1["mfcc_dtw"], c2["mfcc_dtw"])
    if key == "tempo":
        return similitud_tempo(c1, c2)
    if key == "correlacion_rms":
        return similitud_correlacion_rms(c1, c2)
    if key in MEL:
        return similitud_mel_db(c1[key], c2[key])
    if key in ESCALARES:
        return similitud_escalar_relativa(c1[key], c2[key])
    if key in VECTORES:
        return similitud_vector_magnitud(c1[key], c2[key])
    # Si llega algo raro, lo tratamos como vector por si acaso.
    return similitud_vector_magnitud(c1[key], c2[key])


# ==================================================================
# 4. DETECTAR AUDIO VACÍO
# ==================================================================
# Algunos alumnos entregan un archivo en silencio, en blanco o que salió mal.
# No tiene sentido compararlo: si no hay sonido, la nota es 0 directamente.
# Lo detectamos porque casi no tiene energía (rms) ni duración con sonido.

def audio_vacio(c):
    return c["rms"] < 1e-3 or c["duracion_activa"] < 1e-2


# ==================================================================
# 5. LEER LA CONFIGURACIÓN (qué características usar)
# ==================================================================
# Lee el JSON y aplica los filtros que pongas en consola.
# El JSON puede tener true/false o un número (peso): 0 = no usar, N = pesa N.

def cargar_config(config_path, only=None, exclude=None):
    with open(config_path, "r") as f:
        config = json.load(f)

    pesos = {}
    for k, v in config.items():
        if isinstance(v, bool):
            pesos[k] = 1 if v else 0
        elif isinstance(v, (int, float)):
            pesos[k] = float(v)
        else:
            pesos[k] = 0

    # --only: usar SOLO estas (las demás fuera)
    if only:
        pesos = {k: (pesos[k] if k in pesos else 1) for k in only}
    # --exclude: usar todas MENOS estas
    if exclude:
        pesos = {k: (0 if k in exclude else p) for k, p in pesos.items()}

    # Nos quedamos solo con las que tienen peso > 0
    return {k: p for k, p in pesos.items() if p > 0}


# ==================================================================
# 6. JUNTAR TODO Y PONER LA NOTA FINAL
# ==================================================================

def comparar(y1, sr1, y2, sr2, pesos, suelo=0.0):
    """
    Devuelve:
      - notas: diccionario {caracteristica: (nota, peso)} para poder enseñarlo
      - nota_final: la media de todas las notas, teniendo en cuenta los pesos

    suelo: nota mínima que puede sacar cada característica (por defecto 0).
           Si lo subes (ej. 2), evitas que un solo 0 hunda la nota final.
    """
    c1 = extraer_caracteristicas(y1, sr1)   # profe (referencia)
    c2 = extraer_caracteristicas(y2, sr2)   # alumno

    # Regla de audio vacío: si el alumno no entregó sonido -> 0 y no comparamos.
    if audio_vacio(c2):
        return {}, 0.0

    notas = {}
    for key, peso in pesos.items():
        score = nota_de_caracteristica(key, c1, c2)
        notas[key] = (round(max(suelo, score), 2), peso)

    # Nota final = media ponderada (cada característica cuenta según su peso).
    total_peso = sum(p for _, p in notas.values())
    nota_final = round(sum(s * p for s, p in notas.values()) / total_peso, 2)
    return notas, nota_final


# ==================================================================
# 7. DIBUJOS Y EXTRAS (opcionales, solo modo individual)
# ==================================================================

def mostrar_wave(y1, sr1, y2, sr2, ruta1, ruta2):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    librosa.display.waveshow(y1, sr=sr1, ax=axes[0])
    axes[0].set_title(f"Onda - Referencia: {ruta1}")
    librosa.display.waveshow(y2, sr=sr2, ax=axes[1])
    axes[1].set_title(f"Onda - Test: {ruta2}")
    plt.tight_layout()
    plt.show()


def mostrar_spec(y1, sr1, y2, sr2, ruta1, ruta2):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    spec1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max)
    librosa.display.specshow(spec1, sr=sr1, x_axis="time", y_axis="hz", ax=axes[0])
    axes[0].set_title(f"Espectrograma - Referencia: {ruta1}")
    spec2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max)
    librosa.display.specshow(spec2, sr=sr2, x_axis="time", y_axis="hz", ax=axes[1])
    axes[1].set_title(f"Espectrograma - Test: {ruta2}")
    plt.tight_layout()
    plt.show()


def mostrar_features(y1, sr1, y2, sr2, ruta1, ruta2, modo):
    mfccs1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)[1:, :]
    mfccs2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)[1:, :]
    if modo in ["temp", "all"]:
        print("\n--- MFCCs (vector medio) ---")
        print(f"Referencia ({ruta1}):")
        print(", ".join(f"{v:.4f}" for v in np.mean(mfccs1, axis=1)))
        print(f"\nTest ({ruta2}):")
        print(", ".join(f"{v:.4f}" for v in np.mean(mfccs2, axis=1)))
    if modo in ["frec", "all"]:
        print("\n--- Sample Rate ---")
        print(f"Referencia ({ruta1}): {sr1} Hz")
        print(f"Test ({ruta2}):       {sr2} Hz")


# ==================================================================
# 8. CONTROL DE ERRORES
# ==================================================================

def validar_archivo(ruta):
    if not os.path.exists(ruta):
        print(f"Error: el archivo '{ruta}' no existe")
        return False
    if not ruta.endswith(".wav"):
        print(f"Error: el archivo '{ruta}' no es un .wav")
        return False
    return True


# ==================================================================
# 9. ARGUMENTOS DE CONSOLA
# ==================================================================

CARACTERISTICAS_VALIDAS = [
    "mfcc", "mfcc_std", "mfcc_dtw",
    "spectral_centroid", "spectral_centroid_std", "spectral_centroid_relativo",
    "spectral_bandwidth", "spectral_bandwidth_std",
    "spectral_rolloff", "spectral_rolloff_std",
    "chroma", "chroma_std",
    "rms", "rms_std", "duracion_activa", "correlacion_rms",
    "zero_crossing_rate", "zero_crossing_rate_std",
    "tempo",
    "spectral_contrast", "spectral_contrast_std",
    "spectral_flatness", "spectral_flatness_std",
    "tonnetz", "tonnetz_std",
    "mel_spectrogram",
]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluador de audio (alumno vs profe)")
    parser.add_argument("--ref", "-r", type=str, default=None,
                        help="Audio de referencia (el del profesor)")
    parser.add_argument("--test", "-t", type=str, default=None,
                        help="Audio de un alumno (modo individual)")
    parser.add_argument("--carpeta", type=str, default=None,
                        help="Carpeta con audios de alumnos (modo carpeta)")
    parser.add_argument("--config", "-c", type=str, default="caracteristicas.json",
                        help="JSON con las características activas")
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        choices=CARACTERISTICAS_VALIDAS,
                        help="Usar SOLO estas características")
    parser.add_argument("--exclude", type=str, nargs="+", default=None,
                        choices=CARACTERISTICAS_VALIDAS,
                        help="Usar todas MENOS estas")
    parser.add_argument("--display", "-d", type=str, nargs="+", default=None,
                        choices=["wave", "spec"],
                        help="Dibujar onda y/o espectrograma (solo individual)")
    parser.add_argument("--features", "-f", type=str, default=None,
                        choices=["temp", "frec", "all"],
                        help="Mostrar vectores por consola (solo individual)")
    parser.add_argument("--suelo", "-s", type=float, default=0.0,
                        help="Nota mínima por característica (default 0.0)")
    return parser.parse_args()


# ==================================================================
# 10. PROGRAMA PRINCIPAL
# ==================================================================

def main():
    args = parse_arguments()

    # --- Comprobaciones básicas ---
    if args.ref is None:
        print("Error: tienes que indicar --ref (el audio del profesor)")
        return
    if not validar_archivo(args.ref):
        return
    if not os.path.exists(args.config):
        print(f"Error: no se encuentra '{args.config}'")
        return

    pesos = cargar_config(args.config, args.only, args.exclude)
    if not pesos:
        print("Error: no hay características activas. Revisa el JSON o los filtros.")
        return

    # =============== MODO CARPETA ===============
    if args.carpeta is not None:
        if not os.path.isdir(args.carpeta):
            print(f"Error: '{args.carpeta}' no es una carpeta válida")
            return

        # Buscamos todos los .wav, menos la referencia y menos lo de 
        # macOS (archivos que empiezan por "._").
        ref_abs = os.path.abspath(args.ref)
        archivos = sorted([
            f for f in os.listdir(args.carpeta)
            if f.endswith(".wav")
            and not f.startswith("._")
            and os.path.abspath(os.path.join(args.carpeta, f)) != ref_abs
        ])
        if not archivos:
            print(f"Error: no hay archivos .wav en '{args.carpeta}'")
            return

        print(f"\nReferencia:            {args.ref}")
        print(f"Carpeta:               {args.carpeta}")
        print(f"Alumnos encontrados:   {len(archivos)}")
        print(f"Características:        {', '.join(pesos.keys())}")
        print("\nAnalizando...\n")

        # Cargamos la referencia una sola vez (es siempre la misma).
        y_ref, sr_ref = cargar_audio(args.ref)
        y_ref = librosa.util.normalize(y_ref)

        notas_finales = {}
        errores = []
        for nombre in archivos:
            ruta = os.path.join(args.carpeta, nombre)
            try:
                y2, sr2 = cargar_audio(ruta)
                y1, sr1, y2, sr2 = igualar_sample_rate(y_ref.copy(), sr_ref, y2, sr2)
                _, y2 = normalizar_audios(y1, y2)
                _, nota = comparar(y1, sr1, y2, sr2, pesos, suelo=args.suelo)
                notas_finales[nombre] = nota
            except Exception as e:
                errores.append((nombre, str(e)))

        # Tabla ordenada de mayor a menor nota.
        print(f"{'Alumno':<20} {'Nota':>7}")
        print("-" * 29)
        for nombre in sorted(notas_finales, key=lambda x: notas_finales[x], reverse=True):
            print(f"{nombre:<20} {notas_finales[nombre]:>6.2f}")

        if errores:
            print(f"\nErrores ({len(errores)}):")
            for nombre, msg in errores:
                print(f"  {nombre}: {msg}")
        return

    # =============== MODO INDIVIDUAL ===============
    if args.test is None:
        print("Error: indica --test (un alumno) o --carpeta (todos)")
        return
    if not validar_archivo(args.test):
        return

    print(f"\nReferencia:        {args.ref}")
    print(f"Test:              {args.test}")
    print(f"Características:    {', '.join(pesos.keys())}")
    print("\nAnalizando...")

    y1, sr1 = cargar_audio(args.ref)
    y2, sr2 = cargar_audio(args.test)
    y1, sr1, y2, sr2 = igualar_sample_rate(y1, sr1, y2, sr2)
    y1, y2 = normalizar_audios(y1, y2)

    notas, nota_final = comparar(y1, sr1, y2, sr2, pesos, suelo=args.suelo)

    # Si el diccionario está vacío es que el audio estaba vacío.
    if not notas:
        print("\nEl audio del alumno no tiene señal (silencio o archivo vacío).")
        print("\n--- NOTA FINAL ---")
        print("Nota: 0.00 / 10")
        return

    print("\n--- PUNTUACIÓN POR CARACTERÍSTICA ---")
    for k, (score, peso) in notas.items():
        marca_peso = f"(x{peso:.0f})" if peso != 1 else ""
        print(f"{k:<35} -> {score:>5.2f} / 10  {marca_peso}")

    print("\n--- NOTA FINAL ---")
    print(f"Nota: {nota_final:.2f} / 10")

    if args.features:
        mostrar_features(y1, sr1, y2, sr2, args.ref, args.test, args.features)
    if args.display:
        if "wave" in args.display:
            mostrar_wave(y1, sr1, y2, sr2, args.ref, args.test)
        if "spec" in args.display:
            mostrar_spec(y1, sr1, y2, sr2, args.ref, args.test)


if __name__ == "__main__":
    main()
