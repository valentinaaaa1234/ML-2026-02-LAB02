"""Carga de LFW. La selección de identidades la completa el alumno."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_lfw_people

from src.config import DIR_CACHE
from src.datos.base import CargadorRostros, validar_conjunto
from src.modelos import ConjuntoRostros

# Criterios de selección pedidos en el enunciado.
MIN_FOTOS_POR_PERSONA = 20
MAX_IDENTIDADES = 50
RESIZE = 0.5


class CargadorLFW(CargadorRostros):
    """Fotografías de LFW con mayor diversidad de pose, luz y fondo.

    Pasos:

    1. ``fetch_lfw_people(min_faces_per_person=20, resize=0.5)``.
    2. Conservar como máximo 50 identidades, las de mayor frecuencia.
    3. Reindexar ``y`` con ``np.unique(..., return_inverse=True)``.
    4. Leer alto y ancho desde ``images.shape``.
    5. Llamar a ``validar_conjunto`` antes de retornar.
    """

    nombre = "lfw"

    def cargar(self) -> ConjuntoRostros:
        rostros = fetch_lfw_people(
            min_faces_per_person=MIN_FOTOS_POR_PERSONA,
            resize=RESIZE,
            data_home=str(DIR_CACHE),
            download_if_missing=True,
        )

        images = np.asarray(rostros.images, dtype=float)
        X = np.asarray(rostros.data, dtype=float)
        y = np.asarray(rostros.target, dtype=int)
        nombres_originales = list(rostros.target_names)

        # 2. Las identidades con más fotografías, hasta MAX_IDENTIDADES.
        frecuencias = np.bincount(y, minlength=len(nombres_originales))
        orden = np.argsort(-frecuencias, kind="stable")
        seleccionadas = np.sort(orden[:MAX_IDENTIDADES])

        mascara = np.isin(y, seleccionadas)
        images = images[mascara]
        X = X[mascara]
        y = y[mascara]

        # 3. Reindexar: las etiquetas quedan 0..C-1 sin huecos.
        ids_presentes, y = np.unique(y, return_inverse=True)
        nombres = [str(nombres_originales[i]) for i in ids_presentes]

        # Misma escala que Olivetti ([0, 1]) para que C=1.0 del SVM sea comparable.
        if X.max() > 1.0:
            X = X / 255.0
            images = images / 255.0

        # 4. Alto y ancho reales del conjunto tras el resize.
        alto, ancho = int(images.shape[1]), int(images.shape[2])

        conjunto = ConjuntoRostros(
            nombre=self.nombre,
            images=images,
            X=X,
            y=y,
            nombres_identidades=nombres,
            alto=alto,
            ancho=ancho,
        )
        # 5. Comprueba formas alineadas y valores finitos.
        validar_conjunto(conjunto)
        return conjunto
