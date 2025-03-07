from tensorflow.keras.models import load_model
import os

model_path = 'modelo_entrenado_correccion.h5'

modelo = load_model(model_path)
