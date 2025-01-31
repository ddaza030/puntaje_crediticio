from tensorflow.keras.models import load_model
import os

model_path = 'modelo_entrenado_final.h5'

modelo = load_model(model_path)
