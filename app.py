import streamlit as st
import pandas as pd
import joblib
from model import modelo
from grafica import create_gauge
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Puntaje Crediticio", layout="wide")

# Cargar el MinMaxScaler guardado
scaler = joblib.load(
    'minmax_scaler_correccion.pkl')  # Asegúrate de tener el archivo 'minmax_scaler.pkl' en el directorio adecuado


# Función para preprocesar los datos
def preprocess_data(data):
    # Cargar las columnas categóricas
    categorical_columns = ['emp_length', "home_ownership", "purpose"]

    # Aplicar codificación one-hot
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=False)

    # Para asegurarnos de que todas las columnas estén presentes, cargamos el conjunto original de columnas de las variables categóricas
    data_dummies = {
        'emp_length': ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
                       '6 years', '7 years', '8 years', '9 years', '10+ years'],
        'home_ownership': ['RENT', 'OWN', 'MORTGAGE', 'OTHER', 'NONE', 'ANY'],
        'purpose': ['credit_card', 'car', 'small_business', 'other', 'wedding',
                    'debt_consolidation',
                    'home_improvement', 'major_purchase', 'medical', 'moving', 'vacation',
                    'house',
                    'renewable_energy', 'educational'],
    }

    # Crear un DataFrame con todas las categorías
    df_dummies = pd.DataFrame({col: pd.Series(values) for col, values in data_dummies.items()})
    original_columns = pd.get_dummies(df_dummies, drop_first=False)

    # Obtener las columnas en un DataFrame de todas las posibles combinaciones de categorías
    all_columns = list(original_columns.columns)

    # Escalar las columnas numéricas
    final_columns = ['loan_amnt', 'annual_inc', 'dti', 'open_acc',
                     'emp_length_1 year', 'emp_length_10+ years', 'emp_length_2 years',
       'emp_length_3 years', 'emp_length_4 years', 'emp_length_5 years',
       'emp_length_6 years', 'emp_length_7 years', 'emp_length_8 years',
       'emp_length_9 years', 'emp_length_< 1 year', 'home_ownership_ANY',
       'home_ownership_MORTGAGE', 'home_ownership_NONE',
       'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
       'purpose_car', 'purpose_credit_card', 'purpose_debt_consolidation',
       'purpose_educational', 'purpose_home_improvement', 'purpose_house',
       'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
       'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
       'purpose_vacation', 'purpose_wedding']

    num_columns = ['loan_amnt', 'annual_inc', 'dti', 'open_acc']

    data[num_columns] = scaler.transform(data[num_columns])
    data = data.reindex(columns=final_columns, fill_value=0)
    data = data.astype(float)

    data = np.array(data)
    respuesta = int(modelo.predict(data)[0][0] * 100)

    return respuesta


# Sidebar para la navegación
st.sidebar.title("Navegación")
option = st.sidebar.radio("Seleccione una opción", ["Documentación", "Predicción"])

# Pestaña de Documentación
if option == "Documentación":
    st.markdown("""
    #### Información del Proyecto
    Esta aplicación utiliza un modelo de machine learning para predecir el puntaje crediticio de un usuario basado en diversas características financieras. El modelo fue entrenado con datos históricos.

    La información de como se realizó el modelo se encuentra en este [blog](https://deepnote.com/app/alejandra-uribe-sierra-6d3e/Modelo-de-riesgo-de-credito-0d40c66e-6bef-428d-8841-7e7903e9a4a8?utm_source=app-settings&utm_medium=product-shared-content&utm_campaign=data-app&utm_content=0d40c66e-6bef-428d-8841-7e7903e9a4a8)

    #### Créditos de Desarrollo:
    - **Juan Manuel Vera Echeverri**  
      [jverae@unal.edu.co](mailto:jverae@unal.edu.co)
    - **Daniel Daza Macías**  
      [dadazam@unal.edu.co](mailto:dadazam@unal.edu.co)
    - **Carlos Sebastián Zamora Rosero**  
      [cazamorar@unal.edu.co](mailto:cazamorar@unal.edu.co)
    - **Alejandra Uribe Sierra**  
      [aluribes@unal.edu.co](mailto:aluribes@unal.edu.co)

    #### Repositorio de GitHub:
    Puedes encontrar el código fuente y más información sobre este proyecto en el siguiente enlace:  
    [repositorio](https://github.com/ddaza030/puntaje_crediticio)

    #### Video Publicitario
    A continuación, te presentamos un video publicitario relacionado con el mundo financiero y crediticio:
    """)

    # Insertar video de YouTube
    st.video("https://youtu.be/A8_-KQ83DFs")  # Reemplaza con el enlace de tu video
    st.video("https://youtu.be/SmO42f9jaOY?si=Y2KIvVzR-TIYN4uv")

# Pestaña de Predicción
elif option == "Predicción":
    st.title("Obtenga su puntaje crediticio")

    # Crear formulario en Streamlit
    with st.form("input_form"):
        loan_amnt = st.number_input(
            "Monto listado del préstamo solicitado por el prestatario.", value=800000)
        emp_length = st.selectbox("Tiempo de empleo",
                                  ['< 1 year', '1 year', '2 years', '3 years', '4 years',
                                   '5 years',
                                   '6 years', '7 years', '8 years', '9 years',
                                   '10+ years'],
                                  index=8)  # '9 years' está en el índice 8

        home_ownership = st.selectbox("Estado de propiedad de vivienda",
                                      ['RENT', 'OWN', 'MORTGAGE', 'OTHER', 'NONE', 'ANY'],
                                      index=0)  # 'OWN' está en el índice 1

        purpose = st.selectbox("Proposito del prestamo",
                               ['credit_card', 'car', 'small_business', 'other',
                                'wedding', 'debt_consolidation',
                                'home_improvement', 'major_purchase', 'medical', 'moving',
                                'vacation', 'house',
                                'renewable_energy', 'educational'],
                               index=6)  # 'home_improvement' está en el índice 6

        open_acc = st.number_input(
            "Número de líneas de crédito abiertas en el historial de crédito del prestatario",
            value=5)
        annual_inc = st.number_input("El ingreso anual", min_value=1.0, value=60000.0)
        pago_mensual_deuda = st.number_input("Pagos mensuales de deudas actuales",
                                             min_value=0.0, value=500.0)

        # Botón para enviar el formulario
        submit_button = st.form_submit_button("Preprocesar y mostrar datos")

    # Si el formulario fue enviado
    if submit_button:
        # Crear un DataFrame con los datos ingresados
        input_data = {
            'loan_amnt': [loan_amnt],
            'dti': [(pago_mensual_deuda/(annual_inc/12))*100],
            'home_ownership': [home_ownership],
            'annual_inc': [annual_inc],
            'purpose': [purpose],
            'open_acc': [open_acc],
            'emp_length': [emp_length],
        }

        # Convertir a DataFrame
        input_df = pd.DataFrame(input_data)

        # Preprocesar los datos
        resultado = preprocess_data(input_df)

        st.write("Datos Preprocesados")
        old_min = 0
        max_val = 850
        min_val = 300
        old_max = 100

        resultado_txt = (resultado - old_min) * (max_val - min_val) / (old_max - old_min) + min_val
        fig = create_gauge(resultado, int(resultado_txt))
        st.pyplot(fig)