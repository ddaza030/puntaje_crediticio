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
    'minmax_scaler_new.pkl')  # Asegúrate de tener el archivo 'minmax_scaler.pkl' en el directorio adecuado


# Función para preprocesar los datos
def preprocess_data(data):
    # Cargar las columnas categóricas
    categorical_columns = ["grade", "home_ownership", "purpose"]

    # Aplicar codificación one-hot
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=False)

    # Para asegurarnos de que todas las columnas estén presentes, cargamos el conjunto original de columnas de las variables categóricas
    data_dummies = {
        'grade': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'home_ownership': ['RENT', 'OWN', 'MORTGAGE', 'OTHER', 'NONE', 'ANY'],
        'purpose': ['credit_card', 'car', 'small_business', 'other', 'wedding', 'debt_consolidation',
                    'home_improvement', 'major_purchase', 'medical', 'moving', 'vacation', 'house',
                    'renewable_energy', 'educational'],
    }

    # Crear un DataFrame con todas las categorías
    df_dummies = pd.DataFrame({col: pd.Series(values) for col, values in data_dummies.items()})
    original_columns = pd.get_dummies(df_dummies, drop_first=False)

    # Obtener las columnas en un DataFrame de todas las posibles combinaciones de categorías
    all_columns = list(original_columns.columns)

    # Escalar las columnas numéricas
    final_columns = ['loan_amnt', 'int_rate', 'annual_inc', 'open_acc',
                     'revol_bal', 'total_acc', 'out_prncp', 'total_pymnt', 'total_rec_int',
                     'last_pymnt_amnt', 'tot_cur_bal', 'total_rev_hi_lim', 'grade_A',
                     'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G',
                     'home_ownership_MORTGAGE', 'home_ownership_NONE',
                     'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
                     'purpose_car', 'purpose_credit_card', 'purpose_debt_consolidation',
                     'purpose_educational', 'purpose_home_improvement', 'purpose_house',
                     'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
                     'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
                     'purpose_vacation', 'purpose_wedding']

    num_columns = ['loan_amnt', 'int_rate', 'annual_inc', 'open_acc', 'revol_bal',
                   'total_acc', 'out_prncp', 'total_pymnt', 'total_rec_int',
                   'last_pymnt_amnt', 'tot_cur_bal', 'total_rev_hi_lim']

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
        loan_amnt = st.number_input("Monto listado del préstamo solicitado por el prestatario.")
        int_rate = st.number_input("Tasa de interés del préstamo.")
        grade = st.selectbox("Calificación asignada al préstamo por LC.", ['B', 'C', 'A', 'E', 'F', 'D', 'G'])
        home_ownership = st.selectbox("Estado de propiedad de vivienda",
                                      ['RENT', 'OWN', 'MORTGAGE', 'OTHER', 'NONE', 'ANY'])
        annual_inc = st.number_input("El ingreso anual", min_value=0.0)
        purpose = st.selectbox("Proposito del prestamo",
                               ['credit_card', 'car', 'small_business', 'other', 'wedding', 'debt_consolidation',
                                'home_improvement', 'major_purchase', 'medical', 'moving', 'vacation', 'house',
                                'renewable_energy', 'educational'])
        open_acc = st.number_input("Número de líneas de crédito abiertas en el historial de crédito del prestatario")
        revol_bal = st.number_input("Saldo total de crédito rotativo")
        total_acc = st.number_input(
            "Número total de líneas de crédito actualmente en el historial crediticio del prestatario", min_value=0.0)
        out_prncp = st.number_input("Principal pendiente restante para el monto total financiado", min_value=0.0)
        total_pymnt = st.number_input("Pagos recibidos hasta la fecha para el monto total financiado", min_value=0.0)
        total_rec_int = st.number_input("Intereses recibidos hasta la fecha", min_value=0.0)
        last_pymnt_amnt = st.number_input("Último monto total del pago recibido.", min_value=0.0)
        tot_cur_bal = st.number_input("Saldo total actual de todas las cuentas", min_value=0.0)
        total_rev_hi_lim = st.number_input("Límite de crédito total en líneas de crédito rotativas.", min_value=0.0)

        # Botón para enviar el formulario
        submit_button = st.form_submit_button("Preprocesar y mostrar datos")

    # Si el formulario fue enviado
    if submit_button:
        # Crear un DataFrame con los datos ingresados
        input_data = {
            'loan_amnt': [loan_amnt],
            'int_rate': [int_rate],
            'grade': [grade],
            'home_ownership': [home_ownership],
            'annual_inc': [annual_inc],
            'purpose': [purpose],
            'open_acc': [open_acc],
            'revol_bal': [revol_bal],
            'total_acc': [total_acc],
            'out_prncp': [out_prncp],
            'total_pymnt': [total_pymnt],
            'total_rec_int': [total_rec_int],
            'last_pymnt_amnt': [last_pymnt_amnt],
            'tot_cur_bal': [tot_cur_bal],
            'total_rev_hi_lim': [total_rev_hi_lim],
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