import streamlit as st
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from PIL import Image
import joblib
from pathlib import Path
import time

# Forzar JAX a usar CPU
jax.config.update('jax_platform_name', 'cpu')

# Configuración de la página
st.set_page_config(
    page_title="CIFAR-10 Clasificador 🖼️",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clases CIFAR-10 con emojis
CLASSES = {
    0: "✈️ Avión",
    1: "🚗 Automóvil",
    2: "🐦 Pájaro",
    3: "🐱 Gato",
    4: "🦌 Ciervo",
    5: "🐕 Perro",
    6: "🐸 Rana",
    7: "🐎 Caballo",
    8: "🚢 Barco",
    9: "🚛 Camión"
}

# Estilos CSS personalizados
st.markdown("""
<style>
    .main {
        padding: 1.5rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 8px;
        margin-top: 1rem;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .top-prediction {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #2196F3;
    }
    .progress-container {
        margin: 0.8rem 0;
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .error-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #ffebee;
        color: #c62828;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar .stMarkdown {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .preview-section {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class RobustCNN(nn.Module):
    """CNN robusta para clasificación de imágenes CIFAR-10."""
    @nn.compact
    def __call__(self, x, training=True):
        # Normalización de entrada
        x = (x - 0.5) * 2.0
        
        # Bloque 1
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(0.25, deterministic=not training)(x)
        
        # Bloque 2
        x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(0.25, deterministic=not training)(x)
        
        # Bloque 3
        x = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(0.25, deterministic=not training)(x)
        
        # Capas densas
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=512)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.5, deterministic=not training)(x)
        x = nn.Dense(features=10)(x)
        
        return x

@st.cache_resource(show_spinner="Cargando recursos...")
def load_model():
    """Carga el modelo pre-entrenado."""
    try:
        model_path = Path("models/robust_cifar10_model.joblib")
        if not model_path.exists():
            raise FileNotFoundError("Modelo no encontrado")
            
        model_state = joblib.load(model_path)
        model = RobustCNN()
        # Inicializar el modelo con una forma de entrada de ejemplo
        rng = jax.random.PRNGKey(0)
        input_shape = (1, 32, 32, 3)
        dummy_input = jnp.ones(input_shape)
        variables = model.init(rng, dummy_input, training=False)
        
        # Actualizar con los parámetros guardados
        params = model_state['params']
        batch_stats = model_state.get('batch_stats', {})
        
        return model, params, batch_stats
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None, None, None

def preprocess_image(image, target_size=(32, 32)):
    """Preprocesa una imagen para la predicción."""
    # Redimensionar
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convertir a array y normalizar
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Asegurar que tiene 3 canales
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    
    # Añadir dimensión de batch y convertir a JAX array
    return jnp.asarray(img_array[None, ...])

def predict(model, params, batch_stats, image):
    """Realiza la predicción."""
    variables = {'params': params}
    if batch_stats:
        variables['batch_stats'] = batch_stats
    
    # Realizar la predicción
    output = model.apply(variables, image, training=False, mutable=False)
    return jax.nn.softmax(output)

def show_prediction_results(probs, duration):
    """Muestra los resultados de la predicción de manera atractiva."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Predicción Principal")
        top_class = jnp.argmax(probs)
        confidence = float(probs[0][top_class])
        
        st.markdown(
            f"""
            <div class='prediction-box' style='background-color: {"#e3f2fd" if confidence > 0.5 else "#fff3e0"}'>
                <div class='top-prediction'>{CLASSES[int(top_class)]}</div>
                <div>Confianza: {confidence:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(f"⚡ Tiempo de predicción: {duration:.3f} segundos")
    
    with col2:
        st.markdown("### 📊 Todas las Predicciones")
        # Ordenar predicciones por probabilidad
        class_probs = [(CLASSES[i], float(p)) for i, p in enumerate(probs[0])]
        class_probs.sort(key=lambda x: x[1], reverse=True)
        
        for clase, prob in class_probs:
            st.markdown(
                f"""
                <div class='progress-container'>
                    <div>{clase}</div>
                    <div class="progress">
                        <div class="progress-bar" role="progressbar" 
                             style="width: {prob:.1%}; 
                                    background-color: {"#2196f3" if prob > 0.2 else "#90caf9"};">
                            {prob:.1%}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

def main():
    """Función principal de la aplicación."""
    # Cargar el modelo
    model, params, batch_stats = load_model()
    if not all([model, params, batch_stats]):
        st.error("No se pudo cargar el modelo. Por favor, verifica que el archivo del modelo existe.")
        return

    # Sidebar con información
    with st.sidebar:
        st.markdown("## 🤖 Clasificador CIFAR-10")
        st.markdown("""
        ### 📝 Instrucciones
        1. Sube una imagen usando el botón de carga
        2. Previsualiza la imagen
        3. Haz clic en 'Realizar Predicción'
        
        ### 🎯 Clases Disponibles
        """)
        for clase in CLASSES.values():
            st.markdown(f"- {clase}")

    # Contenedor principal
    st.markdown("# 🖼️ Clasificador de Imágenes CIFAR-10")
    st.markdown("#### Clasifica tus imágenes en 10 categorías diferentes usando Inteligencia Artificial")
    
    # Sección de carga de imagen
    with st.container():
        st.markdown("### 📤 Subir Imagen")
        uploaded_file = st.file_uploader(
            "Selecciona una imagen...",
            type=["jpg", "jpeg", "png"],
            help="Formatos soportados: JPG, JPEG, PNG"
        )

    # Sección de previsualización y predicción
    if uploaded_file is not None:
        try:
            # Cargar y mostrar la imagen
            image = Image.open(uploaded_file)
            
            # Crear dos columnas para la previsualización
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### 👁️ Previsualización")
                st.image(image, caption="Imagen cargada", width=300)  
                
                # Botón de predicción justo debajo de la imagen
                predict_button = st.button("🔍 Realizar Predicción")
            
            # Realizar predicción cuando se presione el botón
            if predict_button:
                with st.spinner("Procesando imagen..."):
                    # Preprocesar imagen
                    processed_image = preprocess_image(image)
                    
                    # Medir tiempo de predicción
                    start_time = time.time()
                    probs = predict(model, params, batch_stats, processed_image)
                    duration = time.time() - start_time
                    
                    # Mostrar resultados
                    show_prediction_results(probs, duration)

        except Exception as e:
            st.error(f"Error al procesar la imagen: {str(e)}")

if __name__ == "__main__":
    main()