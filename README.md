# 🖼️ Clasificador de Imágenes CIFAR-10 con JAX y Streamlit

### 📝 Archivo de Documentación del Ejercicio de Investigación

El archivo **`JAX_investigacion_German_GE.ipynb`** es el cuaderno Jupyter donde se documenta todo el proceso del ejercicio de investigación relacionado con JAX.

[![Aplicación en vivo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cifar-10-clasificador.streamlit.app/)
[![Documentación en Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rv-4nDLlBcMR7qC8RIs3P_tuWeAGXVZt?usp=sharing)

![CIFAR-10](img/cifar-10.jpg)

Este proyecto implementa un clasificador de imágenes robusto utilizando JAX, Flax y Streamlit. El modelo está entrenado en el dataset CIFAR-10 y puede clasificar imágenes en 10 categorías diferentes, con una interfaz web moderna y fácil de usar.

## 🌟 Características de la Aplicación Web

- **Interfaz Moderna**: Diseño limpio y responsivo con tema personalizado
- **Barra Lateral Informativa**: 
  - Instrucciones paso a paso
  - Lista de categorías disponibles
  - Guía de uso rápido
- **Carga de Imágenes**:
  - Soporte para formatos JPG, JPEG y PNG
  - Previsualización instantánea
  - Redimensionamiento automático a 32x32 píxeles
- **Predicciones en Tiempo Real**:
  - Visualización de resultados con porcentajes de confianza
  - Diseño atractivo para las predicciones
  - Tiempo de procesamiento mostrado
- **Manejo de Errores**:
  - Mensajes de error informativos
  - Validación de formatos de imagen
  - Comprobación de modelo pre-entrenado

## 🎯 Categorías de Clasificación

El modelo puede clasificar imágenes en las siguientes categorías:
- ✈️ Avión
- 🚗 Automóvil
- 🐦 Pájaro
- 🐱 Gato
- 🦌 Ciervo
- 🐕 Perro
- 🐸 Rana
- 🐎 Caballo
- 🚢 Barco
- 🚛 Camión

## 🛠️ Tecnologías Utilizadas

- **JAX y Flax**: 
  - Framework de computación numérica
  - Implementación eficiente de CNN
  - Optimización con CPU
- **Streamlit**: 
  - Framework web interactivo
  - Componentes UI personalizados
  - Caché de recursos para mejor rendimiento
- **Procesamiento de Imágenes**:
  - PIL para manipulación de imágenes
  - NumPy para procesamiento numérico
  - Preprocesamiento automático

## 📋 Requisitos del Sistema

```txt
Python 3.11 o superior
RAM: 4GB mínimo recomendado
CPU: Compatible con operaciones vectoriales
Espacio en disco: ~500MB (incluyendo modelo)
```

## 🚀 Instalación y Uso

1. **Clonar el repositorio**
   ```bash
   git clone <url-del-repositorio>
   cd ejemplo_jax
   ```

2. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

3. **Entrenar el modelo** (opcional)
   ```bash
   python train_robust_model.py
   ```

4. **Ejecutar la aplicación**
   ```bash
   streamlit run streamlit_app.py
   ```

### 🐳 Usando Docker

```bash
docker-compose up --build
```

La aplicación estará disponible en `http://localhost:8501`

## 📁 Estructura del Proyecto

```
ejemplo_jax/
├── data/                     # Datos de entrenamiento y prueba
│   ├── cifar10_test.joblib
│   └── cifar10_train.joblib
├── models/                   # Modelos pre-entrenados
│   └── robust_cifar10_model.joblib
├── streamlit_app.py         # Aplicación web principal
├── train_robust_model.py    # Script de entrenamiento
├── requirements.txt         # Dependencias
├── Dockerfile              # Configuración Docker
└── docker-compose.yml      # Configuración Docker Compose
```

## 🧮 Detalles Técnicos

### Arquitectura del Modelo
- CNN con 3 bloques convolucionales
- Capas de normalización por lotes
- Dropout para regularización
- Capa densa final de 10 clases

### Preprocesamiento de Imágenes
- Redimensionamiento a 32x32 píxeles
- Normalización de valores de píxeles
- Conversión automática a 3 canales RGB
- Manejo de imágenes en escala de grises y RGBA

### Optimizaciones
- Caché de modelo con @st_cache_resource
- Procesamiento eficiente con JAX
- Interfaz responsiva y optimizada

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Haz fork del repositorio
2. Crea una rama para tu feature
3. Envía un pull request

## 📝 Licencia

Este proyecto está bajo la licencia MIT.

## ✨ Agradecimientos

- Equipo de JAX y Flax
- Comunidad de Streamlit
- Creadores del dataset CIFAR-10
