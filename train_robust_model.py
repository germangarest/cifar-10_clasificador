import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from pathlib import Path
import joblib
import os

# Forzar JAX a usar CPU
jax.config.update('jax_platform_name', 'cpu')

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

class TrainState(train_state.TrainState):
    batch_stats: dict

def create_train_state(rng, model, learning_rate, momentum):
    """Crea el estado inicial del entrenamiento."""
    input_shape = (1, 32, 32, 3)
    
    # Separar PRNG para inicialización y dropout
    rng, dropout_rng = jax.random.split(rng)
    
    # Crear variables para el modelo con PRNG para dropout
    variables = model.init(
        {'params': rng, 'dropout': dropout_rng}, 
        jnp.ones(input_shape), 
        training=True
    )
    
    # Extraer parámetros y batch_stats
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})
    
    # Optimizador con momentum y learning rate scheduling
    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=1000,
        decay_rate=0.9
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(learning_rate=schedule, weight_decay=0.0001)  # AdamW con weight decay
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats  # Añadir batch_stats al estado
    )

@jax.jit
def train_step(state, batch, rng):
    """Realiza un paso de entrenamiento."""
    images, labels = batch
    
    # Separar PRNG para dropout
    rng, dropout_rng = jax.random.split(rng)
    
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        logits, new_model_state = state.apply_fn(
            variables,
            images,
            training=True,
            rngs={'dropout': dropout_rng},
            mutable=['batch_stats']
        )
        one_hot = jax.nn.one_hot(labels, 10)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        return loss, (logits, new_model_state)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_model_state)), grads = grad_fn(state.params)
    
    # Actualizar estado con nuevos parámetros y batch_stats
    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state['batch_stats']
    )
    
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {'loss': loss, 'accuracy': accuracy}
    
    return state, metrics, rng

def save_dataset():
    """Descarga y guarda el dataset CIFAR-10."""
    print("\nDescargando y preparando dataset CIFAR-10...")
    
    # Crear directorio para datos
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Descargar y preparar el dataset
    # Usar una cantidad moderada del dataset
    train_ds = tfds.load('cifar10', split='train[:15000]', as_supervised=True)  # 15k imágenes
    test_ds = tfds.load('cifar10', split='test[:3000]', as_supervised=True)    # 3k imágenes
    
    # Convertir a numpy arrays
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    print("Procesando conjunto de entrenamiento...")
    for img, label in train_ds:
        train_images.append(img.numpy())
        train_labels.append(label.numpy())
    
    print("Procesando conjunto de prueba...")
    for img, label in test_ds:
        test_images.append(img.numpy())
        test_labels.append(label.numpy())
    
    # Convertir a arrays numpy
    train_images = np.array(train_images, dtype=np.float32) / 255.0
    train_labels = np.array(train_labels, dtype=np.int32)
    test_images = np.array(test_images, dtype=np.float32) / 255.0
    test_labels = np.array(test_labels, dtype=np.int32)
    
    # Guardar datasets
    print("Guardando datasets...")
    joblib.dump((train_images, train_labels), data_dir / "cifar10_train.joblib")
    joblib.dump((test_images, test_labels), data_dir / "cifar10_test.joblib")
    
    return (train_images, train_labels), (test_images, test_labels)

def main():
    print("Iniciando preparación del modelo robusto...")
    
    # Crear directorios necesarios
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Descargar y guardar dataset
    (train_images, train_labels), (test_images, test_labels) = save_dataset()
    
    # Inicializar modelo y estado de entrenamiento
    print("\nInicializando modelo...")
    model = RobustCNN()
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    # Hiperparámetros balanceados
    learning_rate = 0.002  # Learning rate moderado
    momentum = 0.9
    batch_size = 96      # Batch size moderado
    num_epochs = 10      # Épocas moderadas
    
    # Crear estado inicial
    state = create_train_state(init_rng, model, learning_rate, momentum)
    
    # Entrenamiento
    num_steps_per_epoch = len(train_images) // batch_size
    
    print("\nIniciando entrenamiento balanceado...")
    print(f"Dataset: {len(train_images)} imágenes de entrenamiento, {len(test_images)} de test")
    print(f"Configuración: {num_epochs} épocas, batch size {batch_size}, learning rate {learning_rate}")
    
    for epoch in range(num_epochs):
        # Shuffle de los datos
        perm = jax.random.permutation(rng, len(train_images))
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        
        # Métricas de la época
        epoch_loss = []
        epoch_accuracy = []
        
        for step in range(num_steps_per_epoch):
            rng, step_rng = jax.random.split(rng)
            
            # Preparar batch
            start_idx = step * batch_size
            end_idx = start_idx + batch_size
            batch = (train_images[start_idx:end_idx], train_labels[start_idx:end_idx])
            
            # Paso de entrenamiento
            state, metrics, rng = train_step(state, batch, step_rng)
            epoch_loss.append(metrics['loss'])
            epoch_accuracy.append(metrics['accuracy'])
            
            # Mostrar progreso cada 10 pasos
            if (step + 1) % 10 == 0:
                print(f"Época {epoch + 1}, Paso {step + 1}/{num_steps_per_epoch}", end="\r")
        
        # Mostrar métricas de la época
        avg_loss = np.mean(epoch_loss)
        avg_accuracy = np.mean(epoch_accuracy)
        print(f"\nÉpoca {epoch + 1}/{num_epochs}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.4f}")
    
    # Evaluar en conjunto de prueba
    print("\nEvaluando en conjunto de prueba...")
    test_batch_size = 100
    test_steps = len(test_images) // test_batch_size
    test_accuracy = []
    
    for step in range(test_steps):
        start_idx = step * test_batch_size
        end_idx = start_idx + test_batch_size
        test_batch = (test_images[start_idx:end_idx], test_labels[start_idx:end_idx])
        
        # Predicción
        logits = state.apply_fn(
            {'params': state.params, 'batch_stats': state.batch_stats},
            test_batch[0],
            training=False,
            mutable=False
        )
        predictions = jnp.argmax(logits, axis=1)
        accuracy = jnp.mean(predictions == test_batch[1])
        test_accuracy.append(accuracy)
    
    final_accuracy = np.mean(test_accuracy)
    print(f"\nPrecisión en test: {final_accuracy:.2%}")
    
    # Guardar modelo entrenado
    print("\nGuardando modelo entrenado...")
    model_state = {
        'params': state.params,
        'batch_stats': state.batch_stats,
        'model_config': {
            'name': 'RobustCNN',
            'input_shape': (32, 32, 3),
            'num_classes': 10
        }
    }
    
    joblib.dump(model_state, model_dir / "robust_cifar10_model.joblib")
    print("\n¡Entrenamiento completado! El modelo y los datos han sido guardados.")

if __name__ == "__main__":
    main()
