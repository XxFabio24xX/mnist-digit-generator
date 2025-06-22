import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import requests # Para descargar el modelo de Google Drive
import io

# --- Definición del Modelo VAE (DEBE SER IDÉNTICA A LA DE ENTRENAMIENTO) ---
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 7 * 7)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

# --- Configuración del Modelo ---
LATENT_DIM = 20 # Debe coincidir con el entrenamiento
MODEL_PATH = "vae_mnist_model.pt" # Nombre del archivo del modelo

# Descargar el modelo de Google Drive
# IMPORTANTE: Necesitarás obtener un enlace de descarga directa de Google Drive.
# Sube tu archivo 'vae_mnist_model.pt' a Google Drive, haz clic derecho -> 'Compartir' -> 'Cambiar a Cualquier persona con el enlace' -> 'Copiador enlace'.
# Luego, transforma este enlace en un enlace de descarga directa.
# Por ejemplo, si tu enlace es: https://drive.google.com/file/d/ABCD123_XYZ/view?usp=sharing
# El enlace de descarga directa sería: https://drive.google.com/uc?id=ABCD123_XYZ
GOOGLE_DRIVE_DOWNLOAD_URL = "YOUR_GOOGLE_DRIVE_DIRECT_DOWNLOAD_LINK_HERE" # ¡ACTUALIZA ESTO!

@st.cache_resource # Caching para evitar recargar el modelo en cada interacción
def load_model():
    try:
        st.write("Descargando modelo del Drive...")
        response = requests.get(GOOGLE_DRIVE_DOWNLOAD_URL)
        response.raise_for_status() # Lanza un error para códigos de estado HTTP erróneos
        model_data = io.BytesIO(response.content)

        model = VAE(latent_dim=LATENT_DIM)
        # Asegúrate de cargar el modelo en la CPU si Streamlit se ejecuta en CPU
        model.load_state_dict(torch.load(model_data, map_location=torch.device('cpu')))
        model.eval()
        st.success("Modelo cargado exitosamente.")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.info("Asegúrate de que el enlace de Google Drive sea de descarga directa y público.")
        return None

vae_model = load_model()

# --- Cargar y Preprocesar el Conjunto de Datos MNIST para el Clasificador ---
# Necesitamos imágenes reales de MNIST para el clasificador (para encontrar puntos latentes para dígitos específicos)
transform_mnist = transforms.Compose([transforms.ToTensor()])

# Descargar solo si no existe localmente (para Streamlit Cloud)
@st.cache_data # Caching para el dataset
def load_mnist_data():
    return datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)

mnist_dataset = load_mnist_data()

# --- Clasificador Simple para identificar el dígito de las imágenes generadas ---
# Este clasificador será usado para mapear un dígito deseado a un punto latente "representativo"
# Entrenar un clasificador simple solo para obtener el espacio latente de dígitos específicos.
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.fc(x)
        return x

@st.cache_resource # Caching para el clasificador
def train_classifier(dataset):
    classifier = DigitClassifier()
    optimizer_clf = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    criterion_clf = nn.CrossEntropyLoss()

    clf_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    st.write("Entrenando clasificador auxiliar...")
    for epoch in range(1): # Entrenar por 1 época es suficiente para nuestro propósito
        for batch_idx, (data, target) in enumerate(clf_loader):
            optimizer_clf.zero_grad()
            output = classifier(data)
            loss = criterion_clf(output, target)
            loss.backward()
            optimizer_clf.step()
    st.success("Clasificador auxiliar entrenado.")
    return classifier

# Entrenar un clasificador auxiliar para mapear el espacio latente a dígitos
classifier_model = train_classifier(mnist_dataset)

# --- Lógica de Generación de Imágenes (Adaptada para generar un dígito específico) ---
# Dado que nuestro VAE es incondicional, generaremos 5 puntos latentes aleatorios,
# luego los pasaremos por el clasificador para asegurarnos de que la mayoría sean el dígito deseado.
# O, una mejor estrategia: encontrar puntos latentes de imágenes *reales* del dígito deseado
# y luego añadir ruido a esos puntos latentes para la generación.
@st.cache_data # Caching para los puntos latentes promedio
def get_latent_means_for_digits(vae_model, dataset, num_samples_per_digit=100):
    latent_means = {i: [] for i in range(10)}
    vae_model.eval()
    with torch.no_grad():
        for i, (image, label) in enumerate(dataset):
            if len(latent_means[label]) < num_samples_per_digit:
                image = image.unsqueeze(0) # Añadir dimensión de batch
                mu, _ = vae_model.encoder(image)
                latent_means[label].append(mu.squeeze().numpy())

            # Check if all digits have enough samples
            if all(len(v) >= num_samples_per_digit for v in latent_means.values()):
                break

    avg_latent_vectors = {digit: np.mean(latent_means[digit], axis=0) 
                          for digit in range(10) if latent_means[digit]}
    return avg_latent_vectors

# Calcular los vectores latentes promedio para cada dígito
avg_latent_vectors = get_latent_means_for_digits(vae_model, mnist_dataset)


def generate_specific_digit_images(vae_model, target_digit, num_images=5, latent_dim=LATENT_DIM, device='cpu', avg_latent_vectors=None):
    vae_model.eval()
    generated_images = []

    with torch.no_grad():
        for _ in range(num_images):
            if target_digit in avg_latent_vectors:
                # Usar el vector latente promedio para el dígito y añadir ruido
                base_z = torch.tensor(avg_latent_vectors[target_digit], dtype=torch.float32).to(device)
                # Añadir ruido aleatorio para diversidad
                noise = torch.randn(latent_dim).to(device) * 0.5 # Ajusta la magnitud del ruido
                z = base_z + noise
            else:
                # Si no hay vector promedio, generar desde ruido puro (menos control)
                z = torch.randn(latent_dim).to(device)

            img = vae_model.decoder(z.unsqueeze(0)).cpu().squeeze().numpy()
            generated_images.append(img)
    return generated_images


# --- Interfaz de Usuario de Streamlit ---
st.set_page_config(layout="centered", page_title="Generador de Dígitos MNIST")

st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained VAE model.")

if vae_model is None:
    st.stop() # Detener la ejecución si el modelo no se cargó

# Selección de dígito por el usuario
digit_to_generate = st.selectbox(
    "Choose a digit to generate (0-9):",
    options=list(range(10)),
    index=2 # Valor por defecto
)

if st.button("Generate Images"):
    st.subheader(f"Generated images of digit {digit_to_generate}")

    # Generar las imágenes
    generated_imgs = generate_specific_digit_images(
        vae_model, 
        digit_to_generate, 
        num_images=5, 
        latent_dim=LATENT_DIM, 
        device='cpu', # Streamlit Cloud suele usar CPU
        avg_latent_vectors=avg_latent_vectors
    )

    cols = st.columns(5) # 5 columnas para 5 imágenes

    for i, img_array in enumerate(generated_imgs):
        with cols[i]:
            # Convertir array NumPy a imagen PIL para visualización
            img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
            st.image(img_pil, caption=f"Sample {i+1}", use_column_width=True)

st.markdown("---")
st.markdown("Disclaimer: Images are generated by a Variational Autoencoder (VAE) trained on the MNIST dataset. The quality depends on the training data and model complexity.")
