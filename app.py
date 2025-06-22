import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import requests
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
        recon_x, mu, logvar = self.encoder(x) # Aquí tenías un error, el encoder devuelve mu, logvar
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

# --- Configuración del Modelo ---
LATENT_DIM = 20
MODEL_PATH = "vae_mnist_model.pt"

# --- Modelo de Carga (ahora carga desde PATH local) ---
@st.cache_resource
def load_vae_model(): # Renombrado para claridad
    try:
        model = VAE(latent_dim=LATENT_DIM)
        # Asegúrate de cargar el modelo en la CPU si Streamlit se ejecuta en CPU
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        st.success("Modelo VAE cargado exitosamente.")
        return model
    except FileNotFoundError:
        st.error(f"Error: El archivo del modelo '{MODEL_PATH}' no se encontró en el repositorio.")
        st.info("Asegúrate de haber subido 'vae_mnist_model.pt' al mismo directorio que 'app.py' en GitHub.")
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

vae_model = load_vae_model()


# --- Cargar y Preprocesar el Conjunto de Datos MNIST para el Clasificador ---
transform_mnist = transforms.Compose([transforms.ToTensor()])

@st.cache_data # Caching para el dataset
def load_mnist_data():
    # Descargar solo si no existe localmente (para Streamlit Cloud)
    return datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)

# --- Clasificador Simple para identificar el dígito de las imágenes generadas ---
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
def train_classifier(): 
    dataset = load_mnist_data()
    classifier = DigitClassifier()
    optimizer_clf = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    criterion_clf = nn.CrossEntropyLoss()

    clf_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    st.write("Entrenando clasificador auxiliar...")
    for epoch in range(1):
        for batch_idx, (data, target) in enumerate(clf_loader):
            optimizer_clf.zero_grad()
            output = classifier(data)
            loss = criterion_clf(output, target)
            loss.backward()
            optimizer_clf.step()
    st.success("Clasificador auxiliar entrenado.")
    return classifier

# Llamar la función de entrenamiento sin pasar el dataset directamente
classifier_model = train_classifier()


# --- Lógica de Generación de Imágenes (Adaptada para generar un dígito específico) ---
@st.cache_data
def get_latent_means_for_digits(vae_model, num_samples_per_digit=100):
    # Cargar el dataset AQUI dentro de la función cacheada
    dataset = load_mnist_data() # <-- LLAMAR LA FUNCIÓN CACHEADA DENTRO

    latent_means = {i: [] for i in range(10)}
    vae_model.eval()
    with torch.no_grad():
        for i, (image, label) in enumerate(dataset):
            if len(latent_means[label]) < num_samples_per_digit:
                image = image.unsqueeze(0) # Añadir dimensión de batch
                mu, _ = vae_model.encoder(image)
                latent_means[label].append(mu.squeeze().numpy())

            if all(len(v) >= num_samples_per_digit for v in latent_means.values()):
                break

    avg_latent_vectors = {digit: np.mean(latent_means[digit], axis=0)
                          for digit in range(10) if latent_means[digit]}
    return avg_latent_vectors

# Calcular los vectores latentes promedio para cada dígito
avg_latent_vectors = get_latent_means_for_digits(vae_model)


def generate_specific_digit_images(vae_model, target_digit, num_images=5, latent_dim=LATENT_DIM, device='cpu', avg_latent_vectors=None):
    vae_model.eval()
    generated_images = []

    with torch.no_grad():
        for _ in range(num_images):
            if target_digit in avg_latent_vectors:
                base_z = torch.tensor(avg_latent_vectors[target_digit], dtype=torch.float32).to(device)
                noise = torch.randn(latent_dim).to(device) * 0.5
                z = base_z + noise
            else:
                z = torch.randn(latent_dim).to(device)

            img = vae_model.decoder(z.unsqueeze(0)).cpu().squeeze().numpy()
            generated_images.append(img)
    return generated_images


# --- Interfaz de Usuario de Streamlit ---
st.set_page_config(layout="centered", page_title="Generador de Dígitos MNIST")

st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained VAE model.")

if vae_model is None:
    st.stop()

# Selección de dígito por el usuario
digit_to_generate = st.selectbox(
    "Choose a digit to generate (0-9):",
    options=list(range(10)),
    index=2
)

if st.button("Generate Images"):
    st.subheader(f"Generated images of digit {digit_to_generate}")

    generated_imgs = generate_specific_digit_images(
        vae_model,
        digit_to_generate,
        num_images=5,
        latent_dim=LATENT_DIM,
        device='cpu',
        avg_latent_vectors=avg_latent_vectors
    )

    cols = st.columns(5)

    for i, img_array in enumerate(generated_imgs):
        with cols[i]:
            img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
            st.image(img_pil, caption=f"Sample {i+1}", use_column_width=True)

st.markdown("---")
st.markdown("Disclaimer: Images are generated by a Variational Autoencoder (VAE) trained on the MNIST dataset. The quality depends on the training data and model complexity.")
