import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ----- CLASSES DO CIFAR-10 -----
classes = ['avião', 'automóvel', 'pássaro', 'gato', 'cervo',
           'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']

# ----- TRANSFORMAÇÃO -----
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 é 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ----- MODELO (copie aqui sua classe Net) -----
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1) # observar que a saída é 64, mesmo argumento da nn.Linear
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 500) # a dimensão da saída é a mesma da entrada da próxima
        self.fc2 = nn.Linear(500, 10) # a dimensão de saída é o len(classes)
        self.dropout = nn.Dropout(.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))


        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = self.dropout(x)

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x

# ----- LOAD MODELO -----
modelo = Net()
modelo.load_state_dict(torch.load("modelo_final.pt", map_location=torch.device('cpu')))
modelo.eval()

# ----- INTERFACE -----
st.title("Classificador de Imagem - CIFAR-10")
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagem carregada", use_column_width=True)

    # Prepara imagem
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = modelo(img_tensor)
        pred = output.argmax(dim=1).item()
        probas = torch.softmax(output, dim=1)[0]

    st.markdown(f"### Classe prevista: **{classes[pred]}**")
    
    st.bar_chart(probas.numpy())
