import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Afficher le logo en haut de la barre latérale
st.sidebar.image("images/ehtp_logo.png", use_container_width=True)
st.sidebar.write("---------------")

# Afficher l'objet du projet dans la barre latérale
st.markdown(
              "<style> .center { display: flex; justify-content: center; } </style>", 
              unsafe_allow_html=True
            )
st.sidebar.markdown("<h1 style='text-align: center;'>MSDE6 : Deep Learning Project</h1>", unsafe_allow_html=True)
st.sidebar.image("images/Cat vs Dog.jpg")
st.sidebar.write("---------------")

# Charger l'image à partir de la barre latérale
image_uploaded = st.sidebar.file_uploader("Upload your picture :", type=["jpg", "png", "jpeg"])

# Afficher l'entête de la page principale
st.markdown("<h1 style='text-align: center;'>Cat or Dog ?</h1>", unsafe_allow_html=True)
st.write("---------------")

# Charger le modèle de classification sauvegardé
model = load_model('Cat_or_Dog_Model.keras')

# Créer un ImageDataGenerator pour la normalisation des pixels 
image_gen = ImageDataGenerator( rescale=1/255 )  
                            
# Fonction pour le pré-traitement de l'image avant la prédiction
def image_preprocess(image):
    image = image.resize((360, 360))  # Redimensionner l'image
    image_array = np.array(image, dtype=np.float32)  # Convertir l'image en array
    image_array = np.expand_dims(image_array, axis=0)  # Ajouter une dimension pour simuler un batch de taille 1
    image_array = image_gen.standardize(image_array)  # Normaliser l'image
    return image_array

# Fonction de prédiction
def image_predict(image):
    image_input = image_preprocess(image) # Pré-traiter l'image
    prediction = model.predict(image_input)  # Faire la prédiction
    return prediction

# Si l'image est chargée correctement
if image_uploaded is not None:
    
    # Afficher l'image chargée
    image = Image.open(image_uploaded)
    image_resized = image.resize((500, 500))
    col1, col2, col3 = st.columns([1, 5, 1])  
    with col2:
        st.image(image_resized, use_container_width=False)
        
    # Prédire la classe de l'image
    prediction = image_predict(image)
    class_name = 'Dog' if prediction > 0.5 else 'Cat'
    
    # Afficher la Confiance
    if class_name == 'Cat':
        confiance = round((1 - prediction[0][0])*100)
    else : 
        confiance = round((prediction[0][0])*100)

    st.markdown("""
                    <style>
                        .classe-text {
                                        text-align: center;
                                        color: green;
                                        font-weight: bold;
                                        font-size: 60px;
                                    }
                    </style>
                """, unsafe_allow_html=True)
    st.markdown(f'<div class="classe-text">{class_name}</div>', unsafe_allow_html=True)
    
    st.markdown("""
                    <style>
                        .confiance-text {
                                            text-align: center;
                                            color: red;
                                            font-weight: bold;
                                            font-size: 30px;
                                        }
                    </style>
                """, unsafe_allow_html=True)
    st.markdown(f'<div class="confiance-text">( {confiance}% )</div>', unsafe_allow_html=True) 
        

 
