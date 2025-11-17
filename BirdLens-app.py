import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms
from data_utils import read_txt_to_list

# Streamlit page configuration
st.set_page_config(layout="centered", page_title="Bird Recognition With BirdLens",menu_items={
    'Report a bug': "https://github.com/mrkomoruyi/BirdLens/issues/new", 
    'About': "#### This is BirdLens. An AI bird recognition model. Upload or provide link to an image of a bird and find out it's name! :smile:"})

st.title("Bird Recognition With :blue[BirdLens]", anchor='title')
st.write("### :bird: Upload an image of any bird and find out its name!")
st.warning('BirdLens uses AI! Check for mistakes')

# Load the pre-trained image recognition model
model = torch.jit.load("BirdLens model scripted.pth", map_location='cpu')
model.eval()

# load classes
classes = read_txt_to_list('bird_species.txt')

upload_methods = ['Camera', 'Upload image', 'Use URL to image online']
upload_method = st.selectbox('Select an image input method', options=upload_methods, index=None, placeholder='e.g Camera to take photo of bird live')
uploaded_image = None
uploaded_url = None

if upload_method == upload_methods[0]:
    uploaded_image = st.camera_input('Snap the bird!')
elif upload_method == upload_methods[1]:
    uploaded_image = st.file_uploader("##### Upload an image :camera:", type=["jpg", "png", "jpeg"])
elif upload_method == upload_methods[2]:
    uploaded_url = st.text_input("##### Enter the URL to the image :link:", placeholder='e.g https://.../golden-eagle.jpg')

def get_input():
    if uploaded_url:
        if not uploaded_url.endswith(('.jpg', '.png', '.jpeg')):
            st.error('Provided link not in acceptable format. Link must end in one of: ***.jpg***, .***png*** or ***jpeg***')
            st.stop()
        with st.spinner('Fetching Image...'):
            request = requests.get(uploaded_url)
            image = Image.open(BytesIO(request.content)).convert("RGB")
    else:
        image = Image.open(uploaded_image).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale = True)])
    return transform(image).unsqueeze(0)

def get_prediction(input_tensor):
    # Get model prediction
    with torch.inference_mode():
        output = model(input_tensor).squeeze()
        prediction_probs, class_idxs = output.softmax(0).sort(descending=True)
        top3_class_idxs, top3_prediction_probs = class_idxs[:3], prediction_probs[:3]
        predicted_class, predicted_class_prob = top3_class_idxs[0].item(), top3_prediction_probs[0]
    st.write(f"###### This is a/an : ***{classes[predicted_class].title()}***")
    if predicted_class_prob >= 0.70:
        st.write(f"###### :green[***{int(predicted_class_prob*100)}%***] confidence")
    elif predicted_class_prob < 0.70:
        if predicted_class_prob >= 0.50:
            st.write(f"###### :orange[***{int(predicted_class_prob*100)}%***] confidence")
        else:
            st.write(f"###### :red[***{int(predicted_class_prob*100)}%***] confidence. Please verify as model could be wrong here!")
        st.write('It could also be one of these:')
        for idx in range(1, len(top3_class_idxs)):
            st.write(f'* {classes[top3_class_idxs[idx]].title()} - :red[***{int(top3_prediction_probs[idx]*100)}%***] confidence')

if uploaded_image or uploaded_url:
    input_tensor = get_input()
    get_prediction(input_tensor)



