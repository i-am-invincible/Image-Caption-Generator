"""Image caption generator by Jagrati Dhakar"""

import streamlit as st
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer
from itertools import cycle
import openai
from tqdm import tqdm
from PIL import Image
import torch
import os

# object creation model, tokenizer and processor from HuggingFace

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base") 
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base") 
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")

#Setting for the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Getting the key from env
openai.api_key = os.getenv("api_key")
openai_model ="text-davinci-003"

def caption_generator(description):
  caption_prompt = ('''Please generate 4 captions for the following image: '''+description+'''.  The captions should be fun, trendy and creative.
  Captions:
  1.
  2.
  3.
  4.''')
  
  try:
    # Image Caption generation
    response = openai.Completion.create(
      engine = openai_model,
      prompt  = caption_prompt, 
      max_tokens = (175*4),
      n=1,
      stop = None,
      temperature =0.7,
    )
    #caption = response.choices[0].text.strip().split("\n")
    caption = [choice.text.strip() for choice in response.choices]
    return caption
  except openai.error.RateLimitError as e:
        st.error(f"OpenAI API rate limit exceeded: {e}")
        return None

def prediction(img_list):
  max_length = 16
  num_beams = 4
  gen_kwargs = {"max_length": max_length,"num_beams": num_beams}
  img = []

  for image in tqdm(img_list):
    i_image = Image.open(image) # storing of image
    st.image(i_image, width = 200)  # Display of Image
    if i_image.mode != "RGB": #check if the image is in RGB mode 
      i_image = i_image.convert(mode="RGB")
    
    img.append(i_image)   #Adding to the list
    
  # Image data to pixel values
  pixel_val = processor(images=img, return_tensors="pt").pixel_values
  pixel_val = pixel_val.to(device)

  # Using model to generate output from the pixel values of Image
  output = model.generate(pixel_val, **gen_kwargs)

  # To convert output to text
  predict = tokenizer.batch_decode(output, skip_special_tokens=True) 
  predict = [pred.strip() for pred in predict]

  return predict

def sample():
  # Testcase Images
  sp_images = {'Sample 1':'img1.jpeg','Sample 2':'img2.jpeg', 'Sample 3':'img3.jpeg', 'Sample 4':'Image4.jpg'}
  
  colms = cycle(st.columns(4)) # No. of Columns
  
  for sp in sp_images.values(): # To display the sample images
    next(colms).image(sp, width=150)
    
  for i, sp in enumerate (sp_images.values()): # loop to generate caption and hashtags for the sample images
    
    if next(colms).button("Generate", key=i): # Prediction is called only the selected image
      
      description = prediction([sp])
      st.subheader("About the image:")
      st.write(description[0])

      st.subheader("Captions for this image are:")
      captions = caption_generator(description[0]) # Fuction call to generate caption 
      for caption in captions:   # present Captions
        st.write(caption)

def upload():
  
  # form uploader inside tab
  with st.form("Uploader"):
    # Image input
    image = st.file_uploader("Upload Image", accept_multiple_files=True, type=["jpg","png","jpeg"])
    # Generate button
    submit = st.form_submit_button("Generate Caption")

    if submit: # submit condition
      description = prediction(image)

      st.subheader("About the image") 
      for i, caption in enumerate(description): 
        st.write(caption)

      st.subheader("Captions for this image are:")
      captions = caption_generator(description[0])   #function call to generate caption 
      for caption in captions:  # Present Caption 
        st.write(caption)

def main():
  # title on the tab
  st.set_page_config(page_title="Image Caption Generator by Jagrati Dhakar")
  # Title the page
  st.title("Image Caption Generator")
  # Sub-title of the page
  #st.subheader("Jagrati Dhakar")

  # Tabs on the page
  tab1, tab2 = st.tabs(["Upload Image","Testcase Images"])

  # selection of Tabs
  with tab1:  # Sample images tab
    upload()

  with tab2:  # Upload images tab
    sample()

if __name__ == '__main__':
  main()

