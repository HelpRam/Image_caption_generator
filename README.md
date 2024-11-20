# Image Caption Generator

This repository contains the implementation of an Image Caption Generator that combines computer vision and natural language processing techniques. The project was developed using the **Flickr8k dataset** along with **150 self-gathered real images with captions** for fine-tuning. It leverages **VGG16** for feature extraction and **LSTM** for generating captions, with a user-friendly interface deployed using Streamlit.

---

## 📑 **Table of Contents**
1. Overview  
2. Dataset  
3. Model Architecture  
4. Preprocessing Steps  
5. Results and Challenges  
6. Deployment  
7. How to Use  

---

## 🛠 **Overview**
The Image Caption Generator generates descriptive captions for images by combining features extracted from images with sequence modeling for text generation. This project explores:
- Fine-tuning with a combination of standard datasets and real-world data.
- Efficient feature extraction and text generation using deep learning.

---

## 📂 **Dataset**
1. **Flickr8k Dataset**:  
   - Contains 8,000 images, each paired with five descriptive captions.
2. **Self-Gathered Dataset**:  
   - 150 real-world images collected and manually annotated with captions.  
   - Added for fine-tuning to improve domain-specific accuracy.

---

### **Model Architecture**

The Image Caption Generator integrates **computer vision** and **natural language processing** by combining a **Convolutional Neural Network (CNN)** for feature extraction with a **Recurrent Neural Network (RNN)** for sequence modeling. Below is a detailed breakdown:

#### **1. Feature Extraction (VGG16)**
- **Pretrained Model**:  
  The **VGG16** model, trained on the ImageNet dataset, is used for extracting high-level visual features from images.
  
- **Modification**:  
  - Removed the fully connected layers to focus on feature maps from convolutional layers.
  - These feature maps represent the critical visual elements of the input image.

- **Output**:  
  A fixed-length feature vector for each image, capturing its visual essence.

#### **2. Caption Generation (LSTM)**
- **Input**:  
  The extracted feature vector from VGG16 is used as input to the language model.

- **LSTM Layer**:  
  - Handles the sequential generation of captions.
  - Models the dependencies between words in a sentence, ensuring grammatically correct and contextually relevant captions.

- **Embedding Layer**:  
  - Converts tokenized words into dense vector representations.
  - Facilitates better learning of relationships between words.

- **Decoder**:  
  - Combines the image features and the sequential text features to generate captions word-by-word.

- **Output**:  
  A sequence of words forming a complete caption.

---

### **Preprocessing Steps**

Effective preprocessing ensures clean, structured data for training the model. Here's what was done for both images and captions:

#### **1. Image Preprocessing**
- **Resizing**:  
  Images were resized to a fixed shape (e.g., 224x224 pixels) to match the input requirements of the VGG16 model.
  
- **Normalization**:  
  Pixel values were scaled to a range of [0, 1] or [-1, 1] to accelerate training.

- **Feature Extraction**:  
  - Used the VGG16 model to convert raw images into feature vectors.
  - Saved these features as input for the caption generation model.

#### **2. Text Preprocessing**
- **Cleaning Captions**:  
  - Removed punctuation, special characters, and extra spaces.  
  - Converted all text to lowercase to ensure consistency.

- **Tokenization**:  
  - Split captions into individual words or tokens.  
  - Created a vocabulary of unique words from the dataset.

- **Handling Unknown Words**:  
  - Words not in the vocabulary were replaced with a special `<UNK>` token.

- **Sequence Padding**:  
  - Captions were padded with zeros to match a fixed sequence length.
  - Ensured uniform input sizes for the LSTM model.

#### **3. Data Augmentation (Optional)**
To enhance model performance, you can use techniques like:
- Rotating or flipping images.
- Synonym substitution in captions to diversify training data.

---

## 📊 **Results and Challenges**
- The model achieved reasonable performance during training but faced limitations due to:
  - Insufficient computational resources for extended training.
  - The relatively small size of the self-gathered dataset.
- As a result, accuracy was not optimal but demonstrated the potential of the approach.

---

## 🌐 **Deployment**
The project is deployed using **Streamlit**, offering a simple interface where users can:
1. Upload an image.
2. Receive a generated caption in real time.  

Run the application locally:
```bash
streamlit run app.py
```

---

## 💡 **How to Use**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/HelpRam/Image_caption_generator.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

---

## 🙌 **Acknowledgements**
- **Flickr8k Dataset** for providing a diverse set of images and captions.
- **Pretrained VGG16 Model** for reliable feature extraction.
- The open-source community for tools enabling this project.

Contributions and suggestions are welcome! 😊
