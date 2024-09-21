
# **SMS Spam Detection using Machine Learning**

This project demonstrates how to build a machine learning model to classify SMS messages as **spam** or **legitimate** (ham) using natural language processing (NLP) techniques. The dataset used contains 5,574 SMS messages in English, provided by AFrame Technologies.

## **Table of Contents**
- [Objective](#objective)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Preprocessing Steps](#preprocessing-steps)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Contributions](#contributions)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## **Objective**
The goal of this project is to classify SMS messages as **spam** or **ham** (legitimate) using various machine learning techniques like **TF-IDF** vectorization and classifiers such as **Naive Bayes**, **Logistic Regression**, and **Support Vector Machines** (SVM).

---

## **Dataset**
The dataset, **SMS Spam Collection**, contains 5,574 SMS messages labeled as either spam or ham. It is available [here](https://drive.google.com/drive/folders/13bmD8C-0OvgXOE0g_ab4C3QzjzVYlWQj?usp=sharing), provided by **AFrame Technologies**.

---

## **Models Implemented**

### 1. **Decision Tree Classifier**
- **Accuracy:** 96.861%
- **Description:** The model was trained using **TF-IDF** vectorization to convert text data into numerical features. Decision Trees were employed to classify messages into spam or legitimate.

### 2. **Gaussian Naive Bayes Classifier**
- **Accuracy:** 87.085%
- **Description:** The model used **TF-IDF** vectorization for feature extraction and a Gaussian Naive Bayes classifier for prediction.

---

## **Preprocessing Steps**

1. **Text Cleaning:** Removed punctuation and special characters, and converted all text to lowercase.
2. **Tokenization:** Split the text into individual words.
3. **Stopword Removal:** Removed common English stopwords such as "the", "and", etc.
4. **Stemming:** Applied **Porter Stemmer** to reduce words to their base form.
5. **Vectorization:** Used **TF-IDF** (Term Frequency-Inverse Document Frequency) to transform text data into numerical features.

---

## **Requirements**
Make sure you have the following Python packages installed:
- **Python 3.x**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **NLTK**
- **Re**

Install the required libraries by running:
```bash
pip install -r requirements.txt
```

---

## **Usage**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HEMANTHSWAMY26/sms-spam-detection.git
   ```
   
2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Run the Jupyter Notebook**:
   Open `spam_sms_detection.ipynb` in Jupyter Notebook to explore the model-building process.
   ```bash
   jupyter notebook spam_sms_detection.ipynb
   ```

---

## **Results**

### Model Performance:
- **Decision Tree Classifier**: Achieved an accuracy of **96.861%**.
- **Gaussian Naive Bayes Classifier**: Achieved an accuracy of **87.085%**.

---

## **Contributions**
Pull requests are welcome! For significant changes, kindly open an issue first to discuss what you'd like to modify.

---

## **Acknowledgments**
- Special thanks to **AFrame Technologies** for providing the dataset.

---

## **Contact**
- **Name**: Amidepuram Hemanth Swamy
- **Email**: [hemanthswamy22@gmail.com](mailto:hemanthswamy22@gmail.com)
- **GitHub**: [HEMANTHSWAMY26](https://github.com/HEMANTHSWAMY26)
