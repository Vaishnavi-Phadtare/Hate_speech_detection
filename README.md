Hate Speech Detection using Machine Learning

Overview:
Built a text classification model using machine learning
Preprocessed text using tokenization, stopword removal, and stemming
Evaluated the model using metrics like accuracy, precision, recall, and F1-score
Developed an interactive web app using Streamlit
Organized into modular scripts: app.py, preprocessing.py, and train_model.py

Project Structure:
Hate_speech_detection/
 app.py                # Streamlit web app
 preprocessing.py      # Text preprocessing functions
 train_model.py        # Model training and evaluation script
 Ethos_Dataset_Binary.csv  # Dataset used for training
 requirements.txt      # Project dependencies
 README.md             # Project description

How to Run Locally
Clone the repository:
Install dependencies:

Run the app:
streamlit run app.py

Model Evaluation (Sample Results)
Metric	Train Score	Test Score
Accuracy	98%	89%
Precision	99%	90%
Recall	97%	87%
F1-score	98%	89%

Dataset:
Name: Ethos Dataset

Type: Binary classification (Hate / Not Hate)

Features
Modular and easy-to-understand code
Custom preprocessing pipeline using nltk
Model trained using LogisticRegression and TfidfVectorizer
Streamlit interface to interactively test the model

Future Improvements:
Experiment with deep learning models (e.g., BERT)
Improve UI and add charts for model performance

📎 License
This project is for educational purposes.
