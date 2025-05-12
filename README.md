# API

I have taken my API KEY from the rapid api's JSearch for listing job posts in my project.
You can visit https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch for the API KEY and its code snippets.


# Pre Traine ML Models and Vectorizers

model.pkl and tfidf_vectorizer.pkl are the older models trained on less data(less than 1000 dataset) and less efficient

combined_job_predict_model.zip( ML Model) and combined_tfidf_vectorizer.pkl are new models that are trained(more than 11000 dataset) and more eficient

resume_score_model11.pkl is a pretrained model for predicting the uploaded resume score (Ranged from 0 to 10)

# .env file

I used .env file to store all the sesitive keys and passwords to not to show directly in the code so that it will help to maintain security as well as helps to make chnages related to password, db names or keys in one place and will be refracted to the main code.

# Jupyter Notebook

Another Resume ooooooooooo.ipyn is the notebook where i trained job role prediction model.
Resume Job Role and Score Analysis.ipynb is for resume score model training and testing(do not consider job role prediction section).

# Datasets

combined_dataset_resume.csv is a dataset used for resume job role prediction and Enhanced_Resume_Dataset (1).csv is a dataset used for resume score prediction.
