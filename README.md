## Environment Setup and Data Download
Unfortunately due to this being a Kaggle competition we cannot download the dataset directly from the competition page, https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data, I was able to find a reupload of the dataset here, https://www.kaggle.com/datasets/mohamedsameh0410/jane-street-dataset. Download the dataset locally and use SSH to rsync the dataset into the SDSC server.
![image](https://github.com/user-attachments/assets/859f856e-ac39-4b72-9c38-b3f96fe68989)
![image](https://github.com/user-attachments/assets/4621119f-e27b-4fd8-a889-12f754ea9594)

## Data Exploration and Analysis
The dataset that I am using is a real time market forecasting dataset from Janestreet. The dataset consists of a set of timeseries data with 79 features and 9 responders, all anonymized. The goal is to accurately predict the responder variables based off of the features. One of the difficulties that I ran into were the anonymized features, this led me to use standard deviation as the main method to decide on which features are the most valuable. The columns are date, time and symbol ids, along with weight, features 0-78, and responder 0-8. The data files included are the train parquet consisting of 47127338 observations. The lags parquet contains the responders offset by one date id, this shows the value of the responders at a later point in time. The weights features and responders are all flot values generally centered around 0.
The top responders with the highest standard deviation have no missing data, these are some of the features that will be focused on in the model training step.

![image](https://github.com/user-attachments/assets/119cb104-6085-4e2f-964d-2cf8efd10de6)


## Data Preprocessing
I will be addressing the missing data by using mean/median imputation because all of the features and responders are floats. I will also be using a standard scaler to standardize all of the features used in model training. My feature selection will focus on the top 5 responders according to their standard deviation, subsequent model training will focus on the less variable responders. 

## Link to Notebook
https://github.com/brocchio/DSC-232R-Final-Project/blob/Milestone2/Milestone_2_Explo.ipynb
