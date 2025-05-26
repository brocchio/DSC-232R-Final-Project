## Environment Setup and Data Download
Unfortunately due to this being a Kaggle competition we cannot download the dataset directly from the competition page, https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data, I was able to find a reupload of the dataset here, https://www.kaggle.com/datasets/mohamedsameh0410/jane-street-dataset. Download the dataset locally and use SSH to rsync the dataset into the SDSC server.
![image](https://github.com/user-attachments/assets/859f856e-ac39-4b72-9c38-b3f96fe68989)
![image](https://github.com/user-attachments/assets/4621119f-e27b-4fd8-a889-12f754ea9594)

## Data Exploration and Analysis
The dataset that I am using is a real time market forecasting dataset from Janestreet. The dataset consists of a set of timeseries data with 79 features and 9 responders, all anonymized. The goal is to accurately predict the responder variables based off of the features. One of the difficulties that I ran into were the anonymized features, this led me to use standard deviation as the main method to decide on which features are the most valuable. The columns are date, time and symbol ids, along with weight, features 0-78, and responder 0-8. The data files included are the train parquet consisting of 47127338 observations. The lags parquet contains the responders offset by one date id, this shows the value of the responders at a later point in time. The weights features and responders are all flot values generally centered around 0.
The top responders with the highest standard deviation have no missing data, these are some of the features that will be focused on in the model training step.

![image](https://github.com/user-attachments/assets/119cb104-6085-4e2f-964d-2cf8efd10de6)


## Data Preprocessing
Built a preprocessing pipeline to impute NaNs and infinity values with the median, vectorize the feature columns, and apply a standard scaler to each of the features. This is all done in my pipeline cell that I created at the beginning of my milestone3 notebook. I also chose to split the data into a test training and validation set based on the date_ids, this is to prevent the model from peaking into the future when making its predictions, the training set is based on the first 1350 date_ids, the validation set is from 1351 to 1500, and the test set is any date_id greater than 1500. This puts around 80% of the data in the training set, ~9% in the validation set, and ~12% in the test set. I have not implemented the lags parquet in the training yet as I am still aiming to get a strong baseline. 

## First Models
I chose a linear regression model to get a baseline knowing that it would perform poorly due to the non-linear correlations in the data noticed during my data-exploration steps. The settings I used for the linear regression model included 100 max iterations, low regulatization at 0.01. 

Linear‑Regression RMSE train 0.9351  | val 0.9274  | test 0.8065

Given this output from the linear regression model I can tell that both models are severely underfitting my data, this is understandable because I only used 5 of 79 features even though they were the top features for responder 7. This suggests that the date_ids after 1500 may be easier to predict than the data that the models were trained on. 

After this model I chose to implement a random-forest regressor attempting to capture the non-linear correlations that were seen in the data. The settings I used for the random-forest regressor were 30 trees (keeping it lower so that the compute time doesn't spike too much), a max depth of 4 to prevent some of the overfitting as well as keeping the comput time lower, and a subsampling rate of 0.6 to decrease compute time and increase the variance in the rows. I definitely plan on adjusting these hyperparameters to increase the performance with more resources from SDSC. 

Random‑Forest RMSE train 0.9349  | val 0.9270  | test 0.8063

The output of my evaluation shows that the model might be underfitting the data based on the high training RMSE with a lower test RMSE, it was not expected that the test RMSE would be so much lower than training and validation. It seems like the data after date_id 1500 might have less variance and therefore be easier for the model to predict. This is something that I will look into as I am training my models further. 

## Future Plans
Going forward I plan on implementing the lags parquet into the training of my models, this will give more information on the past values of the features. I also plan on using more of the features if not all of them in my final model training, I need to figure out the proper amount of resources to reserve for my session. If I am able to get better modelling of responder 7 I plan on moving on to more responders and training models on multiple responders. The final model that I would like to try and implement is an XGBoost model to improve the performance and hopefully getter a lower RMSE.

Judging off the fact that I did not gain much accuracy from the linear vs non-linear model, I will need to look more into my train, validation, and test split as well as look into implementing more features that will provide more value to the model allowing it to capture the complex relationships between features and responders. 

## Link to notebook
https://github.com/brocchio/DSC-232R-Final-Project/blob/Milestone3/Milestone_3_frst_mdl.ipynb
