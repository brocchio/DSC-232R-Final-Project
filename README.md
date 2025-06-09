## Introduction
Being able to predict high dimensional time series data is a task that can be critical in many different industries. The goal of my project was to establish a model that would be able to accurately predict anonymized responders from a high dimensional Jane Street Real Time Market Forecasting dataset. In this project there were 79 anonymized features that were related to 9 different responders. This is a project that could be generalized across multiple different projects and tasks. 

## Methods

## Environment Setup and Data Download
Unfortunately due to this being a Kaggle competition we cannot download the dataset directly from the competition page, https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data, I was able to find a reupload of the dataset here, https://www.kaggle.com/datasets/mohamedsameh0410/jane-street-dataset. Download the dataset locally and use SSH to rsync the dataset into the SDSC server.

![image](https://github.com/user-attachments/assets/5b272d75-2d30-4977-a133-91e91c594c15)

![image](https://github.com/user-attachments/assets/8f872ab1-ab1d-47ff-88d9-e3a63f30919b)



## 1. Data Exploration and Analysis
The dataset that I am using is a real time market forecasting dataset from Janestreet. The dataset consists of a set of timeseries data with 79 features and 9 responders, all anonymized. The goal is to accurately predict the responder variables based off of the features. One of the difficulties that I ran into were the anonymized features, this led me to use standard deviation as the main method to decide on which features are the most valuable. The columns are date, time and symbol ids, along with weight, features 0-78, and responder 0-8. The data files included are the train parquet consisting of 47127338 observations. The lags parquet contains the responders offset by one date id, this shows the value of the responders at a later point in time. The weights features and responders are all flot values generally centered around 0.
The top responders with the highest standard deviation have no missing data, these are some of the features that will be focused on in the model training step.

![image](https://github.com/user-attachments/assets/119cb104-6085-4e2f-964d-2cf8efd10de6)

After looking into the missing data I chose to make a heatmap showing the correlation of the top features and top responders. This showed that there was very little linear correlation and each of the features has around the same level of correlation to the responders with some correlating with different features better. 

![image](https://github.com/user-attachments/assets/6b318636-d64a-4cf3-afb9-22f5cbca853c)



Looking into the top responders and their distributions resulted in this visualization, this led to me try and predict one of these responders with my models. 

![image](https://github.com/user-attachments/assets/7a7c7363-227d-4c97-ae3a-49a9a9c6de38)


## 2. Data Preprocessing
Built a preprocessing pipeline to impute NaNs and infinity values with the median, vectorize the feature columns, and apply a standard scaler to each of the features. This is all done in my pipeline cell that I created at the beginning of my milestone3 notebook. I also chose to split the data into a test training and validation set based on the date_ids, this is to prevent the model from peaking into the future when making its predictions, the training set is based on the first 1350 date_ids, the validation set is from 1351 to 1500, and the test set is any date_id greater than 1500. This puts around 80% of the data in the training set, ~9% in the validation set, and ~12% in the test set. I have not implemented the lags parquet in the training yet as I am still aiming to get a strong baseline. 

## Model 1 Linear Regression
For my first model choice I chose a linear regression model to get a baeline using a basic model to test the RMSE. The linear regression model included 100 max iterations, low regulatization at 0.01. I knew that the performance wouldn't be great due to the fact that in my exploration I saw that there was very weak linear correlation. 

Linear‑Regression RMSE train 0.9351  | val 0.9274  | test 0.8065

Given this output from the linear regression model I can tell that both models are severely underfitting my data, this is understandable because I only used 5 of 79 features even though they were the top features for responder 7. This suggests that the date_ids after 1500 may be easier to predict than the data that the models were trained on. 

## Model 2 Random Forest

After this model I chose to implement a random-forest regressor attempting to capture the non-linear correlations that were seen in the data. The settings I used for the random-forest regressor were 30 trees (keeping it lower so that the compute time doesn't spike too much), a max depth of 4 to prevent some of the overfitting as well as keeping the comput time lower, and a subsampling rate of 0.6 to decrease compute time and increase the variance in the rows. I had to adjust the settings so that I would be able to train the model in a reasonable amount of time without using too much of the SDSC resources. 

Random‑Forest RMSE train 0.9349  | val 0.9270  | test 0.8063

The output of my evaluation shows that the model might be underfitting the data based on the high training RMSE with a lower test RMSE, it was not expected that the test RMSE would be so much lower than training and validation. It seems like the data after date_id 1500 might have less variance and therefore be easier for the model to predict. This is something that planned on looking into but was unable to due to the issues with SDSC during this time.

## Discussion

When looking at the predictions from both of the models I was able to train, we can see that both have very similar results of a sever underfitting. This could mean that the data past date_id 1500 have lower variance and therfore be easier to predict than the first 1500. I was only able to use 5 features in my decision tree model which likely had a large impact on the accuracy of my models given the low variance of each of the features. If I was able to continue with my experimentation, I would aim to use more features in my future models, implement the lag parquets into the training of the model. With my later models I would be able to tune the hyperparameters more in depth as well. 

I think that one of my biggest problems was the fact that the dataset was fully anonymized. This is something I am not very familiar with so there was a steep learning curve, and clearly I still did not entirely understand the best methods for feature selection. 

## Conclusion

Given the issues that the class was having with the SDSC connection, I wasn't able to complete this project to my liking, but if I had the opportunity I have ideas for what could have improved the performance of my models. I planned on implementing the lags parquet into the training of my models, this will give more information on the past values of the features. I also planned on using more of the features if not all of them in my final model training, I need to figure out the proper amount of resources to reserve for my session. If I am able to get better modelling of responder 7 I plan on moving on to more responders and training models on multiple responders. The final model that I would have liked to implement would be an XGBoost model. Given the nature of the dataset that I chose, the relationships are non-linear and it is hard to narrow down the feature selection. I think that being able to include more impactful features as well as a better model would have significantly improved the performance. Something that I wish I had done from the start would be including the lags-parquet in all of the training. Giving the model training access to the lags parquet would help the model by giving it the values of features and responders from nearby date_ids.

This project has left me with a couple final thoughts, one of which is that I should have done this project in a group. I found that between work and everything else with school I was having trouble finding a whole lot of time to work on the project. I think that doing this in a group would have allowed me to accomplish more by splitting the work up and being able to put in more man hours. I am still proud of what I was able to accomplish but wish that there could have been more. 

## Collaboration

This project was done alone by Brandon Rocchio all contributions made by me.

## Link to notebook
https://github.com/brocchio/DSC-232R-Final-Project/blob/Milestone3/Milestone_3_frst_mdl.ipynb
