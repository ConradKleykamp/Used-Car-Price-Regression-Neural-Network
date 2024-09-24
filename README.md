# Used-Car-Price-Regression-Neural-Network
Building a neural network to predict the prices of used cars

![image](https://github.com/user-attachments/assets/059895f2-8bc2-4736-8013-77871ccf9687)

---

### Objective
This project has been completed as part of Kaggle's Regression of Used Car Prices competition (S4E9). Furthermore, this project presents an opportunity to practice both data cleaning and the development of neural networks. The overall goal here is to develop a neural network model which will predict the price of a car based upon its predictor variables. This will be accomplished by first determining an efficient method of dealing with missing data. Following this, a neural network will be built and evaluated by the root mean squared error (RMSE) of the predicted car prices compared to the actual car prices.

Two of the datasets used in this project were made available by Kaggle, i.e. the train and test sets. These sets were generated from a deep learning model trained on the Used Car Price Prediction Dataset. To enhance the amount of data used for model training, I have also included the original dataset in this project. The Kaggle train dataset contains 188533 rows and 13 columns of data. The Kaggle test dataset contains 125690 rows and 12 columns of data. The original dataset contains 4009 rows and 12 columns of data. The target variable is price (dollars). There are 11 predictor variables with varying datatypes. 

---

### Methods
Libraries Used
- pandas
- numpy
- matplotlib
- sklearn (LabelEncoder, MinMaxScaler, train_test_split, mean_squared_error)
- tensorflow
- tensorflow.keras (Sequential, Input, Dense, Dropout, BatchNormalization, Adam)

Data Cleaning
- Removing unnecessary string elements (ex: "$", "mi.") from columns in the original dataset
- Concatenating Kaggle and original datasets
- Resolving missing values for columns with nulls

Data Preprocessing
- Encoding object type columns
- Splitting the data into training and testing sets (80/20)
- Scaling the data (MinMaxScaler)

NN Model Parameters
- Model: Sequential
    - Dense (128, 'relu')
    - Batch Normalization
    - Dropout (0.1)
    - Dense 2 (128, 'relu')
    - Batch Normalization 2
    - Dropout 2 (0.1)
    - Dense 3 (64, 'relu')
    - Batch Normalization 3
    - Dense 4 (1, 'linear')
- Total Parameters: 27,649
    - Trainable: 27,009
    - Non-Trainable: 640
- Compiling:
    - Adam (learning rate: 0.001)
    - Mean squared error
- Fitting:
    - Epochs: 50
    - Batch Size: 128

---

### General Results

![image](https://github.com/user-attachments/assets/b29f0af0-62e5-4932-bdd5-b5b122585be6)

Both training and validation losses decreased significantly from epochs 0 to 20, then tapered off afterwards. Neither loss appeared to increase significantly across all 50 epochs, further suggesting that the model did not overfit.

Possible Improvements:
- Adjusting learning rate
- Adding additional NN layers
- Experimenting with different activation functions
- Experimenting with different batch sizes
- Feature engineering
