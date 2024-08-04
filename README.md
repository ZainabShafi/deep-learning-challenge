# deep-learning-challenge

## Model Performance Report 

### Project Overview

The purpose of this analysis is to create a predictive tool for Alphabet Soup, a nonprofit foundation, to help identify which funding applicants are most likely to succeed in their ventures. By applying machine learning and neural networks, I will develop a binary classifier to predict whether an applicant will be successful if funded by Alphabet Soup. This tool will enable Alphabet Soup to make more informed funding decisions, thereby enhancing the effectiveness and impact of their financial support.

**Dataset Overview:** 

The dataset includes various features that provide detailed metadata about each organization. 
Breakdown of dataset columns:

EIN and NAME: These columns provide identification information for each organization.

APPLICATION_TYPE: This indicates the type of application submitted to Alphabet Soup.

AFFILIATION: This describes the affiliated sector of the industry to which the organization belongs.

CLASSIFICATION: This indicates the government classification of the organization.

USE_CASE: This describes the use case for which the funding was requested.

ORGANIZATION: This specifies the type of organization (e.g., nonprofit, for-profit).

STATUS: This indicates the active status of the organization.

INCOME_AMT: This categorizes the organization's income level.

SPECIAL_CONSIDERATIONS: This indicates whether there are any special considerations for the application.

ASK_AMT: This shows the amount of funding requested by the organization.

IS_SUCCESSFUL: This is the target variable indicating whether the funding was used effectively.


**Data Preprocessing Actions:** 

After dropping additional non-beneficial ID columns for the purpose of this analysis:

Target Variable: "IS_SUCCESSFUL"

Feature Variables: "AFFILIATION", "USE_CASE", "ORGANIZATION", "SPECIAL_CONSIDERATIONS", "APPLICATION_TYPE", "CLASSIFICATION", "INCOME_AMT"

1) Selecting cut-off values for "APPLICATION_TYPE" and "CLASSIFICATION" columns based on infrequent categories, and replacing these with 'Other' through for loop - thereby remedying class imbalances.

2) Converting categorical data to numerical data using pd.get_dummies, dropping the unconverted columns, and concatenating with numerical data.

3) Splitting data into training and testing datasets, and then further splitting into feature and target arrays.

4) Utilizing StandardScalar() to scale X data.


Your analysis is clear and well-organized. I've made a few corrections and adjustments for clarity and readability:

### Compiling, Training, and Evaluating Models

To predict whether a campaign is successful (binary classification: successful or not successful), an initial experimental model was built with a focus on simplicity and avoiding overfitting. The following initial parameters were used:

**Neurons = 4 (input layer), 8 (hidden layer), 1 (output layer)**

Four neurons were selected for the input layer to start with simplicity and avoid overfitting. Eight neurons were used in the hidden layer to increase the model's capacity to learn more intricate patterns and relationships between features. The output layer, used for binary classification, has one neuron with a sigmoid activation function to represent the probability of success.

**Layers = 3 layers (1 input, 1 hidden, 1 output)**

Only one hidden layer was included in the initial model to maintain simplicity and identify areas for optimization effectively.

**Activation Functions: 1st layer: tanh, 2nd layer: relu, 3rd layer: sigmoid**

Tanh was used in the first layer due to its zero-centered output values and effectiveness in balancing weights and preventing bias in weight updates. Relu was chosen for the hidden layer to improve performance by providing a constant gradient of 1, making weight updates increasingly proportional to the loss and enhancing the model's ability to learn from less accurate predictions. Sigmoid was used in the output layer to represent probability.

**Was target performance achieved with the initial model?**

Performance metrics using test data:
322/322 - 0s - 225us/step - accuracy: 0.7257 - loss: 0.5562
Loss: 0.5562081336975098, Accuracy: 0.7256559729576111

With an accuracy rate of 72.57%, the model performs relatively well but shows room for improvement, as indicated by the loss rate of 55.62%. This loss value suggests further optimization is needed.

**What steps did you take in your attempts to increase model performance?**

The Keras Tuner was utilized (see "ZS_ModelOptimizer") and additional non-essential or non-enhancing features of the data were removed. Along with "EIN" and "NAME" of the campaigns, the "STATUS" feature was dropped since it only indicates whether a campaign is active or not, providing no new information about its success.

Using the Keras Tuner, a range of hyperparameters were tested:

Activation function choices: Relu, TanH, LeakyRelu.
Min/Max Neurons: 1-30
Min/Max Layers: 1-5
After running a search for the best hyperparameters, the following results were returned for the best validation accuracy:

Validation accuracy: 0.7309

Best validation accuracy so far: 0.7309

For the top 3 hyperparameter configurations, the following results were obtained:

*val_accuracy: 0.7309038043022156*

*Best val_accuracy So Far: 0.7309038043022156*

For the top 3 hyperparameter configurations, the following results were obtained: 

*'activation': 'tanh', 'first_units': 11, 'num_layers': 2, 'units_0': 6, 'units_1': 1, 'units_2': 11, 'units_3': 11, 'units_4': 21, 'tuner/epochs': 20, 'tuner/initial_epoch': 7, 'tuner/bracket': 2, 'tuner/round': 2, 'tuner/trial_id': '0012'*

*'activation': 'leaky_relu', 'first_units': 16, 'num_layers': 2, 'units_0': 1, 'units_1': 6, 'units_2': 1, 'units_3': 21, 'units_4': 11, 'tuner/epochs': 20, 'tuner/initial_epoch': 0, 'tuner/bracket': 0, 'tuner/round':* 

*'activation': 'leaky_relu', 'first_units': 6, 'num_layers': 1, 'units_0': 21, 'units_1': 26, 'units_2': 1, 'units_3': 16, 'units_4': 21, 'tuner/epochs': 7, 'tuner/initial_epoch': 0, 'tuner/bracket': 1, 'tuner/round': 0*

Evaluating the top 3 models against the test data based on the above parameters:

*322/322 - 0s - 386us/step - accuracy: 0.7309 - loss: 0.5580
Loss: 0.5580456256866455, Accuracy: 0.7309038043022156*

*322/322 - 0s - 382us/step - accuracy: 0.7309 - loss: 0.5580
Loss: 0.5579630136489868, Accuracy: 0.7309038043022156*

*322/322 - 0s - 367us/step - accuracy: 0.7307 - loss: 0.5556
Loss: 0.5555964708328247, Accuracy: 0.730709433555603* 

The Keras Tuner's hyperparameter combinations yielded a marginal increase in accuracy—up to 73.07% from 72.57%. However, the loss showed minimal change, with some hyperparameter configurations even leading to a slightly increased loss.

This optimization process has demonstrated a negligible difference in model performance. Therefore, other algorithms, such as Random Forests or perhaps RNNs, should be utilized for further testing.













