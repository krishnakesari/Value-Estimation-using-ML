# Value Estimation using ML

In this repository I will show, how we can build a value estimation system using Supervised Learning

Libraries Used: Numpy, Pandas, Scikit-learn

Data: 40000 House Prices and 18 Features (10 times or more data points than the number of features)

Steps Involved:
1. Feature Selection 
2. Training Data
3. Finding right weights
4. Find Equation to model the prediction
5. Build Cost function to quantify the error in the model
6. Use Optimization algorithm (like gradient descent) to find model parameters that minimize cost function 
7. Gradient Boosting - Ensemble of decision trees to predict values (Building decision based on each other)

Model Interpretation:

Overfitting:
If training set error very low
Test set error very high

Solution: Simpler Decision tress or Small Decision Trees

Underfitting:
Training set error very high
Test set error very high

Solution: Deeper Decision trees 