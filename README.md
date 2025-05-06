Predictive Insights: Enhancing Customer Retention in Fintech through Big Data Analytics 
```

The primary goal of this study is to explore the use of data analysis in improving customer retention and 
enhancing the customer experience, in fintech firms through predictive modeling techniques. Among all the models analyzed, the best performance developed was by the Random Forest model, which gave an accuracy of 73.92%, a precision of 73, a recall of 74, and an F1 score of 73. 
With the random forest, XGBoost came out as a very close second, having an accuracy of 72.19%, and a balanced precision, recall, and F1 scores of 72. 
The current study also established the ANN and GBM models as competitive, the accuracies for both standing at 70.93 and 70.82%, respectively.
Whereas the standard models like Logistic Regression and SVM lagged behind, the Logistic Regression model could reach an accuracy of 65.21%, while SVM reached 64.91%. 
In terms of relationship capture within the dataset, as reflected by the F1 scores of the models, Logistic Regression had an F1 score of 64, while the model SVM scored 63. 
The SHAP analysis also indicated that the top features chosen for churn include purchases_partners, rewards_earned, and reward_rate
LIME supported such findings with a local explanation of churn predictions. It showed that customers with rewards_earned > 44 and purchases_partners > 43 are less likely to churn
