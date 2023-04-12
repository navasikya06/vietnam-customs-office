# Detecting anomalies in customs subsmission at Vietnam Customs Office

Using time series exponential smooth model in detecting anomalies from Customs data

* Read the following time-series data into array
Total import value by month, Total export value by month, Number of import submissions by month, Number of export submissions by month

* 4 models are taken into consideration to detect anomaly in the time series: Simple exponential smoothing, Holt-Winters Double Exponential Smoothing, Holt-Winters Holt-Winters Triple Exponential Smoothing additive, Holt-Winters Holt-Winters Triple Exponential Smoothing multiplicative.

Each model has option to automatically or manually assign parameters like alpha, beta and gamma.

* Graphs include: data and their models, anomalies detected by normal distribution, anomalies detected by quartile, and histogram of the standard errors. 

Data set: https://drive.google.com/file/d/1E6aZEHWqhT3iTtu2Yh-l_voKcgy0keiM/view?usp=sharing
