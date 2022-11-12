### Time-Varying Impact of the WASDE Report on Corn Futures Prices

This repository contains codes for the paper. The article deduced that when the USDA production estimate is 1% higher than the average private analysts' estimate, futures price goes down by 1% using Linear Regression. Kalman filtering, a Bayesian state-space model, confirmed the existence of time-varying impact of the WASDE report on corn futures prices. Several prediction models tried as well. Please find below the citation and the codes details. Thanks!


@article{ssingh_2021, title={Time-Varying Impact of the WASDE Report on Corn Futures Prices}, url={https://drive.google.com/file/d/1BzRdvEAL9JcMR23XFs60_5xWPhIx7Gzg/view}, author={Singh, Sriramjee}, year={2021}}


#### program2_kf.py

This code employs Kalman filtering approach. 


#### program2_pred.py

This code includes prediction models like, Linear Regression, Support Vector Regression, Random Forest Regression, Locally Weighted Linear Regression, and Fully Connected Neural Networks.


#### program2_prod.py

This code comprises data preparation part.


#### program2_LR.py

This code gives Linear Regression results.


#### program2_graph.py

This is for plotting graphs.

