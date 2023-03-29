# ANNDL2022-23
This repository contains the code for reproducing the models built for solving the challenges of the Artificial Neural Network &amp; Deep Learning Course of Politecnico di Milano.
The model was evaluated using [Codalab](https://codalab.lisn.upsaclay.fr/) with hidden tests.
![grade](https://img.shields.io/badge/Grade-100%25-green)

# Challenge 1
The first challenge was an image classification problem. 

The given dataset was composed of 8 different species of plants which we had to correctly classify. The images were tiny, with a size of 96x96 pixels.
The dataset was unbalanced with fewer images for species 1 and 8. In the folder Challenge 1/Final you can read the report with all techniques we have used to address the problem.

The files used for training the model are in the .ipynb format that can be run online on Google Colab.
These are some example of the images in the dataset:

![Image of Challenge 1](https://github.com/Francesco-Grilli/ANNDL2022-23/blob/25a053a06d79ecc3fb741af2b7bcd615e199ac1d/Challenge%201/img1.jpg)
![Image of Challenge 1](https://github.com/Francesco-Grilli/ANNDL2022-23/blob/25a053a06d79ecc3fb741af2b7bcd615e199ac1d/Challenge%201/img2.jpg)
![Image of Challenge 1](https://github.com/Francesco-Grilli/ANNDL2022-23/blob/25a053a06d79ecc3fb741af2b7bcd615e199ac1d/Challenge%201/img3.jpg)
![Image of Challenge 1](https://github.com/Francesco-Grilli/ANNDL2022-23/blob/25a053a06d79ecc3fb741af2b7bcd615e199ac1d/Challenge%201/img4.jpg)

## Result
The result obtained with the XceptionNet model has an accuracy of 0.87 on the Codalab hidden tests.

## Score
The score obtained for the first challenge was 5.5/5

# Challenge 2
The second challenge was a time-series classification problem.

The given dataset was composed of a time series with 6 features and was structured with a window of 36 elements. The length of the time series is 2429 points. 
In the folder Challenge 2/Final you can read the report of all techniques we have used for addressing the problem.
This is an image of the time series split up by its six features:
![Image of Challenge 2](https://github.com/Francesco-Grilli/ANNDL2022-23/blob/cad6776af6a8fec2a81ea3c56d2ed45791d3e2d4/Challenge%202/time-series.png)

## Result
The best results were obtained with the 2D CNN model with an accuracy of 0.727 on the Codalab hidden tests.

## Score
The score obtained for the second challenge was 5.5/5

# Credits
These two challenges have reached the maximum score 11/10 thanks to:

[Francesco Grilli](https://github.com/Francesco-Grilli)

[Flavio Renzi](https://github.com/FlavioRenzi)

[Jaskaran Singh](https://github.com/zJaska)

[Stefano Vighini]()

