# AirBnB_Analysis
Analyzing  AirBnB business and understand how it has grown over the years. I have anlyzed the Airbnb Boston dataset from 2008 to 2016. The dataset has listings activities for hosts and reviews from guests who visited the city during that period. The dataset has 3585 distinct listings.


#### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

### Installation <a name="installation"></a>
- Download and install a recent version of Python Anaconda distribution 
- Clone the repository to a local drive and run all files
- The dataset is available in the repos but, but it is available on Kaggle and can be downloaded [here](https://www.kaggle.com/search?q=boston+airbnb+in%3Adatasets)

### Project Motivation <a name="motivation"></a>

In this project , I am interested in learning more about the evolution of the AirBnB business in Boston and using the current data to predict the price of a listing.

1. How has the AirbnB business growth or evolution in Boston ? 
2. what are the most popular neigbourhoud ? and what make them popular ? 
3. When is the busiest time for the business ?
4. What features are most related to the price ?
5. Can we predict the Booking price ?

### File Description <a name="files"></a>

There are 1 notebooks in this repository for different purpose:
- The `AirbnB_Analysis.ipynb`: file analyzes different aspects of the AirBnB in Boston and answers most of the questions above.
- The `data_utils.py` : files are all functions and steps necessary to perform in order to predict a booking price, all put together.
- The `data` folder has the cvs files used in this analysis
    - reviews.csv  : unique id for each reviewer and detailed comments
    - listings.csv : Listings, including full descriptions and average review score
    - calendar.csv : including listing id and the price and availability for that day

### Results <a name="results"></a>

My findings can be found [here](https://medium.com/p/980d409a0482).

### Licensing, Authors, Acknowledgements<a name="licensing"></a>

You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/search?q=boston+airbnb+in%3Adatasets).  