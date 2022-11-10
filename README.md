# mlzoomcamp-midterm

In this project, I develop a model to predict solar irradiation based on factors such as temperature, pressure, humidity,
wind direction and speed. The dataset comes from a Kaggle dataset https://www.kaggle.com/datasets/dronio/SolarEnergy. Knowing
factors that contribute to the intensity of solar irradiation enable us to predict the best way to utilize solar energy.

The dataset consist of these features: UNIX time, date data, time data, radiation (target feature), temperature, pressure,
humidity, wind direction, wind speed, and time of sunrise. After doing correlation among features, temperature is the strongest
factor to consider for solar irradiation intensity. Therefore, I drop all non numerical features and retain temperature, pressure,
humidity, wind direction and speed only to consider for the irradiation.

My base model is a linear regression model between solar radiation and temperature. To get a better model, I try XGBoosting
regressor and XGBoosting with Dmatrix and the ability to tweak its parameters. Both perform better than the base model. After
trying several parameters, I conclude with a final model of XGBoosting with Dmatrix. The program to try several models is
"notebook.ipynb".

Please put the dataset, notebook.ipynb, train.py, predict.py, Pipfile, Pipfile.LOCK and Dockerfile in a single directory. Please
create /app directory as an empty Docker directory. I assume Flask and waitress already installed and Docker daemon is running in
the background.

To run docker, you need to run docker build -t predict . and docker run -it -p 9696:9696 predict:latest
