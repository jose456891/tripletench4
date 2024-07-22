# Car Price Analysis App

This is a Streamlit web application for analyzing car prices using histograms and scatterplots with various filters.


## Demo

https://tripletench4.onrender.com/

## Features

- **Histogram Page**: Display a histogram of car prices by model with optional filters.
- **Scatterplot Page**: Display a scatterplot of car prices over the years with optional filters.
- **Filters**: Filter cars by model year, cylinders, condition, fuel, transmission, car type, paint color, and 4WD status.
- **Aggregation**: Choose between mean and sum aggregation methods for the histogram.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/jose456891/tripletench4.git
    cd tripletench4
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py

## Notes
    tested with Python 3.10.0
    ```

## Usage

1. **Histogram Page**:
   - Use the sidebar to select filters and the aggregation method.
   - Click the "Apply" button to display the histogram with the selected filters.

2. **Scatterplot Page**:
   - Use the sidebar to select filters.
   - Click the "Apply" button to display the scatterplot with the selected filters.

## File Structure
tripletench4/

├── .streamlit/

    ├── config.toml # streamlit config

├── notebooks/

    ├── EDA.ipynb # jupynter notebook for exploratory analysis and scratch pad

├── screenshots/

    ├── "Average car price dovetail.png" # This graph represents average vehicle prices by year in the platform. Loks like a dovetail. The plateau is around the 2000s cars. These years seems to have the lowest price on average.

    ├── "Capitalizaiton by vehicle year.png" # This graph represents vehicle years and ther market capitailization in the platform. Looks like cars 2 years old 2016 (assuming 2018 is current data year) have the highest market capitalization.

    ├── "Market Capitalization by Model" # This graph represents vehicle model market capitalization. Looks like the F150 is America's most popular model.

├── app.py # Main Streamlit app

├── functions.py # Helper functions for data preparation and plotting

├── vehicles_us.csv # Dataset containing car information

├── requirements.txt # List of required Python packages

├── LICENSE # GNU License

└── README.md # This readme file

## Conclusions

Without using any filers I've obtained 3 different conclusions on the data from this organizaion: (Max data year is 2018)

1.  Average car prices form a dovetail by not selecting any filetes, using scatterplot and selecting the "Average Vehicle Price". Please see screenshot. It looks like a dovetail, with the lowest average car prioces belogning to vehicle years of the 2000s.

2.  Capitalization by vehicle year. It looks like a b ell curve, please see the screenshot by the same name. Assuming the last data yeart is 2018, 201y cars mnake the majority of the cars for sale in this platform.

3.  Market Capitalization by Model. Look for screenshot of same name. The F150 is the most populat vehicle by capitalizaiton.


## Data

The dataset (`vehicles_us.csv`) should contain the following columns:
- `price`: The price of the car.
- `model_year`: The year the car model was made.
- `model`: The model of the car.
- `cylinders`: The number of cylinders in the car's engine.
- `condition`: The condition of the car (e.g., good, like new).
- `fuel`: The type of fuel the car uses (e.g., gas, diesel).
- `transmission`: The type of transmission (e.g., automatic).
- `type`: The type of car (e.g., sedan, SUV).
- `paint_color`: The color of the car's paint.
- `is_4wd`: Whether the car is 4WD (True or False).
- `date_posted`: The date the car was posted for sale.
- `days_listed`: The number of days the car was listed for sale.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU License. Please see LICENSE file. Please feel free to contact me at jose456891@gmail.com

## Acknowledgments

- [Streamlit](https://streamlit.io/) for providing an easy-to-use framework for creating web applications in Python.
- [Plotly](https://plotly.com/python/) for the interactive plotting library.
- [TripleTen ](https://tripleten.com/) for the this exciting excersice.



