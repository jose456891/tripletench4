import pandas as pd
import plotly.express as px

def dataprep(df):
    """
    Cleans and converts columns in the DataFrame to the desired data types and drops rows with missing data.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The cleaned and converted DataFrame.
    """
    # Drop rows with missing data
    # df = df.dropna()

    # conversions datatype
    # Convert 'price' to float
    df['price'] = df['price'].astype('float64')

    # Convert 'model_year' to int
    df['model_year'] = df['model_year'].astype('Int64')

    # Convert 'cylinders' to int
    df['cylinders'] = df['cylinders'].astype('Int64')

    # Convert 'is_4wd' to boolean, skipping NaN values
    df['is_4wd'] = df['is_4wd'].map({1.0: True, 0.0: False})

    # Convert 'date_posted' to datetime
    df['date_posted'] = pd.to_datetime(df['date_posted']).dt.floor('D')

    return df

def plot_histogram_price_filtered(df, model_year=None, cylinders=None, condition=None, fuel=None, transmission=None, 
                                  car_type=None, paint_color=None, is_4wd=None, models=None, aggregation='mean'):
    """
    Plots a histogram of car prices by model with optional filters.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    model_year (list or int, optional): The specific model year(s) to filter and plot. Defaults to None.
    cylinders (list or int, optional): The number of cylinders to filter and plot. Defaults to None.
    condition (list or str, optional): The condition of the car to filter and plot. Defaults to None.
    fuel (list or str, optional): The type of fuel to filter and plot. Defaults to None.
    transmission (list or str, optional): The type of transmission to filter and plot. Defaults to None.
    car_type (list or str, optional): The type of car to filter and plot. Defaults to None.
    paint_color (list or str, optional): The paint color of the car to filter and plot. Defaults to None.
    is_4wd (list or bool, optional): Whether the car is 4WD to filter and plot. Defaults to None.
    models (list of str, optional): List of substrings to filter models. Defaults to None.
    aggregation (str, optional): The aggregation method for price ('mean' or 'sum'). Defaults to 'mean'.
    """
    # Apply filters
    filters_applied = {
        'model_year': model_year,
        'cylinders': cylinders,
        'condition': condition,
        'fuel': fuel,
        'transmission': transmission,
        'car_type': car_type,
        'paint_color': paint_color,
        'is_4wd': is_4wd,
        'models': models
    }

    # Treat empty lists as None
    for key, value in filters_applied.items():
        if isinstance(value, list) and len(value) == 0:
            filters_applied[key] = None

    if filters_applied['model_year'] is not None:
        if isinstance(filters_applied['model_year'], list):
            df = df[df['model_year'].isin(filters_applied['model_year'])]
        else:
            df = df[df['model_year'] == filters_applied['model_year']]
    if filters_applied['cylinders'] is not None:
        if isinstance(filters_applied['cylinders'], list):
            df = df[df['cylinders'].isin(filters_applied['cylinders'])]
        else:
            df = df[df['cylinders'] == filters_applied['cylinders']]
    if filters_applied['condition'] is not None:
        if isinstance(filters_applied['condition'], list):
            df = df[df['condition'].isin(filters_applied['condition'])]
        else:
            df = df[df['condition'] == filters_applied['condition']]
    if filters_applied['fuel'] is not None:
        if isinstance(filters_applied['fuel'], list):
            df = df[df['fuel'].isin(filters_applied['fuel'])]
        else:
            df = df[df['fuel'] == filters_applied['fuel']]
    if filters_applied['transmission'] is not None:
        if isinstance(filters_applied['transmission'], list):
            df = df[df['transmission'].isin(filters_applied['transmission'])]
        else:
            df = df[df['transmission'] == filters_applied['transmission']]
    if filters_applied['car_type'] is not None:
        if isinstance(filters_applied['car_type'], list):
            df = df[df['type'].isin(filters_applied['car_type'])]
        else:
            df = df[df['type'] == filters_applied['car_type']]
    if filters_applied['paint_color'] is not None:
        if isinstance(filters_applied['paint_color'], list):
            df = df[df['paint_color'].isin(filters_applied['paint_color'])]
        else:
            df = df[df['paint_color'] == filters_applied['paint_color']]
    if filters_applied['is_4wd'] is not None:
        if isinstance(filters_applied['is_4wd'], list):
            df = df[df['is_4wd'].isin(filters_applied['is_4wd'])]
        else:
            df = df[df['is_4wd'] == filters_applied['is_4wd']]
    if filters_applied['models'] is not None:
        model_pattern = '|'.join(filters_applied['models'])
        df = df[df['model'].str.contains(model_pattern, case=False, na=False)]
    
    # Set the aggregation function for price
    if aggregation == 'mean':
        df_grouped = df.groupby('model')['price'].mean().reset_index()
        y_title = 'Mean Price'
    else:
        df_grouped = df.groupby('model')['price'].sum().reset_index()
        y_title = 'Market Share'
    
    # Create the bar chart
    fig = px.bar(df_grouped, x='model', y='price',
                 title="Histogram of Car Prices by Model" + (f" for {filters_applied['model_year']}" if filters_applied['model_year'] else ""),
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(xaxis_title='Model', yaxis_title=y_title, template='plotly_white')

    return fig, len(df)

def plot_scatterplot_price_year(df, model_year=None, cylinders=None, condition=None, fuel=None, transmission=None, 
                                car_type=None, paint_color=None, is_4wd=None, models=None, aggregation='mean'):
    """
    Plots a scatterplot of car prices over the years with optional filters.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    model_year (list or int, optional): The specific model year(s) to filter and plot. Defaults to None.
    cylinders (list or int, optional): The number of cylinders to filter and plot. Defaults to None.
    condition (list or str, optional): The condition of the car to filter and plot. Defaults to None.
    fuel (list or str, optional): The type of fuel to filter and plot. Defaults to None.
    transmission (list or str, optional): The type of transmission to filter and plot. Defaults to None.
    car_type (list or str, optional): The type of car to filter and plot. Defaults to None.
    paint_color (list or str, optional): The paint color of the car to filter and plot. Defaults to None.
    is_4wd (list or bool, optional): Whether the car is 4WD to filter and plot. Defaults to None.
    models (list of str, optional): List of substrings to filter models. Defaults to None.
    aggregation string of median or sum. Defaults to median.
    """
    # Apply filters
    filters_applied = {
        'model_year': model_year,
        'cylinders': cylinders,
        'condition': condition,
        'fuel': fuel,
        'transmission': transmission,
        'car_type': car_type,
        'paint_color': paint_color,
        'is_4wd': is_4wd,
        'models': models
    }

    # Treat empty lists as None
    for key, value in filters_applied.items():
        if isinstance(value, list) and len(value) == 0:
            filters_applied[key] = None

    if filters_applied['model_year'] is not None:
        if isinstance(filters_applied['model_year'], list):
            df = df[df['model_year'].isin(filters_applied['model_year'])]
        else:
            df = df[df['model_year'] == filters_applied['model_year']]
    if filters_applied['cylinders'] is not None:
        if isinstance(filters_applied['cylinders'], list):
            df = df[df['cylinders'].isin(filters_applied['cylinders'])]
        else:
            df = df[df['cylinders'] == filters_applied['cylinders']]
    if filters_applied['condition'] is not None:
        if isinstance(filters_applied['condition'], list):
            df = df[df['condition'].isin(filters_applied['condition'])]
        else:
            df = df[df['condition'] == filters_applied['condition']]
    if filters_applied['fuel'] is not None:
        if isinstance(filters_applied['fuel'], list):
            df = df[df['fuel'].isin(filters_applied['fuel'])]
        else:
            df = df[df['fuel'] == filters_applied['fuel']]
    if filters_applied['transmission'] is not None:
        if isinstance(filters_applied['transmission'], list):
            df = df[df['transmission'].isin(filters_applied['transmission'])]
        else:
            df = df[df['transmission'] == filters_applied['transmission']]
    if filters_applied['car_type'] is not None:
        if isinstance(filters_applied['car_type'], list):
            df = df[df['type'].isin(filters_applied['car_type'])]
        else:
            df = df[df['type'] == filters_applied['car_type']]
    if filters_applied['paint_color'] is not None:
        if isinstance(filters_applied['paint_color'], list):
            df = df[df['paint_color'].isin(filters_applied['paint_color'])]
        else:
            df = df[df['paint_color'] == filters_applied['paint_color']]
    if filters_applied['is_4wd'] is not None:
        if isinstance(filters_applied['is_4wd'], list):
            df = df[df['is_4wd'].isin(filters_applied['is_4wd'])]
        else:
            df = df[df['is_4wd'] == filters_applied['is_4wd']]
    if filters_applied['models'] is not None:
        model_pattern = '|'.join(filters_applied['models'])
        df = df[df['model'].str.contains(model_pattern, case=False, na=False)]
    
    # Set the aggregation function for price
    if aggregation == 'mean':
        df_grouped = df.groupby('model')['price'].mean().reset_index()
        y_title = 'Mean Price'
    else:
        df_grouped = df.groupby('model')['price'].sum().reset_index()
        y_title = 'Market Share'

    # Create the scatter plot
    fig = px.scatter(df_grouped, x='model_year', y='price', color='model',
                     title="Scatterplot of Car Prices over Years",
                     labels={'model_year': 'Model Year', 'price': 'Price'},
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(xaxis_title='Model Year', yaxis_title=y_title, template='plotly_white')

    return fig, len(df_grouped)

def update_filter_options(df, selected_filters):
    if selected_filters['model_year']:
        df = df[df['model_year'].isin(selected_filters['model_year'])]
    if selected_filters['cylinders']:
        df = df[df['cylinders'].isin(selected_filters['cylinders'])]
    if selected_filters['condition']:
        df = df[df['condition'].isin(selected_filters['condition'])]
    if selected_filters['fuel']:
        df = df[df['fuel'].isin(selected_filters['fuel'])]
    if selected_filters['transmission']:
        df = df[df['transmission'].isin(selected_filters['transmission'])]
    if selected_filters['car_type']:
        df = df[df['type'].isin(selected_filters['car_type'])]
    if selected_filters['paint_color']:
        df = df[df['paint_color'].isin(selected_filters['paint_color'])]
    if selected_filters['is_4wd']:
        df = df[df['is_4wd'].isin(selected_filters['is_4wd'])]
    if selected_filters['models']:
        model_pattern = '|'.join(selected_filters['models'])
        df = df[df['model'].str.contains(model_pattern, case=False, na=False)]
    return df
