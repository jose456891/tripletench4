import pandas as pd
import numpy as np
import plotly.express as px

def dataprep(df):
    """
    Cleans and converts columns in the DataFrame to the desired data types and drops rows with missing data.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The cleaned and converted DataFrame.
    """
    # Drop rows with missing model_year
    df = df.dropna(subset=['model_year'])

    # Fill missing cylinders values
    # Calculate the mean value for each model
    model_cylinders_mean = df.groupby('model')['cylinders'].mean()
    df['cylinders'] = df.apply(
        lambda row: model_cylinders_mean[row['model']] if pd.isna(row['cylinders']) else row['cylinders'], axis=1
    )
    # Fill any remaining NaN values with the overall mean cylinders value
    overall_mean_cylinders = df['cylinders'].mean()
    df['cylinders'] = df['cylinders'].fillna(overall_mean_cylinders)

    # Fill missing odometer values
    # Calculate the mean odometer value for each condition
    condition_odometer_mean = df.groupby('condition')['odometer'].mean()
    df['odometer'] = df.apply(
        lambda row: condition_odometer_mean[row['condition']] if pd.isna(row['odometer']) else row['odometer'], axis=1
    )

    # Fill missing paint_color values
    # Calculate the most common value for each model
    model_paint_color_mode = df.groupby('model')['paint_color'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    df['paint_color'] = df.apply(
        lambda row: model_paint_color_mode[row['model']] if pd.isna(row['paint_color']) else row['paint_color'], axis=1
    )
    # Fill any remaining NaN values with the overall most common paint color
    overall_most_common_paint_color = df['paint_color'].mode().iloc[0]
    df['paint_color'] = df['paint_color'].fillna(overall_most_common_paint_color)

    # Fill missing is_4wd values with 0.0
    df['is_4wd'] = df['is_4wd'].fillna(0.0)

    # Convert data types
    df['price'] = df['price'].astype('float64')
    df['model_year'] = df['model_year'].astype('int64')
    df['cylinders'] = df['cylinders'].astype('int64')
    df['is_4wd'] = df['is_4wd'].map({1.0: True, 0.0: False})
    df['date_posted'] = pd.to_datetime(df['date_posted']).dt.floor('D')

    return df

def plot_histogram_price_filtered(df, model_year, cylinders=None, condition=None, fuel=None, transmission=None, 
                                  car_type=None, paint_color=None, is_4wd=None, models=None, odometer=None, aggregation='Average Price'):
    """
    Plots a histogram of car prices by model with optional filters.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    model_year (str or tuple): The model year range (tuple) or specific years (str).
    cylinders (list or int, optional): The number of cylinders to filter and plot. Defaults to None.
    condition (list or str, optional): The condition of the car to filter and plot. Defaults to None.
    fuel (list or str, optional): The type of fuel to filter and plot. Defaults to None.
    transmission (list or str, optional): The type of transmission to filter and plot. Defaults to None.
    car_type (list or str, optional): The type of car to filter and plot. Defaults to None.
    paint_color (list or str, optional): The paint color of the car to filter and plot. Defaults to None.
    is_4wd (list or bool, optional): Whether the car is 4WD to filter and plot. Defaults to None.
    models (list of str, optional): List of substrings to filter models. Defaults to None.
    odometer (list of str, optional): The odometer range to filter and plot. Defaults to None.
    aggregation (str, optional): The aggregation method for price ('Average Price' or 'Market Capitalization'). Defaults to 'Average Price'.
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
        'models': models,
        'odometer': odometer
    }

    # Treat empty lists as None
    for key, value in filters_applied.items():
        if isinstance(value, list) and len(value) == 0:
            filters_applied[key] = None

    # Apply model_year filter
    if filters_applied['model_year'] is not None:
        if isinstance(filters_applied['model_year'], tuple):
            df = df[df['model_year'].between(filters_applied['model_year'][0], filters_applied['model_year'][1])]
        else:
            years = [int(year) for year in filters_applied['model_year'].split(',')]
            df = df[df['model_year'].isin(years)]

    # Apply remaining filters
    if filters_applied['cylinders'] is not None:
        df = df[df['cylinders'].isin(filters_applied['cylinders'])]
    if filters_applied['condition'] is not None:
        df = df[df['condition'].isin(filters_applied['condition'])]
    if filters_applied['fuel'] is not None:
        df = df[df['fuel'].isin(filters_applied['fuel'])]
    if filters_applied['transmission'] is not None:
        df = df[df['transmission'].isin(filters_applied['transmission'])]
    if filters_applied['car_type'] is not None:
        df = df[df['type'].isin(filters_applied['car_type'])]
    if filters_applied['paint_color'] is not None:
        df = df[df['paint_color'].isin(filters_applied['paint_color'])]
    if filters_applied['is_4wd'] is not None:
        df = df[df['is_4wd'].isin(filters_applied['is_4wd'])]
    if filters_applied['models'] is not None:
        model_pattern = '|'.join(filters_applied['models'])
        df = df[df['model'].str.contains(model_pattern, case=False, na=False)]
    if filters_applied['odometer'] is not None:
        odometer_bins = {
            "<50K": (0, 50000),
            "50K-100K": (50000, 100000),
            "100K-150K": (100000, 150000),
            "150K-200K": (150000, 200000),
            "200K+": (200000, float('inf'))
        }
        df = pd.concat([df[(df['odometer'] >= odometer_bins[bin][0]) & (df['odometer'] < odometer_bins[bin][1])] for bin in filters_applied['odometer']])

    # Set the aggregation function for price
    if aggregation == 'Average Price':
        df_grouped = df.groupby('model')['price'].mean().reset_index()
        y_title = 'Average Price'
    else:
        df_grouped = df.groupby('model')['price'].sum().reset_index()
        y_title = 'Market Capitalization'
    
    # Create the bar chart
    fig = px.bar(df_grouped, x='model', y='price',
                 title="Histogram of Car Prices by Model" + (f" for {filters_applied['model_year']}" if filters_applied['model_year'] else ""),
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(xaxis_title='Model', yaxis_title=y_title, template='plotly_white')

    return fig, len(df)

def plot_scatterplot_price_year(df, model_year, cylinders=None, condition=None, fuel=None, transmission=None, 
                                car_type=None, paint_color=None, is_4wd=None, models=None, odometer=None, aggregation='Average Price'):
    """
    Plots a scatterplot of car prices over the years with optional filters.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    model_year (str or tuple): The model year range (tuple) or specific years (str).
    cylinders (list or int, optional): The number of cylinders to filter and plot. Defaults to None.
    condition (list or str, optional): The condition of the car to filter and plot. Defaults to None.
    fuel (list or str, optional): The type of fuel to filter and plot. Defaults to None.
    transmission (list or str, optional): The type of transmission to filter and plot. Defaults to None.
    car_type (list or str, optional): The type of car to filter and plot. Defaults to None.
    paint_color (list or str, optional): The paint color of the car to filter and plot. Defaults to None.
    is_4wd (list or bool, optional): Whether the car is 4WD to filter and plot. Defaults to None.
    models (list of str, optional): List of substrings to filter models. Defaults to None.
    odometer (list of str, optional): The odometer range to filter and plot. Defaults to None.
    aggregation (str, optional): The aggregation method for price ('Average Price' or 'Market Capitalization'). Defaults to 'Average Price'.
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
        'models': models,
        'odometer': odometer
    }

    # Treat empty lists as None
    for key, value in filters_applied.items():
        if isinstance(value, list) and len(value) == 0:
            filters_applied[key] = None

    # Apply model_year filter
    if filters_applied['model_year'] is not None:
        if isinstance(filters_applied['model_year'], tuple):
            df = df[df['model_year'].between(filters_applied['model_year'][0], filters_applied['model_year'][1])]
        else:
            years = [int(year) for year in filters_applied['model_year'].split(',')]
            df = df[df['model_year'].isin(years)]

    # Apply remaining filters
    if filters_applied['cylinders'] is not None:
        df = df[df['cylinders'].isin(filters_applied['cylinders'])]
    if filters_applied['condition'] is not None:
        df = df[df['condition'].isin(filters_applied['condition'])]
    if filters_applied['fuel'] is not None:
        df = df[df['fuel'].isin(filters_applied['fuel'])]
    if filters_applied['transmission'] is not None:
        df = df[df['transmission'].isin(filters_applied['transmission'])]
    if filters_applied['car_type'] is not None:
        df = df[df['type'].isin(filters_applied['car_type'])]
    if filters_applied['paint_color'] is not None:
        df = df[df['paint_color'].isin(filters_applied['paint_color'])]
    if filters_applied['is_4wd'] is not None:
        df = df[df['is_4wd'].isin(filters_applied['is_4wd'])]
    if filters_applied['models'] is not None:
        model_pattern = '|'.join(filters_applied['models'])
        df = df[df['model'].str.contains(model_pattern, case=False, na=False)]
    if filters_applied['odometer'] is not None:
        odometer_bins = {
            "<50K": (0, 50000),
            "50K-100K": (50000, 100000),
            "100K-150K": (100000, 150000),
            "150K-200K": (150000, 200000),
            "200K+": (200000, float('inf'))
        }
        df = pd.concat([df[(df['odometer'] >= odometer_bins[bin][0]) & (df['odometer'] < odometer_bins[bin][1])] for bin in filters_applied['odometer']])

    # Set the aggregation function for price
    if aggregation == 'Average Price':
        df_grouped = df.groupby('model_year')['price'].mean().reset_index()
        y_title = 'Average Price'
    else:
        df_grouped = df.groupby('model_year')['price'].sum().reset_index()
        y_title = 'Market Capitalization'
    
    # Create the scatter plot
    fig = px.scatter(df_grouped, x='model_year', y='price', 
                     title="Scatterplot of Car Prices over Years",
                     labels={'model_year': 'Model Year', 'price': y_title},
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(xaxis_title='Model Year', yaxis_title=y_title, template='plotly_white')

    return fig, len(df)

def update_filter_options(df, selected_filters):
    # Apply model_year filter
    if selected_filters['model_year']:
        if isinstance(selected_filters['model_year'], tuple):
            df = df[df['model_year'].between(selected_filters['model_year'][0], selected_filters['model_year'][1])]
        else:
            years = [int(year) for year in selected_filters['model_year'].split(',')]
            df = df[df['model_year'].isin(years)]
    # Apply remaining filters
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
    if selected_filters['odometer']:
        odometer_bins = {
            "<50K": (0, 50000),
            "50K-100K": (50000, 100000),
            "100K-150K": (100000, 150000),
            "150K-200K": (150000, 200000),
            "200K+": (200000, float('inf'))
        }
        df = pd.concat([df[(df['odometer'] >= odometer_bins[bin][0]) & (df['odometer'] < odometer_bins[bin][1])] for bin in selected_filters['odometer']])
    return df