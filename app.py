import streamlit as st
import pandas as pd
from functions import plot_histogram_price_filtered, update_filter_options, dataprep, plot_scatterplot_price_year

# Load Data
df_dirty = pd.read_csv('vehicles_us.csv')

# Clean Data
df = dataprep(df_dirty)

# Print df to terminal
df.info()

st.title("Car Price Analysis")

# Sidebar menu for navigation
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu("Menu", ["Histogram", "Scatterplot"], icons=['bar-chart', 'scatter-chart'], menu_icon="cast", default_index=0)

# Get unique filter options based on current selection
def get_unique_options(df, column):
    return df[column].dropna().unique().tolist()

# Initialize filter options
selected_filters = {
    'model_year': st.sidebar.multiselect("Model Year", get_unique_options(df, 'model_year')),
    'cylinders': st.sidebar.multiselect("Cylinders", get_unique_options(df, 'cylinders')),
    'condition': st.sidebar.multiselect("Condition", get_unique_options(df, 'condition')),
    'fuel': st.sidebar.multiselect("Fuel", get_unique_options(df, 'fuel')),
    'transmission': st.sidebar.multiselect("Transmission", get_unique_options(df, 'transmission')),
    'car_type': st.sidebar.multiselect("Car Type", get_unique_options(df, 'type')),
    'paint_color': st.sidebar.multiselect("Paint Color", get_unique_options(df, 'paint_color')),
    'is_4wd': st.sidebar.multiselect("Is 4WD", [True, False]),
    'models': st.sidebar.multiselect("Models", get_unique_options(df, 'model'))
}
aggregation = st.sidebar.radio("Aggregation Method", ['Average', 'Market Share'])


# Apply button
if st.sidebar.button("Apply"):
    filtered_df = update_filter_options(df, selected_filters)
    if selected == "Histogram":
        fig, count = plot_histogram_price_filtered(filtered_df, model_year=selected_filters['model_year'], 
                                                   cylinders=selected_filters['cylinders'], condition=selected_filters['condition'], 
                                                   fuel=selected_filters['fuel'], transmission=selected_filters['transmission'], 
                                                   car_type=selected_filters['car_type'], paint_color=selected_filters['paint_color'], 
                                                   is_4wd=selected_filters['is_4wd'], models=selected_filters['models'], aggregation=aggregation)
        st.sidebar.write(f"Number of cars matching filters: {count}")
        st.plotly_chart(fig)
    elif selected == "Scatterplot":
        fig, count = plot_scatterplot_price_year(filtered_df, model_year=selected_filters['model_year'], 
                                                 cylinders=selected_filters['cylinders'], condition=selected_filters['condition'], 
                                                 fuel=selected_filters['fuel'], transmission=selected_filters['transmission'], 
                                                 car_type=selected_filters['car_type'], paint_color=selected_filters['paint_color'], 
                                                 is_4wd=selected_filters['is_4wd'], models=selected_filters['models'])
        st.sidebar.write(f"Number of cars matching filters: {count}")
        st.plotly_chart(fig)


