import streamlit as st
import pandas as pd
from functions import plot_histogram_price_filtered, plot_scatterplot_price_year, update_filter_options, dataprep

# Load Data
df_dirty = pd.read_csv('vehicles_us.csv')

# Clean Data
df = dataprep(df_dirty)

# Print df to terminal
df.info()

st.title("Car Price Analysis with Filters")

# Sidebar menu for navigation
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu("Menu", ["Histogram", "Scatterplot"], icons=['bar-chart', 'scatter-chart'], menu_icon="cast", default_index=0)

# Get unique filter options based on current selection
def get_unique_options(df, column):
    return df[column].dropna().unique().tolist()

# Model year selection with slider or input
st.sidebar.write("Model Year")
model_year_option = st.sidebar.radio("Select Option", ["Slider", "Input"])
if model_year_option == "Slider":
    min_year, max_year = int(df['model_year'].min()), int(df['model_year'].max())
    selected_model_years = st.sidebar.slider("Model Year Range", min_year, max_year, (min_year, max_year))
else:
    selected_model_years = st.sidebar.text_input("Enter Model Year(s) (comma separated)", "2000,2010")

# Initialize filter options
selected_filters = {
    'model_year': selected_model_years,
    'cylinders': st.sidebar.multiselect("Cylinders", get_unique_options(df, 'cylinders')),
    'condition': st.sidebar.multiselect("Condition", get_unique_options(df, 'condition')),
    'fuel': st.sidebar.multiselect("Fuel", get_unique_options(df, 'fuel')),
    'transmission': st.sidebar.multiselect("Transmission", get_unique_options(df, 'transmission')),
    'car_type': st.sidebar.multiselect("Car Type", get_unique_options(df, 'type')),
    'paint_color': st.sidebar.multiselect("Paint Color", get_unique_options(df, 'paint_color')),
    'is_4wd': st.sidebar.multiselect("Is 4WD", [True, False]),
    'models': st.sidebar.multiselect("Models", get_unique_options(df, 'model')),
    'odometer': st.sidebar.multiselect("Odometer", ["<50K", "50K-100K", "100K-150K", "150K-200K", "200K+"])
}
aggregation = st.sidebar.radio("Aggregation Method", ['Average Price', 'Market Capitalization'])

filtered_df = update_filter_options(df, selected_filters)

# Apply button
if st.sidebar.button("Apply"):
    if selected == "Histogram":
        fig, count = plot_histogram_price_filtered(filtered_df, selected_filters['model_year'], 
                                                   selected_filters['cylinders'], selected_filters['condition'], 
                                                   selected_filters['fuel'], selected_filters['transmission'], 
                                                   selected_filters['car_type'], selected_filters['paint_color'], 
                                                   selected_filters['is_4wd'], selected_filters['models'], 
                                                   selected_filters['odometer'], aggregation)
        st.sidebar.write(f"Number of cars matching filters: {count}")
        st.plotly_chart(fig)
    elif selected == "Scatterplot":
        fig, count = plot_scatterplot_price_year(filtered_df, selected_filters['model_year'], 
                                                 selected_filters['cylinders'], selected_filters['condition'], 
                                                 selected_filters['fuel'], selected_filters['transmission'], 
                                                 selected_filters['car_type'], selected_filters['paint_color'], 
                                                 selected_filters['is_4wd'], selected_filters['models'], 
                                                 selected_filters['odometer'], aggregation)
        st.sidebar.write(f"Number of cars matching filters: {count}")
        st.plotly_chart(fig)
