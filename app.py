import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pycountry
import pycountry_convert as pc
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Charting the Path to Decent Work",
    page_icon="📊",
    layout="wide"
)

# --- Data Loading and Processing ---
@st.cache_data
def load_and_process_data():
    # 1. Load Data
    try:
        edu_df = pd.read_csv('education.csv')
        gen_df = pd.read_csv('gender.csv')
        pov_df = pd.read_csv('poverty.csv')
        soc_df = pd.read_csv('social_protection_labor.csv')
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        return None, None

    # 2. On-the-fly Joining
    df = soc_df.merge(edu_df, on=['country', 'year'], how='outer')
    df = df.merge(gen_df, on=['country', 'year'], how='outer')
    df = df.merge(pov_df, on=['country', 'year'], how='outer')

    # 3. Preprocessing
    df.rename(columns={'country': 'Country', 'year': 'Year'}, inplace=True)
    df = df[(df['Year'] >= 1991) & (df['Year'] <= 2023)]

    # Helper to get continent
    def get_continent(alpha_3):
        try:
            country = pycountry.countries.get(alpha_3=alpha_3)
            if country:
                alpha_2 = country.alpha_2
                continent_code = pc.country_alpha2_to_continent_code(alpha_2)
                continent_name = pc.convert_continent_code_to_continent_name(continent_code)
                return continent_name
        except:
            return None
        return None

    # Helper to get country name
    def get_country_name(alpha_3):
        try:
            country = pycountry.countries.get(alpha_3=alpha_3)
            if country:
                return country.name
        except:
            pass
        return alpha_3

    df['Continent'] = df['Country'].apply(get_continent)
    df['Country Name'] = df['Country'].apply(get_country_name)
    countries_df = df[df['Continent'].notna()].copy()

    # Define Column Constants
    col_unemp_total_orig = 'Unemployment, total (% of total labor force) (modeled ILO estimate)'
    col_unemp_youth_orig = 'Unemployment, youth total (% of total labor force ages 15-24) (modeled ILO estimate)'
    col_gdp_orig = 'GDP per person employed (constant 2021 PPP $)'
    col_vuln_orig = 'Vulnerable employment, total (% of total employment) (modeled ILO estimate)'
    col_serv_orig = 'Employment in services (% of total employment) (modeled ILO estimate)'

    # Rename columns
    rename_map = {
        col_unemp_total_orig: 'Total Unemployment (%)',
        col_unemp_youth_orig: 'Youth Unemployment (%)',
        col_gdp_orig: 'GDP per Person ($)',
        col_vuln_orig: 'Vulnerable Employment (%)',
        col_serv_orig: 'Services Employment (%)'
    }

    df.rename(columns=rename_map, inplace=True)
    countries_df.rename(columns=rename_map, inplace=True)

    return df, countries_df

# Load data
df, countries_df = load_and_process_data()

if df is not None and countries_df is not None:
    # Column names for easy access
    col_unemp_total = 'Total Unemployment (%)'
    col_unemp_youth = 'Youth Unemployment (%)'
    col_gdp = 'GDP per Person ($)'
    col_vuln = 'Vulnerable Employment (%)'
    col_serv = 'Services Employment (%)'

    # --- App Layout ---
    # Sidebar
    with st.sidebar:
        st.header("About the Project")
        st.markdown("""
        **Authors:**
        *   De Guzman, Eric Julian
        *   Fermin, Raudmon Yvhan
        *   Palanog, Alexandra Antonette
        
        *University of Santo Tomas, Manila, Philippines*
        """)
        st.markdown("---")
        st.info("Data Source: World Bank & ILO (1991-2023)")

    # Hero Section
    st.title("Charting the Path to Decent Work")
    st.markdown("### A Visual Exploration of Progress and Disparities in SDG 8")
    
    # Key Metrics (2022)
    df_2022_metrics = countries_df[countries_df['Year'] == 2022]
    avg_unemp = df_2022_metrics[col_unemp_total].mean()
    avg_youth_unemp = df_2022_metrics[col_unemp_youth].mean()
    med_gdp = df_2022_metrics[col_gdp].median()

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Total Unemployment (2022)", f"{avg_unemp:.1f}%")
    c2.metric("Avg Youth Unemployment (2022)", f"{avg_youth_unemp:.1f}%", delta=f"{avg_youth_unemp - avg_unemp:.1f}% Gap", delta_color="inverse")
    c3.metric("Median GDP/Person (2022)", f"${med_gdp:,.0f}")
    
    st.markdown("---")

    st.header("Data Analysis & Visualizations")

    with st.expander("Exploratory Data Analysis (EDA)"):
        st.subheader("Summary Statistics")
        st.write("""
        Descriptive statistics from 1,729 country-year records highlight key patterns. The global mean youth unemployment rate (16.95%) is more than double that of total unemployment (7.73%), with a standard deviation twice as large, confirming youth unemployment as both more severe and volatile. 'GDP per person employed' is highly skewed, with a mean ($60,503) far above the median ($51,064) and a very large standard deviation, underscoring vast economic disparities between nations.
        """)
        
        st.subheader("Distribution Analysis")
        st.write("""
        The frequency distributions of the key indicators provide deeper insight into the data’s structure. The total unemployment rate is clearly right-skewed, with most countries clustered below 10% but a long tail of nations facing much higher rates. 'GDP per person employed' also shows extreme right-skewness, making a logarithmic scale necessary for effective visualization. Even on the log scale, the distribution remains multi-modal, indicating distinct economic clusters of low-, middle-, and high-income countries rather than a smooth continuum.
        """)

        st.subheader("Data Availability Check")
        st.write("""
        A heatmap of the 'Unemployment, total' indicator shows a critical dataset limitation. While high-income countries like Germany, Japan, and the USA have nearly complete data since 1991, coverage in developing and middle-income nations is far patchier, especially in earlier years. The plot also reveals sparse data for the most recent years (post-2021) across almost all sampled countries underscoring the need to focus analysis on 1991–2023 for robustness.
        """)
    
    # --- Viz 1: Global Unemployment Trends ---
    st.subheader("1. Global Unemployment Trends (1991-2023)")
    
    v1_col1, v1_col2 = st.columns([1, 2])
    
    with v1_col1:
        st.markdown("#### The Youth Deficit")
        st.write("""
        **RQ1 & RQ2:** The analysis reveals that global labor markets are highly sensitive to macro-economic shocks (2008 Crisis, COVID-19). 
        
        However, the most critical insight regarding SDG 8.6 is the persistent "youth deficit." Youth unemployment has remained consistently double the total rate (approx. 15-16% vs. 7-8%) for over thirty years. 
        
        The "last in, first out" phenomenon is clearly visible here, where youth unemployment spikes more sharply during crises and recovers more slowly.
        """)

    with v1_col2:
        global_trends = countries_df.groupby('Year')[[col_unemp_total, col_unemp_youth]].mean().reset_index()
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=global_trends['Year'], y=global_trends[col_unemp_total],
                            mode='lines', name='Total Unemployment'))
        fig1.add_trace(go.Scatter(x=global_trends['Year'], y=global_trends[col_unemp_youth],
                            mode='lines', name='Youth Unemployment'))
        fig1.update_layout(title='Global Unemployment Trends (1991-2023)',
                           xaxis_title='Year',
                           yaxis_title='Unemployment Rate (%)',
                           legend_title='Indicator',
                           template='plotly_white',
                           margin=dict(l=20, r=20, t=40, b=20))
        fig1.add_vrect(x0=2008, x1=2009, fillcolor="gray", opacity=0.2, layer="below", line_width=0, annotation_text="2008 Crisis", annotation_position="top left")
        fig1.add_vrect(x0=2020, x1=2021, fillcolor="gray", opacity=0.2, layer="below", line_width=0, annotation_text="COVID-19", annotation_position="top left")
        st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("---")

    # --- Viz 2: Productivity by Region ---
    st.subheader("2. Economic Productivity by Region (2022)")
    
    v2_col1, v2_col2 = st.columns([2, 1])
    
    with v2_col1:
        df_2022 = countries_df[countries_df['Year'] == 2022]
        fig2 = px.box(df_2022, x='Continent', y=col_gdp, 
                      points="all", 
                      hover_name='Country Name',
                      title='Economic Productivity by Region (2022)',
                      log_y=True,
                      color='Continent',
                      template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)

    with v2_col2:
        st.markdown("#### Regional Disparities")
        st.write("""
        **RQ3:** Figure 2 highlights the immense challenge in achieving SDG 8.2. While North America and Europe exhibit high median productivity, Africa and Asia show significantly lower medians and extreme outliers. 
        
        The extreme outliers in Asia and Africa likely represent developed or oil-rich nations, masking the lower productivity of the majority. This suggests that regional averages are insufficient for understanding local economic realities.
        """)
    
    st.markdown("---")

    # --- Viz 3: Map ---
    st.subheader("3. Global Unemployment Rate Map (2022)")
    st.write("""
    A geographic heatmap showing the intensity of total unemployment rates worldwide. Unemployment is not randomly distributed; it often clusters regionally.
    
    **Insight:** While high unemployment is a clear sign of distress, low unemployment in developing regions (visible in parts of Southeast Asia and Africa) should be interpreted with caution. It often reflects high informality and "survival employment," where individuals cannot afford to be unemployed.
    """)
    fig4 = px.choropleth(df_2022, locations="Country",
                        color=col_unemp_total,
                        hover_name="Country Name",
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title="Global Unemployment Rate (2022)",
                        template='plotly_white')
    fig4.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("---")

    # --- Viz 4: Scatter ---
    st.subheader("4. GDP per Person vs. Unemployment Rate (2022)")
    
    v4_col1, v4_col2 = st.columns([1, 2])
    
    with v4_col1:
        st.markdown("#### Wealth vs. Jobs")
        st.write("""
        This scatter plot tests the hypothesis that "richer countries have lower unemployment". The visual evidence suggests **no strong linear correlation**.
        
        High-income nations often maintain moderate unemployment due to frictional factors, while low-income nations may report low unemployment due to "survival work".
        
        **Key Takeaway:** Economic growth alone is not a silver bullet. As countries develop, unemployment might initially *rise* as workers can afford to search for better jobs.
        """)

    with v4_col2:
        fig5 = px.scatter(df_2022, x=col_gdp, 
                          y=col_unemp_total,
                          color="Continent",
                          hover_name="Country Name",
                          log_x=True,
                          title="GDP per Person Employed vs. Unemployment Rate (2022)",
                          template='plotly_white')
        st.plotly_chart(fig5, use_container_width=True)
    
    st.markdown("---")

    # --- Viz 5: Philippines Case Study ---
    st.subheader("5. Philippines Case Study: Services vs. Vulnerable Employment")
    
    v5_col1, v5_col2 = st.columns([2, 1])
    
    with v5_col1:
        phl_df = df[df['Country'] == 'PHL'].sort_values('Year')
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(x=phl_df['Year'], y=phl_df[col_serv],
                            mode='lines', name='Employment in Services'), secondary_y=False)
        fig3.add_trace(go.Scatter(x=phl_df['Year'], y=phl_df[col_vuln],
                            mode='lines', name='Vulnerable Employment', line=dict(dash='dash')), secondary_y=True)
        fig3.update_layout(title='Philippines: Services vs. Vulnerable Employment (1991-2023)',
                           xaxis_title='Year',
                           template='plotly_white',
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig3.update_yaxes(title_text="Employment in Services (%)", secondary_y=False)
        fig3.update_yaxes(title_text="Vulnerable Employment (%)", secondary_y=True)
        st.plotly_chart(fig3, use_container_width=True)

    with v5_col2:
        st.markdown("#### Structural Transformation")
        st.write("""
        **RQ4:** The Philippines provides a concrete example of progress toward SDG 8.3. 
        
        The chart demonstrates a **strong inverse correlation**: as the economy transitioned toward services (rising from ~40% to nearly 60%), vulnerable employment steadily declined.
        
        This confirms that structural transformation has been a primary driver of improved job quality, moving workers from precarious informal roles into more stable wage employment.
        """)
    
    st.markdown("---")

    # --- Viz 6: Correlation Matrix ---
    st.subheader("6. Correlation Matrix of Key Indicators")
    st.write("""
    This matrix statistically validates the "interwoven" nature of these challenges. 
    *   **Services vs. Vulnerable (-0.83):** Reinforces that expanding the service sector reduces labor vulnerability.
    *   **Youth vs. Total Unemployment (0.94):** Confirms that youth outcomes are inextricably tied to general labor market health.
    *   **GDP vs. Unemployment (-0.01):** Reinforces that economic growth alone is not a silver bullet for job creation.
    """)
    
    cols = [col_unemp_total, col_unemp_youth, col_gdp, col_vuln, col_serv]
    cols_renamed = {
        col_unemp_total: 'Total Unemployment',
        col_unemp_youth: 'Youth Unemployment',
        col_gdp: 'GDP per Person',
        col_vuln: 'Vulnerable Emp.',
        col_serv: 'Services Emp.'
    }
    corr_df = countries_df[cols].rename(columns=cols_renamed)
    corr_matrix = corr_df.corr()
    fig6 = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix of Key Indicators",
                     color_continuous_scale='RdBu_r', origin='lower')
    st.plotly_chart(fig6, use_container_width=True)
