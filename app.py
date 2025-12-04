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

    df['Continent'] = df['Country'].apply(get_continent)
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
    st.title("Charting the Path to Decent Work")
    st.subheader("A Visual Exploration of Progress and Disparities in SDG 8")
    st.markdown("""
    **Authors:**
    *   De Guzman, Eric Julian
    *   Fermin, Raudmon Yvhan
    *   Palanog, Alexandra Antonette
    
    *University of Santo Tomas, Manila, Philippines*
    """)

    tab1, tab2, tab3 = st.tabs(["Project Background", "Visualizations & Analysis", "Conclusion"])

    with tab1:
        st.header("1. Project Background")
        st.write("""
        In 2024, the International Labour Organization projected that the global unemployment rate would reach 5.2%, representing over 200 million people without work (International Labour Organization, 2024). However, this figure only scratches the surface of a more complex global challenge: the persistent deficit of decent work opportunities. This study, Charting the Path to Decent Work, provides a data-driven analysis of this multifaceted issue. The investigation extends beyond the total unemployment rate to examine disproportionately high rates among youth, the prevalence of insecure or "vulnerable" employment, and deep-seated gender disparities within the labor market (World Bank, 2025; International Labour Organization, 2024).

        The analysis directly aligns with UN Sustainable Development Goal (SDG) 8: Decent Work and Economic Growth, which emphasizes “full and productive employment and decent work for all” (United Nations, 2023). Targets include boosting productivity, safeguarding labor rights, and reducing the share of youth not in employment, education, or training (United Nations, 2023). Employment also intersects with other SDGs: poverty (SDG 1), education (SDG 4), and gender equality (SDG 5) (United Nations, 2023). By visualizing World Bank labor indicators, this study uncovers global trends, national disparities, and cross-cutting relationships often hidden in raw statistics, while clear and insightful visualizations make labor market challenges such as the youth–adult unemployment gap immediately clear to policymakers and the public (World Bank, 2025).
        """)

        st.header("2. Statement of the Problem")
        st.subheader("2.1 Research Questions")
        st.write("""
        Global unemployment is often reported as a single rate, but this oversimplifies complex realities. Youth joblessness, poor job quality, and productivity gaps are interwoven with national development yet remain hidden in large datasets like those of the World Bank (World Bank, 2025). Without clear, data-driven visualizations, policymakers and the public cannot fully assess progress toward SDG 8 (United Nations, 2023). To address this gap, the study transforms complex World Bank data, incorporating modeled ILO estimates for cross-country comparability, into an intuitive visual narrative that reveals trends and relationships raw numbers alone obscure (International Labour Organization, 2024; World Bank, 2025).

        Given the main problem, this study is guided by the following central research questions:
        1.  What are the global unemployment trends from 1991–2023, especially during major economic events, and which countries consistently face the most significant challenges?
        2.  How does youth unemployment compare with total unemployment worldwide, and what does this reveal about the structural challenges younger populations face in entering the labor market?
        3.  What is the relationship between a country's economic development (measured by GDP per person employed) and its levels of unemployment and vulnerable employment?
        4.  How has the Philippines’ shift toward a service-based economy shaped vulnerable employment trends over the past three decades?
        """)

        st.subheader("2.2 Objectives")
        st.write("""
        Considering the problem statement, we have set the following specific objectives:
        *   Identify and visualize the countries with the highest average rates of total and youth unemployment and their relation to SDG 8 (United Nations, 2023).
        *   Analyze the historical trend of global unemployment rates to identify patterns corresponding to major economic events (World Bank, 2025).
        *   Examine correlations between unemployment, GDP per worker, and vulnerable employment as defined in global labor and development databases (World Bank, 2025).
        *   Conduct a country-level case study on the Philippines to visualize the relationship between its economic restructuring and changes in job quality over time (World Bank, 2025).
        """)
        
        st.header("3. Background of the Dataset")
        st.write("""
        **Dataset Name & Source:** World Bank Indicators (1960–Present) – World Development Indicators, accessed via Kaggle repository.
        
        **Dataset Description:**
        Organized as a country-year panel, it enables exploration of interdisciplinary relationships through merging thematic files. Four key files are employed in this study:
        *   `social_protection_labor.csv`: Provides primary indicators for labor market health.
        *   `poverty.csv`: Provides data on unemployment’s socio-economic impacts.
        *   `education.csv`: Offers context on the relationship between education and employment.
        *   `gender.csv`: Instrumental for analyzing gendered dynamics in the labor market.
        """)

    with tab2:
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
        st.write("""
        **RQ1 & RQ2:** The analysis reveals that global labor markets are highly sensitive to macro-economic shocks, with distinct spikes visible during the 2008 financial crisis and the 2020 COVID-19 pandemic. 
        
        However, the most critical insight regarding SDG 8.6 (reducing youth not in employment) is the persistent "youth deficit." The data shows that the youth unemployment rate has remained consistently double that of the total rate (approx. 15-16% vs. 7-8%) for over thirty years. This indicates that despite global economic growth, structural barriers continue to prevent young people from accessing decent work, making them the most vulnerable demographic during economic downturns.
        """)
        
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
                           template='plotly_white')
        fig1.add_vrect(x0=2008, x1=2009, fillcolor="gray", opacity=0.2, layer="below", line_width=0, annotation_text="2008 Crisis", annotation_position="top left")
        fig1.add_vrect(x0=2020, x1=2021, fillcolor="gray", opacity=0.2, layer="below", line_width=0, annotation_text="COVID-19", annotation_position="top left")
        st.plotly_chart(fig1, use_container_width=True)

        # --- Viz 2: Productivity by Region ---
        st.subheader("2. Economic Productivity by Region (2022)")
        st.write("""
        **RQ3:** Figure 2 highlights the immense challenge in achieving SDG 8.2 (higher levels of economic productivity). While North America and Europe exhibit high median productivity with low variance, Africa and Asia show significantly lower medians and extreme outliers. 
        
        This disparity is further contextualized by the scatter plot below, which reveals a counter-intuitive finding: there is no strong linear correlation between a country's wealth and its unemployment rate. High-income nations often maintain moderate unemployment due to frictional factors, while low-income nations may report low unemployment due to the necessity of survival work. This underscores that "unemployment rate" alone is an insufficient metric for decent work in developing contexts; productivity is the missing half of the picture.
        """)
        
        df_2022 = countries_df[countries_df['Year'] == 2022]
        fig2 = px.box(df_2022, x='Continent', y=col_gdp, 
                      points="all", 
                      title='Economic Productivity by Region (2022)',
                      log_y=True,
                      color='Continent',
                      template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)

        # --- Viz 4: Map ---
        st.subheader("3. Global Unemployment Rate Map (2022)")
        st.write("""
        A geographic heatmap showing the intensity of total unemployment rates worldwide. Unemployment is not randomly distributed; it often clusters regionally. This map allows policymakers to instantly identify "hotspots" of labor market distress (such as in Southern Africa) that might be missed in a tabular format.
        """)
        fig4 = px.choropleth(df_2022, locations="Country",
                            color=col_unemp_total,
                            hover_name="Country",
                            color_continuous_scale=px.colors.sequential.Plasma,
                            title="Global Unemployment Rate (2022)",
                            template='plotly_white')
        st.plotly_chart(fig4, use_container_width=True)

        # --- Viz 5: Scatter ---
        st.subheader("4. GDP per Person vs. Unemployment Rate (2022)")
        st.write("""
        This scatter plot tests the hypothesis that "richer countries have lower unemployment". The visual evidence suggests there is no strong linear correlation between a country's wealth and its unemployment rate. 
        
        High-income nations often maintain moderate unemployment due to frictional factors, while low-income nations may report low unemployment due to the necessity of survival work. This underscores that "unemployment rate" alone is an insufficient metric for decent work in developing contexts; productivity is the missing half of the picture.
        """)
        fig5 = px.scatter(df_2022, x=col_gdp, 
                          y=col_unemp_total,
                          color="Continent",
                          hover_name="Country",
                          log_x=True,
                          title="GDP per Person Employed vs. Unemployment Rate (2022)",
                          template='plotly_white')
        st.plotly_chart(fig5, use_container_width=True)

        # --- Viz 3: Philippines Case Study ---
        st.subheader("5. Philippines Case Study: Services vs. Vulnerable Employment")
        st.write("""
        **RQ4:** The investigation into the Philippines provides a concrete example of progress toward SDG 8.3 (promoting development-oriented policies). The chart demonstrates a strong inverse correlation: as the Philippines transitioned toward a service-based economy (rising from ~40% to nearly 60% employment share), the rate of vulnerable employment steadily declined. 
        
        This confirms that for this developing nation, structural transformation has been a primary driver of improved job quality, successfully moving workers from precarious informal roles into more stable, formal wage employment.
        """)
        
        phl_df = df[df['Country'] == 'PHL'].sort_values('Year')
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(x=phl_df['Year'], y=phl_df[col_serv],
                            mode='lines', name='Employment in Services'), secondary_y=False)
        fig3.add_trace(go.Scatter(x=phl_df['Year'], y=phl_df[col_vuln],
                            mode='lines', name='Vulnerable Employment', line=dict(dash='dash')), secondary_y=True)
        fig3.update_layout(title='Philippines: Services vs. Vulnerable Employment (1991-2023)',
                           xaxis_title='Year',
                           template='plotly_white')
        fig3.update_yaxes(title_text="Employment in Services (%)", secondary_y=False)
        fig3.update_yaxes(title_text="Vulnerable Employment (%)", secondary_y=True)
        st.plotly_chart(fig3, use_container_width=True)

        # --- Viz 6: Correlation Matrix ---
        st.subheader("6. Correlation Matrix of Key Indicators")
        st.write("""
        This matrix statistically validates the "interwoven" nature of these challenges. 
        *   The **strong negative correlation (-0.83)** between Services Employment and Vulnerable Employment reinforces the global applicability of the Philippines case study, suggesting that expanding the service sector is a key pathway to reducing labor vulnerability. 
        *   The **strong positive correlation (0.94)** between Youth and Total Unemployment confirms that youth outcomes are inextricably tied to general labor market health, suggesting that SDG 8 targets cannot be achieved in isolation; policies to boost general economic growth must be paired with targeted interventions for youth and vulnerable workers.
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

    with tab3:
        st.header("7. Conclusion")
        st.subheader("Summary of Findings")
        st.write("""
        This study successfully utilized World Bank data to chart the global path toward decent work. The analysis confirms that while global unemployment rates fluctuate with economic cycles, the structural exclusion of youth is a permanent feature of the modern labor market. Furthermore, the data reveals that "decent work" is inextricably linked to economic structure; countries that successfully transition into service-oriented economies (like the Philippines) see significant reductions in vulnerable employment. However, the stark regional disparities in productivity highlight that for many developing nations, the challenge is not just creating jobs, but creating productive jobs that lift workers out of poverty.
        """)

        st.subheader("Implications for SDG 8")
        st.write("""
        The findings suggest that achieving SDG 8 requires a two-pronged approach. For developed economies, the focus must be on closing the youth unemployment gap through targeted training and entry-level integration. For developing economies, the priority must be structural transformation, shifting labor from low-productivity agriculture to higher-value services and industry, to reduce the prevalence of vulnerable employment.
        """)

        st.subheader("Recommendations & Future Exploration")
        st.write("""
        Policymakers should prioritize "active labor market policies" specifically targeting youth to break the cycle of structural exclusion. Future research should leverage the gender.csv dataset more extensively to perform a deep-dive into the "double burden" faced by young women in the labor market. Additionally, applying machine learning clustering techniques to the dataset could identify "peer groups" of countries facing similar labor challenges, allowing for more tailored development strategies.
        """)
        
        st.header("References")
        st.write("""
        *   International Labour Organization. (2024). World employment and social outlook: Trends 2024.
        *   International Labour Organization. (2024). Global employment trends for youth 2024.
        *   United Nations. (2023). Sustainable Development Goals.
        *   World Bank. (2025). World development indicators.
        *   World Bank. (2025). Making labor markets work for the youth.
        *   World Bank. (2022). Global job quality.
        *   Asian Development Bank. (2025). Expanding support for Philippines labor market reforms.
        """)
