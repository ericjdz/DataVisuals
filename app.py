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
    page_title='Charting the Path to Decent Work',
    page_icon='📊',
    layout='wide'
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
        st.error(f'Error loading data: {e}')
        return None

    # 2. On-the-fly Joining
    # Merge on country and year
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
    
    # Filter out entries without a continent (usually aggregates or invalid codes)
    countries_df = df[df['Continent'].notna()].copy()

    # --- Column Renaming ---
    # Define original column names from the CSVs
    col_map = {
        # Social Protection & Labor
        'Unemployment, total (% of total labor force) (modeled ILO estimate)': 'Total Unemployment (%)',
        'Unemployment, youth total (% of total labor force ages 15-24) (modeled ILO estimate)': 'Youth Unemployment (%)',
        'GDP per person employed (constant 2021 PPP $)': 'GDP per Person ($)',
        'Vulnerable employment, total (% of total employment) (modeled ILO estimate)': 'Vulnerable Employment (%)',
        'Employment in services (% of total employment) (modeled ILO estimate)': 'Services Employment (%)',
        'Employment in agriculture (% of total employment) (modeled ILO estimate)': 'Agriculture Employment (%)',
        'Employment in industry (% of total employment) (modeled ILO estimate)': 'Industry Employment (%)',
        'Share of youth not in education, employment or training, total (% of youth population)  (modeled ILO estimate)': 'NEET Rate (%)',
        'Unemployment, male (% of male labor force) (modeled ILO estimate)': 'Male Unemployment (%)',
        'Labor force, total': 'Labor Force Size',
        
        # Gender
        'Unemployment, female (% of female labor force) (modeled ILO estimate)': 'Female Unemployment (%)',
        'Labor force participation rate, female (% of female population ages 15+) (modeled ILO estimate)': 'Female Labor Force Part. (%)',
        
        # Education
        'Literacy rate, youth total (% of people ages 15-24)': 'Youth Literacy Rate (%)',
        'School enrollment, tertiary (% gross)': 'Tertiary Enrollment (%)'
    }

    countries_df.rename(columns=col_map, inplace=True)
    
    return countries_df

# Load data
df = load_and_process_data()

if df is not None:
    # --- Sidebar ---
    with st.sidebar:
        st.header('About the Project')
        st.markdown('''
        **Authors:**
        *   De Guzman, Eric Julian
        *   Fermin, Raudmon Yvhan
        *   Palanog, Alexandra Antonette
        
        *University of Santo Tomas*
        ''')
        st.info('Data Source: World Bank & ILO (1991-2023)')
        st.markdown('---')
        
        st.header('Global Filters')
        
        # Continent Filter
        all_continents = sorted(df['Continent'].unique())
        selected_continents = st.multiselect('Select Continents', all_continents, default=all_continents)
        
        # Year Filter (Global)
        min_year = int(df['Year'].min())
        max_year = int(df['Year'].max())
        selected_year = st.slider('Select Year (for Map & Snapshots)', min_value=min_year, max_value=max_year, value=2022)

    # Filter data based on selection
    filtered_df = df[df['Continent'].isin(selected_continents)]
    year_df = filtered_df[filtered_df['Year'] == selected_year]

    # --- Main App ---
    st.title('Charting the Path to Decent Work')
    st.markdown('### A Visual Exploration of Progress and Disparities in SDG 8')

    # Tabs for better organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Global Trends (RQ1 & RQ2)', 'Wealth & Productivity (RQ3)', 'Structural Transformation (RQ4)', 'Synthesis & Correlations', 'New Insights (Gender/Edu)'])

    # --- TAB 1: GLOBAL TRENDS (RQ1 & RQ2) ---
    with tab1:
        st.subheader(f'Global Snapshot ({selected_year})')
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        avg_unemp = year_df['Total Unemployment (%)'].mean()
        avg_youth_unemp = year_df['Youth Unemployment (%)'].mean()
        avg_neet = year_df['NEET Rate (%)'].mean()
        med_gdp = year_df['GDP per Person ($)'].median()
        
        col1.metric('Avg Total Unemployment', f'{avg_unemp:.1f}%')
        col2.metric('Avg Youth Unemployment', f'{avg_youth_unemp:.1f}%', delta=f'{avg_youth_unemp - avg_unemp:.1f}% Gap', delta_color='inverse')
        col3.metric('Avg NEET Rate', f'{avg_neet:.1f}%' if pd.notna(avg_neet) else 'N/A')
        col4.metric('Median GDP/Person', f'')

        st.markdown('---')

        # Viz 1: Global Trends
        st.subheader('1. Global Unemployment Trends (1991-2023)')
        
        v1_col1, v1_col2 = st.columns([1, 2])
        
        with v1_col1:
            st.markdown('#### The Youth Deficit')
            st.caption('**Chart Type:** Time-series Line Chart | **RQ1 & RQ2**')
            st.write('''
            The analysis reveals that global labor markets are highly sensitive to macro-economic shocks (2008 Crisis, COVID-19). 
            
            However, the most critical insight regarding SDG 8.6 is the persistent **youth deficit.** Youth unemployment has remained consistently double the total rate (approx. 15-16% vs. 7-8%) for over thirty years. 
            
            The last in, first out phenomenon is clearly visible here, where youth unemployment spikes more sharply during crises and recovers more slowly.
            ''')

        with v1_col2:
            # Group by Year and calculate means
            trend_df = filtered_df.groupby('Year')[['Total Unemployment (%)', 'Youth Unemployment (%)']].mean().reset_index()
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=trend_df['Year'], y=trend_df['Total Unemployment (%)'],
                                mode='lines', name='Total Unemployment'))
            fig1.add_trace(go.Scatter(x=trend_df['Year'], y=trend_df['Youth Unemployment (%)'],
                                mode='lines', name='Youth Unemployment'))
            
            # Add crisis annotations
            fig1.add_vrect(x0=2008, x1=2009, fillcolor='gray', opacity=0.2, layer='below', line_width=0, annotation_text='2008 Crisis', annotation_position='top left')
            fig1.add_vrect(x0=2020, x1=2021, fillcolor='gray', opacity=0.2, layer='below', line_width=0, annotation_text='COVID-19', annotation_position='top left')
            
            fig1.update_layout(xaxis_title='Year', yaxis_title='Rate (%)', template='plotly_white', hovermode='x unified')
            st.plotly_chart(fig1, use_container_width=True)

        st.markdown('---')

        # Viz 2: Map
        st.subheader(f'2. Global Unemployment Map ({selected_year})')
        st.caption('**Chart Type:** Choropleth Map | **RQ1**')
        st.write('''
        A geographic heatmap showing the intensity of total unemployment rates worldwide. Unemployment is not randomly distributed; it often clusters regionally.
        
        **Insight:** While high unemployment is a clear sign of distress, low unemployment in developing regions (visible in parts of Africa) should be interpreted with caution. It often reflects high informality and survival employment, where individuals cannot afford to be unemployed.
        ''')
        
        fig_map = px.choropleth(year_df, locations='Country',
                            color='Total Unemployment (%)',
                            hover_name='Country Name',
                            color_continuous_scale=px.colors.sequential.Plasma,
                            template='plotly_white')
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

    # --- TAB 2: WEALTH & PRODUCTIVITY (RQ3) ---
    with tab2:
        st.subheader('3. Economic Productivity by Region')
        
        col_r1, col_r2 = st.columns([2, 1])
        
        with col_r1:
            # Viz 3: Box Plot
            fig_box = px.box(year_df, x='Continent', y='GDP per Person ($)', 
                          points='all', 
                          hover_name='Country Name',
                          title=f'Economic Productivity by Region ({selected_year})',
                          log_y=True,
                          color='Continent',
                          template='plotly_white')
            st.plotly_chart(fig_box, use_container_width=True)
            
        with col_r2:
            st.markdown('#### Regional Disparities')
            st.caption('**Chart Type:** Box Plot (Log Scale) | **RQ3**')
            st.write('''
            Figure highlights the immense challenge in achieving SDG 8.2. While North America and Europe exhibit high median productivity, Africa and Asia show significantly lower medians and extreme outliers. 
            
            The extreme outliers in Asia and Africa likely represent developed or oil-rich nations, masking the lower productivity of the majority. This suggests that regional averages are insufficient for understanding local economic realities.
            ''')

        st.markdown('---')
        
        # Viz 4: Animated Scatter
        st.subheader('4. Wealth vs. Jobs: Evolution Over Time')
        
        v4_col1, v4_col2 = st.columns([1, 2])
        
        with v4_col1:
            st.markdown('#### Wealth vs. Jobs')
            st.caption('**Chart Type:** Animated Scatter Plot | **RQ3**')
            st.write('''
            This scatter plot tests the hypothesis that richer countries have lower unemployment. The visual evidence suggests **no strong linear correlation**.
            
            High-income nations often maintain moderate unemployment due to frictional factors, while low-income nations may report low unemployment due to survival work.
            
            **Key Takeaway:** Economic growth alone is not a silver bullet. As countries develop, unemployment might initially *rise* as workers can afford to search for better jobs.
            ''')
            st.info('💡 **Interaction:** Press Play to see how countries have moved over the last 30 years.')

        with v4_col2:
            # For animation, we need the full filtered dataset, not just the selected year
            # We drop NaNs in the specific columns to avoid animation errors
            anim_df = filtered_df.dropna(subset=['GDP per Person ($)', 'Total Unemployment (%)', 'Continent', 'Country Name', 'Labor Force Size'])
            anim_df = anim_df.sort_values('Year')
            
            fig_anim = px.scatter(anim_df, x='GDP per Person ($)', y='Total Unemployment (%)',
                                  animation_frame='Year', animation_group='Country Name',
                                  size='Labor Force Size', color='Continent', 
                                  hover_name='Country Name',
                                  log_x=True, size_max=60,
                                  range_x=[1000, 200000], range_y=[0, 40],
                                  template='plotly_white')
            
            st.plotly_chart(fig_anim, use_container_width=True)

    # --- TAB 3: STRUCTURAL TRANSFORMATION (RQ4) ---
    with tab3:
        st.subheader('5. Structural Transformation & Vulnerability')
        
        # Country Selector
        countries_list = sorted(filtered_df['Country Name'].unique())
        default_ix = countries_list.index('Philippines') if 'Philippines' in countries_list else 0
        selected_country = st.selectbox('Select a Country to Analyze', countries_list, index=default_ix)
        
        # Filter for country
        country_data = df[df['Country Name'] == selected_country].sort_values('Year')
        
        col_d1, col_d2 = st.columns([2, 1])
        
        with col_d1:
            # Dual Axis Chart (Services vs Vulnerable)
            fig_vuln = make_subplots(specs=[[{'secondary_y': True}]])
            fig_vuln.add_trace(go.Scatter(x=country_data['Year'], y=country_data['Services Employment (%)'],
                                mode='lines', name='Employment in Services'), secondary_y=False)
            fig_vuln.add_trace(go.Scatter(x=country_data['Year'], y=country_data['Vulnerable Employment (%)'],
                                mode='lines', name='Vulnerable Employment', line=dict(dash='dash')), secondary_y=True)
            
            fig_vuln.update_layout(title=f'{selected_country}: Services vs. Vulnerable Employment',
                               xaxis_title='Year',
                               template='plotly_white',
                               legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
            fig_vuln.update_yaxes(title_text='Employment in Services (%)', secondary_y=False)
            fig_vuln.update_yaxes(title_text='Vulnerable Employment (%)', secondary_y=True)
            st.plotly_chart(fig_vuln, use_container_width=True)

        with col_d2:
            st.markdown('#### Structural Transformation')
            st.caption('**Chart Type:** Dual-Axis Line Chart | **RQ4**')
            
            if selected_country == 'Philippines':
                st.write('''
                The Philippines provides a concrete example of progress toward SDG 8.3. 
                
                The chart demonstrates a **strong inverse correlation**: as the economy transitioned toward services (rising from ~40% to nearly 60%), vulnerable employment steadily declined.
                
                This confirms that structural transformation has been a primary driver of improved job quality, moving workers from precarious informal roles into more stable wage employment.
                ''')
            else:
                st.write(f'''
                Analyze the relationship between the Service Sector and Vulnerable Employment in **{selected_country}**.
                
                **Theory:** As economies modernize (structural transformation), we expect to see a rise in Service employment and a decline in Vulnerable employment (informal work).
                
                *Does this trend hold true for {selected_country}?*
                ''')

        # Extra Chart: Sector Breakdown
        st.markdown('#### Detailed Sector Breakdown')
        sector_cols = ['Agriculture Employment (%)', 'Industry Employment (%)', 'Services Employment (%)']
        if country_data[sector_cols].notna().any().any():
            fig_sector = px.area(country_data, x='Year', y=sector_cols,
                                    title=f'Employment Distribution in {selected_country}',
                                    labels={'value': 'Percentage (%)', 'variable': 'Sector'},
                                    template='plotly_white')
            st.plotly_chart(fig_sector, use_container_width=True)

    # --- TAB 4: SYNTHESIS ---
    with tab4:
        st.subheader('6. Correlation Matrix of Key Indicators')
        st.caption('**Chart Type:** Heatmap | **RQ1-RQ4 (Synthesis)**')
        
        col_s1, col_s2 = st.columns([1, 2])
        
        with col_s1:
            st.write('''
            This matrix statistically validates the interwoven nature of these challenges. 
            *   **Services vs. Vulnerable (-0.83):** Reinforces that expanding the service sector reduces labor vulnerability.
            *   **Youth vs. Total Unemployment (0.94):** Confirms that youth outcomes are inextricably tied to general labor market health.
            *   **GDP vs. Unemployment (-0.01):** Reinforces that economic growth alone is not a silver bullet for job creation.
            ''')
        
        with col_s2:
            cols = ['Total Unemployment (%)', 'Youth Unemployment (%)', 'GDP per Person ($)', 'Vulnerable Employment (%)', 'Services Employment (%)']
            corr_df = filtered_df[cols]
            corr_matrix = corr_df.corr()
            fig6 = px.imshow(corr_matrix, text_auto=True, title='Correlation Matrix of Key Indicators',
                            color_continuous_scale='RdBu_r', origin='lower')
            st.plotly_chart(fig6, use_container_width=True)

    # --- TAB 5: NEW INSIGHTS ---
    with tab5:
        st.header('Further Exploration: Gender & Education')
        st.write('Expanding the analysis beyond the core research questions to explore Gender Parity and Education outcomes.')
        
        st.subheader('The Gender Divide')
        # Prepare data for Gender Gap
        gender_cols = ['Male Unemployment (%)', 'Female Unemployment (%)']
        gender_df = year_df.dropna(subset=gender_cols)
        gender_agg = gender_df.groupby('Continent')[gender_cols].mean().reset_index()
        gender_melt = gender_agg.melt(id_vars='Continent', var_name='Metric', value_name='Rate')
        
        fig_gender = px.bar(gender_melt, x='Continent', y='Rate', color='Metric',
                            barmode='group', title=f'Male vs. Female Unemployment by Region ({selected_year})',
                            template='plotly_white', color_discrete_sequence=['#1f77b4', '#e377c2'])
        st.plotly_chart(fig_gender, use_container_width=True)
        
        st.markdown('---')
        
        st.subheader('Education Paradox')
        st.write('Does higher education enrollment correlate with lower youth unemployment? In some regions, high enrollment might coexist with high unemployment due to skills mismatch.')
        
        # Scatter: Tertiary Enrollment vs Youth Unemployment
        edu_scatter_df = year_df.dropna(subset=['Tertiary Enrollment (%)', 'Youth Unemployment (%)'])
        
        if not edu_scatter_df.empty:
            fig_edu = px.scatter(edu_scatter_df, x='Tertiary Enrollment (%)', y='Youth Unemployment (%)',
                                 color='Continent', hover_name='Country Name',
                                 trendline='ols',
                                 title=f'Tertiary Enrollment vs. Youth Unemployment ({selected_year})',
                                 template='plotly_white')
            st.plotly_chart(fig_edu, use_container_width=True)

