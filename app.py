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
    
    # Calculate Estimated Counts (Frequency)
    # Total Unemployed = (Total Unemployment % / 100) * Labor Force Size
    countries_df['Total Unemployed Count'] = (countries_df['Total Unemployment (%)'] / 100) * countries_df['Labor Force Size']
    
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
        col4.metric('Median GDP/Person', f'${med_gdp:,.0f}' if pd.notna(med_gdp) else 'N/A')

        st.markdown('---')

        # Viz 1: Global Trends
        st.subheader('1. Global Unemployment Trends (1991-2023)')
        
        # Add View Mode Selector
        view_mode = st.radio("View Mode:", ["Global/Regional Snapshot (Detailed)", "Comparative Trends (Multi-Country)"], horizontal=True)
        
        if view_mode == "Global/Regional Snapshot (Detailed)":
            # Layout: Chart (Left) - Stats (Right)
            v1_col1, v1_col2 = st.columns([3, 1])
            
            with v1_col1:
                # Group by Year and calculate means and sums
                trend_df = filtered_df.groupby('Year').agg({
                    'Total Unemployment (%)': 'mean',
                    'Youth Unemployment (%)': 'mean',
                    'Total Unemployed Count': 'sum'
                }).reset_index()
                
                # Create Dual-Axis Chart to show Rate vs Volume
                fig1 = make_subplots(specs=[[{'secondary_y': True}]])
                
                # Trace 1: Total Unemployed Count (Bars - Background Context)
                fig1.add_trace(go.Bar(x=trend_df['Year'], y=trend_df['Total Unemployed Count'],
                                    name='Total Unemployed Population (Count)', marker_color='lightgray', opacity=0.5), secondary_y=True)
                
                # Trace 2: Rates (Lines - Foreground Focus)
                fig1.add_trace(go.Scatter(x=trend_df['Year'], y=trend_df['Total Unemployment (%)'],
                                    mode='lines', name='Total Unemployment Rate (%)', line=dict(color='blue', width=3)), secondary_y=False)
                fig1.add_trace(go.Scatter(x=trend_df['Year'], y=trend_df['Youth Unemployment (%)'],
                                    mode='lines', name='Youth Unemployment Rate (%)', line=dict(color='red', width=3)), secondary_y=False)
                
                # Add crisis annotations
                fig1.add_vrect(x0=2008, x1=2009, fillcolor='gray', opacity=0.1, layer='below', line_width=0, annotation_text='2008 Crisis', annotation_position='top left')
                fig1.add_vrect(x0=2020, x1=2021, fillcolor='gray', opacity=0.1, layer='below', line_width=0, annotation_text='COVID-19', annotation_position='top left')
                
                fig1.update_layout(
                    title='Global Trends: Unemployment Rates vs. Absolute Counts',
                    xaxis_title='Year',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    template='plotly_white',
                    hovermode='x unified',
                    height=500
                )
                
                fig1.update_yaxes(title_text='Unemployment Rate (%)', secondary_y=False)
                fig1.update_yaxes(title_text='Total Unemployed (Count)', secondary_y=True, showgrid=False)
                
                st.plotly_chart(fig1, use_container_width=True)

            with v1_col2:
                st.markdown('#### Rate vs. Volume')
                
                # Calculate stats for the whole period
                max_youth_unemp = trend_df['Youth Unemployment (%)'].max()
                max_youth_year = trend_df.loc[trend_df['Youth Unemployment (%)'].idxmax(), 'Year']
                
                total_unemployed_latest = trend_df.iloc[-1]['Total Unemployed Count']
                total_unemployed_peak = trend_df['Total Unemployed Count'].max()
                
                st.metric("Peak Youth Rate", f"{max_youth_unemp:.1f}%", f"Year: {int(max_youth_year)}")
                st.metric("Latest Total Unemployed", f"{total_unemployed_latest/1e6:.1f}M")
                st.metric("Peak Total Unemployed", f"{total_unemployed_peak/1e6:.1f}M")
                
                st.warning('''
                **Interpretation Note:**
                While **Youth Unemployment Rates** (Red Line) are drastically higher, they represent a smaller subset of the population.
                
                The **Total Unemployment Rate** (Blue Line) appears lower because it is "diluted" by the larger, more stable Adult workforce.
                
                The **Grey Bars** show the actual number of unemployed people, which continues to rise due to population growth even when rates stabilize.
                ''')
        
        else: # Comparative Trends
            st.markdown("#### Compare Trends Across Countries/Continents")
            
            comp_col1, comp_col2 = st.columns([1, 3])
            
            with comp_col1:
                comp_type = st.selectbox("Compare By:", ["Continent", "Country"])
                comp_metric = st.selectbox("Metric:", ["Total Unemployment (%)", "Youth Unemployment (%)", "Total Unemployed Count"])
                
                if comp_type == "Continent":
                    options = sorted(df['Continent'].unique())
                    default_sel = options[:3] if len(options) >= 3 else options
                    selection = st.multiselect("Select Continents:", options, default=default_sel)
                    comp_data = df[df['Continent'].isin(selection)]
                    group_col = 'Continent'
                else:
                    options = sorted(df['Country Name'].unique())
                    default_sel = ['United States', 'China', 'India']
                    # Filter defaults to exist in data
                    default_sel = [x for x in default_sel if x in options]
                    selection = st.multiselect("Select Countries:", options, default=default_sel)
                    comp_data = df[df['Country Name'].isin(selection)]
                    group_col = 'Country Name'
            
            with comp_col2:
                if not selection:
                    st.warning("Please select at least one entity to compare.")
                else:
                    # Aggregate if comparing continents, otherwise just take country data
                    if comp_type == "Continent":
                        # Need to aggregate properly. For rates, mean is okay-ish for visualization, but weighted average is better. 
                        # For simplicity in this viz tool, we'll use mean of rates, sum of counts.
                        if "Count" in comp_metric:
                            comp_trend = comp_data.groupby(['Year', 'Continent'])[comp_metric].sum().reset_index()
                        else:
                            comp_trend = comp_data.groupby(['Year', 'Continent'])[comp_metric].mean().reset_index()
                    else:
                        # Country level - just filter
                        if "Count" in comp_metric:
                             # Ensure count is calculated for all rows if not already
                             comp_data['Total Unemployed Count'] = (comp_data['Total Unemployment (%)'] / 100) * comp_data['Labor Force Size']
                        comp_trend = comp_data[['Year', 'Country Name', comp_metric]].copy()
                    
                    fig_comp = px.line(comp_trend, x='Year', y=comp_metric, color=group_col,
                                     title=f'Comparative Trend: {comp_metric}',
                                     template='plotly_white')
                    st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown('---')

        # Viz 2: Map
        st.subheader(f'2. Global Unemployment Map ({selected_year})')
        
        v2_col1, v2_col2 = st.columns([3, 1])
        
        with v2_col1:
            fig_map = px.choropleth(year_df, locations='Country',
                                color='Total Unemployment (%)',
                                hover_name='Country Name',
                                color_continuous_scale=px.colors.sequential.Plasma,
                                template='plotly_white')
            fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_map, use_container_width=True)
            
        with v2_col2:
            st.markdown('#### Regional Stats')
            # Top 5 Highest Unemployment
            top_5 = year_df.nlargest(5, 'Total Unemployment (%)')[['Country Name', 'Total Unemployment (%)']]
            st.write("**Highest Unemployment:**")
            for i, row in top_5.iterrows():
                st.write(f"{row['Country Name']}: **{row['Total Unemployment (%)']:.1f}%**")
            
            st.write("---")
            # Regional Averages
            reg_avg = year_df.groupby('Continent')['Total Unemployment (%)'].mean().sort_values(ascending=False)
            st.write("**Avg by Continent:**")
            st.dataframe(reg_avg.to_frame().style.format("{:.1f}%"), use_container_width=True)

    # --- TAB 2: WEALTH & PRODUCTIVITY (RQ3) ---
    with tab2:
        st.subheader('3. Economic Productivity by Region')
        
        col_r1, col_r2 = st.columns([3, 1])
        
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
            st.markdown('#### Regional Stats')
            
            # Calculate Median GDP by Continent
            med_gdp_cont = year_df.groupby('Continent')['GDP per Person ($)'].median().sort_values(ascending=False)
            
            st.write("**Median GDP ($):**")
            st.dataframe(med_gdp_cont.to_frame().style.format("${:,.0f}"), use_container_width=True)
            
            # Richest and Poorest in selection
            if not year_df.empty:
                richest = year_df.loc[year_df['GDP per Person ($)'].idxmax()]
                poorest = year_df.loc[year_df['GDP per Person ($)'].idxmin()]
                
                st.metric("Highest GDP", f"${richest['GDP per Person ($)']:,.0f}", richest['Country Name'])
                st.metric("Lowest GDP", f"${poorest['GDP per Person ($)']:,.0f}", poorest['Country Name'])

        st.markdown('---')
        
        # Viz 4: Animated Scatter
        st.subheader('4. Wealth vs. Jobs: Evolution Over Time')
        
        v4_col1, v4_col2 = st.columns([3, 1])
        
        with v4_col1:
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

        with v4_col2:
            st.markdown('#### Correlation Analysis')
            
            # Calculate correlation for the selected year
            corr_df = year_df.dropna(subset=['GDP per Person ($)', 'Total Unemployment (%)'])
            if not corr_df.empty:
                correlation = corr_df['GDP per Person ($)'].corr(corr_df['Total Unemployment (%)'])
                st.metric(f"Correlation (r) in {selected_year}", f"{correlation:.2f}")
                
                st.write(f"**Sample Size:** {len(corr_df)} countries")
                
                if abs(correlation) < 0.3:
                    strength = "Weak"
                elif abs(correlation) < 0.7:
                    strength = "Moderate"
                else:
                    strength = "Strong"
                
                st.info(f"**Interpretation:** {strength} {'negative' if correlation < 0 else 'positive'} correlation.")
            
            st.caption('**Chart Type:** Animated Scatter Plot | **RQ3**')
            st.write('''
            This scatter plot tests the hypothesis that richer countries have lower unemployment.
            
            **Key Takeaway:** Economic growth alone is not a silver bullet. As countries develop, unemployment might initially *rise* as workers can afford to search for better jobs.
            ''')

    # --- TAB 3: STRUCTURAL TRANSFORMATION (RQ4) ---
    with tab3:
        st.subheader('5. Structural Transformation & Vulnerability')
        
        # Country Selector
        countries_list = sorted(filtered_df['Country Name'].unique())
        default_ix = countries_list.index('Philippines') if 'Philippines' in countries_list else 0
        selected_country = st.selectbox('Select a Country to Analyze', countries_list, index=default_ix)
        
        # Filter for country
        country_data = df[df['Country Name'] == selected_country].sort_values('Year')
        
        col_d1, col_d2 = st.columns([3, 1])
        
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
            st.markdown('#### Country Stats')
            
            if not country_data.empty:
                latest_yr = country_data['Year'].max()
                latest_row = country_data[country_data['Year'] == latest_yr].iloc[0]
                
                earliest_yr = country_data['Year'].min()
                earliest_row = country_data[country_data['Year'] == earliest_yr].iloc[0]
                
                serv_change = latest_row['Services Employment (%)'] - earliest_row['Services Employment (%)']
                vuln_change = latest_row['Vulnerable Employment (%)'] - earliest_row['Vulnerable Employment (%)']
                
                st.metric("Services Emp (Latest)", f"{latest_row['Services Employment (%)']:.1f}%", f"{serv_change:+.1f}% since {earliest_yr}")
                st.metric("Vulnerable Emp (Latest)", f"{latest_row['Vulnerable Employment (%)']:.1f}%", f"{vuln_change:+.1f}% since {earliest_yr}", delta_color='inverse')

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
        
        with col_s2:
            cols = ['Total Unemployment (%)', 'Youth Unemployment (%)', 'GDP per Person ($)', 'Vulnerable Employment (%)', 'Services Employment (%)']
            corr_df = filtered_df[cols]
            corr_matrix = corr_df.corr()
            fig6 = px.imshow(corr_matrix, text_auto=True, title='Correlation Matrix of Key Indicators',
                            color_continuous_scale='RdBu_r', origin='lower')
            st.plotly_chart(fig6, use_container_width=True)

        with col_s1:
            st.markdown('#### Key Relationships')
            
            # Find strongest correlations (excluding self-correlation)
            # Mask diagonal
            mask = pd.DataFrame(False, index=corr_matrix.index, columns=corr_matrix.columns)
            for i in range(len(corr_matrix)):
                mask.iloc[i, i] = True
            
            masked_corr = corr_matrix.mask(mask)
            
            strongest_pos = masked_corr.stack().idxmax()
            max_val = masked_corr.max().max()
            
            strongest_neg = masked_corr.stack().idxmin()
            min_val = masked_corr.min().min()
            
            st.metric("Strongest Positive", f"{max_val:.2f}", f"{strongest_pos[0]} & {strongest_pos[1]}")
            st.metric("Strongest Negative", f"{min_val:.2f}", f"{strongest_neg[0]} & {strongest_neg[1]}")
            
            st.write('''
            **Interpretation:**
            *   **Services vs. Vulnerable:** Strong negative correlation confirms that expanding the service sector reduces labor vulnerability.
            *   **Youth vs. Total Unemployment:** Very high positive correlation confirms that youth outcomes are inextricably tied to general labor market health.
            ''')

    # --- TAB 5: NEW INSIGHTS ---
    with tab5:
        st.header('Further Exploration: Gender & Education')
        st.write('Expanding the analysis beyond the core research questions to explore Gender Parity and Education outcomes.')
        
        st.subheader('The Gender Divide')
        
        col_g1, col_g2 = st.columns([3, 1])
        
        with col_g1:
            # Prepare data for Gender Gap
            gender_cols = ['Male Unemployment (%)', 'Female Unemployment (%)']
            
            # 1. Animated Bar Chart (Yearly Evolution)
            # We need the full dataset for animation, not just the selected year
            gender_anim_df = filtered_df.dropna(subset=gender_cols + ['Continent', 'Year'])
            # Aggregate by Continent and Year
            gender_anim_agg = gender_anim_df.groupby(['Continent', 'Year'])[gender_cols].mean().reset_index()
            gender_anim_melt = gender_anim_agg.melt(id_vars=['Continent', 'Year'], var_name='Metric', value_name='Rate')
            
            fig_gender = px.bar(gender_anim_melt, x='Continent', y='Rate', color='Metric',
                                animation_frame='Year', barmode='group',
                                title='Male vs. Female Unemployment by Region (Animated)',
                                range_y=[0, gender_anim_melt['Rate'].max() * 1.1], # Fix y-axis
                                template='plotly_white', color_discrete_sequence=['#1f77b4', '#e377c2'])
            st.plotly_chart(fig_gender, use_container_width=True)
            
            st.markdown("#### Global Gender Gap Evolution")
            # 2. Line Chart for Global Gap
            global_gender = filtered_df.groupby('Year')[gender_cols].mean().reset_index()
            global_gender['Gender Gap'] = global_gender['Female Unemployment (%)'] - global_gender['Male Unemployment (%)']
            
            fig_gap = px.line(global_gender, x='Year', y='Gender Gap',
                              title='Global Average Gender Unemployment Gap (Female - Male)',
                              markers=True, template='plotly_white')
            fig_gap.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_gap, use_container_width=True)
            
        with col_g2:
            st.markdown('#### Gender Gap Stats')
            if not gender_anim_agg.empty:
                # Get stats for the currently selected year (from the global slider)
                curr_stats = gender_anim_agg[gender_anim_agg['Year'] == selected_year].copy()
                if not curr_stats.empty:
                    curr_stats['Gap'] = curr_stats['Female Unemployment (%)'] - curr_stats['Male Unemployment (%)']
                    largest_gap_cont = curr_stats.loc[curr_stats['Gap'].idxmax()]
                    
                    st.write(f"**Stats for {selected_year}:**")
                    st.metric("Largest Gap Region", f"{largest_gap_cont['Continent']}", f"{largest_gap_cont['Gap']:.1f}% Gap")
                
                # Overall trend stats
                start_gap = global_gender.iloc[0]['Gender Gap']
                end_gap = global_gender.iloc[-1]['Gender Gap']
                st.metric("Global Gap Change", f"{end_gap:.2f}%", f"{end_gap - start_gap:+.2f}% since 1991")
                
                st.info("A positive gap means Female unemployment is higher.")

        st.markdown('---')
        
        st.subheader('Education Paradox')
        
        col_e1, col_e2 = st.columns([3, 1])
        
        with col_e1:
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
        
        with col_e2:
            st.markdown('#### Correlation')
            if not edu_scatter_df.empty:
                corr_edu = edu_scatter_df['Tertiary Enrollment (%)'].corr(edu_scatter_df['Youth Unemployment (%)'])
                st.metric("Correlation (r)", f"{corr_edu:.2f}")
                
                if abs(corr_edu) < 0.3:
                    st.info("Weak correlation suggests education alone doesn't guarantee jobs.")
                elif corr_edu > 0:
                    st.warning("Positive correlation! Higher education might be linked to higher youth unemployment in some contexts (e.g., 'wait unemployment').")
                else:
                    st.success("Negative correlation. Higher education is linked to lower unemployment.")

