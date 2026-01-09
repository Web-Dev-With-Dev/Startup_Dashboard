import seaborn as sns
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="StartUp Analysis")

df = pd.read_csv('https://github.com/Web-Dev-With-Dev/Startup_Dashboard/blob/master/Startup_dashboard/startup_cleaned.csv')
df.dropna(subset=['Date', 'Startup', 'Vertical', 'City', 'Investors', 'Round', 'Amount'], inplace=True)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)
df.set_index('Sr No', inplace=True)
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month


def overall_analysis():
    st.title('Overall Analysis')

    col1,col2,col3,col4 = st.columns(4)

    #total invested amount
    total = round(sum(df['Amount'])/10)
    # max amount infused in a startup
    max_funding = df.groupby('Startup')['Amount'].max().sort_values(ascending=False).head(1).values[0]/10
    # average funding
    avg_funding = round(df.groupby('Startup')['Amount'].sum().mean()/10)
    # total funded startups
    total_startups = df['Startup'].nunique()

    with col1:
        st.metric('Total', str(total) + ' Cr')
    with col2:
        st.metric('Max Funding',str(max_funding) + ' Cr')
    with col3:
        st.metric('Average Funding',str(avg_funding) + ' Cr')
    with col4:
        st.metric('Total Startups',str(total_startups))

    st.header('MoM Graph')
    selected_option = st.selectbox('Select Type',['Total','Count'],key='select_type_1')
    if selected_option == 'Total':
        temp_df = df.groupby(['year', 'month'])['Amount'].sum().reset_index()
    else:
        temp_df = df.groupby(['year', 'month'])['Amount'].count().reset_index()
    temp_df['x_axis'] = temp_df['month'].astype('str') + '-' + temp_df['year'].astype('str')

    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.plot(range(len(temp_df)), temp_df['Amount'])
    step = max(1, len(temp_df) // 8) 
    ax4.set_xticks(range(0, len(temp_df), step))
    ax4.set_xticklabels(temp_df['x_axis'][::step], rotation=45)
    plt.tight_layout()
    st.pyplot(fig4)

    st.header('Sector Analysis :')
    sector = st.selectbox('Select Type',['Total','Count'],key='select_type_2')
    if sector == 'Total':
        sector_series = df.groupby('Vertical')['Amount'].sum().sort_values(ascending=False).head(8)
    else:
        sector_series = df.groupby('Vertical')['Amount'].count().sort_values(ascending=False).head(5)
    fig5, ax5 = plt.subplots()
    ax5.pie(sector_series.values,labels=sector_series.index, autopct='%0.01f%%',shadow=False,startangle=90)
    st.pyplot(fig5)

    st.subheader('Top 50 Highest Funded City')
    st.dataframe(round(df.groupby('City')['Amount'].sum()).sort_values(ascending=False).head(50).reset_index().style.hide(
        axis="index"))

    st.subheader('Enter City ( For Funding Information )')
    select2 = st.selectbox('City',df['City'])
    city_funded(select2)

    st.subheader('Top Startups')
    overall , year = st.columns(2)

    with year:
        st.subheader('Year wise top Startup')
        top_in_year = df.groupby('year').apply(lambda x: x.nlargest(1, 'Amount')).reset_index(drop=True)
        year_selected = st.selectbox('Year' , df['year'].unique())
        top__in_year(year_selected)

    with overall:
        st.subheader('Overall Top Startup')
        highest = df.groupby('Startup')['Amount'].sum().sort_values(ascending=False).head(3)
        st.dataframe(highest)

    st.subheader('Top 20 Investors')
    top_investors = df.groupby('Investors')['Amount'].sum().sort_values(ascending=False).head(20).reset_index()
    st.dataframe(top_investors,hide_index=True)
    create_year_sector_heatmap(df)


def create_year_sector_heatmap(df):
    st.header("Year × Sector Funding Heatmap")
    st.caption("See which sectors attracted most funding each year")
    pivot = df.pivot_table(
        index='Vertical',
        columns='year',
        values='Amount',
        aggfunc='sum',
        fill_value=0
    )
    top_sectors = df.groupby('Vertical')['Amount'].sum().nlargest(20).index
    pivot = pivot.loc[top_sectors]

    col1, col2, col3 = st.columns(3)
    with col1:
        hottest_year = pivot.sum(axis=0).idxmax()
        st.metric("Hottest Year", hottest_year)
    with col2:
        hottest_sector = pivot.sum(axis=1).idxmax()
        st.metric("Hottest Sector", hottest_sector)
    with col3:
        peak_value = pivot.max().max()
        st.metric("Peak Funding", f"{peak_value:,.0f} Cr")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(
        pivot / 1e6,  # Convert to millions
        cmap='RdYlGn',
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Funding (in $ Millions)'}
    )
    ax.set_title('Funding Distribution: Sector × Year ($ Millions)', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Industry Sector', fontsize=12)

    st.pyplot(fig)

    # Insights
    with st.expander("Key Insights"):
        st.markdown(f"""
        1. **{hottest_sector}** received the most cumulative funding
        2. **{hottest_year}** was the peak funding year overall
        3. **Emerging Sectors**: {list(pivot.iloc[:, -3:].sum(axis=1).nlargest(3).index)}
        4. **Declining Sectors**: {list(pivot.iloc[:, -3:].sum(axis=1).nsmallest(3).index)}
        """)


def city_funded(city):
    money = round(df[df['City'] == city]['Amount'].sum())
    st.write(str(money) + ' Cr')

def top__in_year(year):
    sstartup = df[df['year'] == year].nlargest(1, 'Amount')['Startup'].iloc[0]
    st.write(sstartup)


def load_investor_details(investor):
    st.title(investor)
    
    investor_df = df[df['Investors'].str.contains(investor, na=False, case=False)]

    if investor_df.empty:
        st.warning(f"No investments found for {investor}")
        return

    last5_df = investor_df.head()[['Date', 'Startup', 'Vertical', 'City', 'Round', 'Amount']]
    st.subheader('Most Recent Investments')
    st.dataframe(last5_df)

    col1, col2 = st.columns(2)

    with col1:
        if not investor_df.empty:
            big_series = investor_df.groupby('Startup')['Amount'].sum().sort_values(ascending=False).head()
            st.subheader('Biggest Investments')
            if not big_series.empty:
                fig, ax = plt.subplots()
                ax.bar(big_series.index, big_series.values, color='blue')
                ax.set_xticklabels(big_series.index, rotation=45, ha='right')
                ax.set_ylabel('Amount')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.write("No data available")

    with col2:
        if not investor_df.empty:
            vertical_series = investor_df.groupby('Vertical')['Amount'].sum()
            st.subheader('Sectors Invested in')
            if not vertical_series.empty:
                fig1, ax1 = plt.subplots()
                ax1.pie(vertical_series.values, labels=vertical_series.index, autopct='%0.01f%%',
                        shadow=False, startangle=90)
                st.pyplot(fig1)
            else:
                st.write("No data available")

    col3, col4 = st.columns(2)

    with col3:
        if not investor_df.empty:
            st.subheader('Invested Stage')
            round_series = investor_df.groupby('Round')['Amount'].sum()
            if not round_series.empty:
                fig2, ax2 = plt.subplots()
                ax2.plot(round_series.index, round_series.values, marker='o')
                ax2.set_xlabel('Round')
                ax2.set_ylabel('Amount')
                ax2.set_xticklabels(round_series.index, rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.write("No data available")

    with col4:
        if not investor_df.empty:
            st.subheader('Invested By City')
            city_series = investor_df.groupby('City')['Amount'].sum().sort_values(ascending=False).head(10)

            if not city_series.empty:
                fig, ax = plt.subplots()
                ax.bar(city_series.index, city_series.values)
                ax.set_xlabel('City')
                ax.set_ylabel('Amount')
                ax.set_title('Top 10 Cities by Investment')
                ax.set_xticklabels(city_series.index, rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.write("No data available")

    # Investment Year
    if not investor_df.empty:
        st.subheader('Investment Year')
        year_series = investor_df.groupby('year')['Amount'].sum()
        if not year_series.empty:
            fig3, ax3 = plt.subplots()
            ax3.plot(year_series.index, year_series.values, marker='*')
            ax3.set_xlabel('Year')
            ax3.set_ylabel('Amount')
            ax3.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig3)
        else:
            st.write("No data available")


def find_similar_investors(df , investor_name, top_n=10):
    st.subheader('Similar Investors :')
    investor_investments = df[df['Investors'].str.contains(investor_name, na=False, case=False)]

    if investor_investments.empty:
        st.warning(f"No investments found for {investor_name}")
        return pd.DataFrame()

    investor_startups = investor_investments['Startup'].unique()
    co_investor_data = {}

    for startup in investor_startups:
        startup_investments = df[df['Startup'] == startup]

        for idx, row in startup_investments.iterrows():
            if pd.isna(row['Investors']):
                continue

            investors = str(row['Investors']).split(',')
            investors = [inv.strip() for inv in investors]

            investors = [inv for inv in investors
                         if investor_name.lower() not in inv.lower() and inv != '']

            for co_investor in investors:
                if co_investor not in co_investor_data:
                    co_investor_data[co_investor] = {
                        'count': 0,
                        'common_startups': set()
                    }

                co_investor_data[co_investor]['count'] += 1
                co_investor_data[co_investor]['common_startups'].add(startup)
                
    if not co_investor_data:
        return pd.DataFrame()

    results = []
    for investor, data in co_investor_data.items():
        results.append({
            'Investor': investor,
            'Co-Investment Count': data['count'],
            'Common Startups': len(data['common_startups']),
            'Sample Startups': ', '.join(list(data['common_startups'])[:3]) +
                               (f'... (+{len(data["common_startups"]) - 3} more)'
                                if len(data['common_startups']) > 3 else '')
        })

    similar_df = pd.DataFrame(results)
    similar_df = similar_df.sort_values('Co-Investment Count', ascending=False).head(top_n)

    return similar_df

def startup_detail(startup):
    st.title(selected_startup)
    col1, col2 = st.columns(2)
    verticals = df.loc[df['Startup'] == startup, 'Vertical'].unique()
    subverticals = df.loc[df['Startup'] == startup, 'Subvertical'].unique()

    with col1:
        st.subheader("Verticals")
        st.write(list(verticals))

    with col2:
        st.subheader("Subverticals")
        st.write(list(subverticals))

    cityss = df.loc[df['Startup']== startup ]['City'].unique()
    total_earning = df.loc[df['Startup']== startup ]['Amount'].sum()

    with col1:
        st.subheader('City')
        st.write(list(cityss))

    with col2:
        st.subheader('Total Funding')
        st.subheader(str(total_earning) + ' Cr')

    Round = df[df['Startup'] == 'Didi']['Round']
    st.subheader('Round')
    st.write(list(Round))

    investors_details = df[df['Startup']== startup ][['Date','Investors','Round','City','Amount']]
    st.subheader('Investments Details :')
    st.dataframe(investors_details,hide_index=True)

investors = []
for inv_list in df['Investors'].dropna():
    for inv in str(inv_list).split(','):
        investors.append(inv.strip())

investors = sorted(set([inv for inv in investors if inv]))

st.sidebar.title('Startup Funding Analysis')

option = st.sidebar.selectbox('Select One', ['Overall Analysis', 'StartUp', 'Investor'])

if option == 'Overall Analysis':
        overall_analysis()

elif option == 'StartUp':
    startup_options = sorted(df['Startup'].unique().tolist())
    selected_startup = st.sidebar.selectbox('Select Startup', startup_options)
    btn1 = st.sidebar.button('Find Startup Details')
    if btn1:
        st.write(f"Details for {selected_startup}")
        startup_detail(selected_startup)

else:
    selected_investor = st.sidebar.selectbox('Select Investor', investors)
    btn2 = st.sidebar.button('Find Investor Details')
    if btn2:
        load_investor_details(selected_investor)
        st.dataframe(find_similar_investors(df , selected_investor,10))
