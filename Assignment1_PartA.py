# Assignment 1 - Predictive Modelling of Eating-Out Problem - Part A Exploratory Data Analysis
# Question 1 - Load and explore dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('/Users/stacey/Desktop/u3257317_assignment1 of Data Science technology and systems/zomato_df_final_data.csv')

print("------Basic Dataset Information")
print(df.info())  

print("\n----Dataset Shape")
print(f"Dataset shape: {df.shape}")  # Number of rows and columns

print("\n-----First 5 Rows")
print(df.head())

print("\n-----Descriptive Statistics for Numerical Columns")
print(df.describe())

print("\n----Missing Values Check ")
print(df.isnull().sum()) # Show only columns with missing values

print("\n---Data Types")
print(df.dtypes)




# Question 2 - Answer the following with description and visuals
# 2.1 How many unique cuisines are served?
# Calculate the number of unique cuisines
unique_cuisines = df['cuisine'].nunique()
print(f"Number of unique cuisine types: {unique_cuisines}")

#2.2 Which 3 suburbs have the most restaurants?
# Calculate the number of restaurants per suburb
suburb_counts = df['subzone'].value_counts()
top_suburbs = suburb_counts.head(3)
print("Top 3 suburbs by number of restaurants:")
for i, (suburb, count) in enumerate(top_suburbs.items(), 1):
    print(f"{i}. {suburb}: {count} restaurants")

#2.3 Are restaurants with“Excellent”ratings more expensive than those with“Poor”ratings? Support with visuals (histograms 
# 1) Build a numeric price column (prefer cost $→1–4; fallback to cost_2; then price_numeric)
if 'cost' in df.columns:
    mapping = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4}
   
    price = df['cost'].map(mapping)
    if price.isna().all():
        price = pd.to_numeric(df['cost'], errors='coerce')
elif 'cost_2' in df.columns:
    price = pd.to_numeric(df['cost_2'], errors='coerce')
else:
    price = pd.to_numeric(df.get('price_numeric'), errors='coerce')

# 2) Filter by rating 
rt = df['rating_text'].astype(str).str.strip().str.casefold()
ex_prices = price[rt.eq('excellent')].dropna()
po_prices = price[rt.eq('poor')].dropna()

# 3) Histogram 
plt.figure(figsize=(9, 5))
if len(ex_prices) and len(po_prices):
    bins = np.histogram_bin_edges(
        np.concatenate([ex_prices.values, po_prices.values]), bins='auto'
    )
else:
    bins = np.histogram_bin_edges(
        ex_prices.values if len(ex_prices) else po_prices.values, bins='auto'
    )

if len(po_prices):
    plt.hist(poor_prices := po_prices, bins=bins, alpha=0.6, label='Poor', edgecolor='black')
if len(ex_prices):
    plt.hist(excellent_prices := ex_prices, bins=bins, alpha=0.6, label='Excellent', edgecolor='black')

plt.xlabel('Price range')
plt.ylabel('Number of Restaurants')
plt.title('Histogram: Price Distribution by Rating')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




#Questions3:Explore key variables
# 3.1 Distribution of cost, ratings, and restaurant types
print("\n--- 3.1 Distribution Analysis")

plt.figure(figsize=(15, 5))

# Subplot 1: Cost distribution histogram (uses 'cost' column)
plt.subplot(1, 3, 1)
# Drop missing values
cost_data = df['cost'].dropna()
plt.hist(cost_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Restaurant Cost')
plt.xlabel('Cost ($)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Add summary lines
mean_cost = cost_data.mean()
median_cost = cost_data.median()
plt.axvline(mean_cost, color='red', linestyle='--', label=f'Mean: ${mean_cost:.2f}')
plt.axvline(median_cost, color='green', linestyle='--', label=f'Median: ${median_cost:.2f}')
plt.legend()

# Subplot 2: Rating distribution (uses 'rating_number')
plt.subplot(1, 3, 2)
rating_data = df['rating_number'].dropna()
plt.hist(rating_data, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('Distribution of Restaurant Ratings')
plt.xlabel('Rating number')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Add mean line
mean_rating = rating_data.mean()
plt.axvline(mean_rating, color='red', linestyle='--', label=f'Mean: {mean_rating:.2f}')
plt.legend()

# Subplot 3: Restaurant type distribution (uses 'types')
plt.subplot(1, 3, 3)
type_data = df['type'].dropna()
type_counts = type_data.value_counts().head(15)  # show top 15 types
plt.barh(type_counts.index, type_counts.values, alpha=0.7, color='gold')
plt.title('Top 20 Restaurant Types')
plt.xlabel('Number of Restaurants')
plt.tight_layout()
plt.show()

# Summary stats
print(f"Cost Statistics:")
print(f"  Mean cost: ${df['cost'].mean():.2f}")
print(f"  Median cost: ${df['cost'].median():.2f}")
print(f"  Cost range: ${df['cost'].min():.2f} - ${df['cost'].max():.2f}")
print(f"  Missing values: {df['cost'].isnull().sum()}")

print(f"\nRating Statistics:")
print(f"  Mean rating: {df['rating_number'].mean():.2f}")
print(f"  Rating range: {df['rating_number'].min():.1f} - {df['rating_number'].max():.1f}")
print(f"  Missing values: {df['rating_number'].isnull().sum()}")

print(f"\nType Statistics:")
print(f"  Number of unique types: {df['type'].nunique()}")
print(f"  Most common type: {df['type'].mode().iloc[0] if not df['type'].mode().empty else 'N/A'}")
print(f"  Missing values: {df['type'].isnull().sum()}")

# 3.2 Correlation between cost and votes
print("\n--- 3.2 Correlation Analysis: Cost vs Votes ")

# Scatter plot of cost vs number of votes
plt.figure(figsize=(10, 6))

# Keep valid rows only
valid_data = df[['cost', 'votes']].dropna()

if len(valid_data) > 0:
    plt.scatter(valid_data['cost'], valid_data['votes'], alpha=0.5, color='purple')
    plt.title('Correlation between Cost and Votes')
    plt.xlabel('Cost ($)')
    plt.ylabel('Number of Votes')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    try:
        z = np.polyfit(valid_data['cost'], valid_data['votes'], 1)
        p = np.poly1d(z)
        plt.plot(valid_data['cost'], p(valid_data['cost']), "r--", alpha=0.8, label='Trend line')
        plt.legend()
    except:
        pass
    
    plt.tight_layout()
    plt.show()
    
    # Correlation coefficient
    correlation = valid_data['cost'].corr(valid_data['votes'])
    print(f"Correlation coefficient between cost and votes: {correlation:.3f}")
    
    if correlation > 0.3:
        print("→ Positive correlation: More expensive restaurants tend to have more votes")
    elif correlation < -0.3:
        print("→ Negative correlation: Less expensive restaurants tend to have more votes")
    else:
        print("→ Weak correlation: Little relationship between cost and number of votes")
else:
    print("Insufficient data for correlation analysis")

# Additional analysis: cost vs rating
print("\n--- Additional Analysis: Cost vs Rating ")

valid_data_rating = df[['cost', 'rating_number']].dropna()

if len(valid_data_rating) > 0:
    plt.figure(figsize=(10, 6))
    plt.scatter(valid_data_rating['cost'], valid_data_rating['rating_number'], alpha=0.5, color='blue')
    plt.title('Correlation between Cost and Rating Score')
    plt.xlabel('Cost ($)')
    plt.ylabel('Rating Score')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    try:
        z = np.polyfit(valid_data_rating['cost'], valid_data_rating['rating_number'], 1)
        p = np.poly1d(z)
        plt.plot(valid_data_rating['cost'], p(valid_data_rating['cost']), "r--", alpha=0.8, label='Trend line')
        plt.legend()
    except:
        pass
    
    plt.tight_layout()
    plt.show()
    
    correlation_rating = valid_data_rating['cost'].corr(valid_data_rating['rating_number'])
    print(f"Correlation coefficient between cost and rating: {correlation_rating:.3f}")
    
    if correlation_rating > 0.2:
        print("→ Positive trend: More expensive restaurants tend to have higher ratings")
    elif correlation_rating < -0.2:
        print("→ Negative trend: Less expensive restaurants tend to have higher ratings")
    else:
        print("→ Weak relationship between cost and rating")

# 3.3 Interesting trends
print("\n--- 3.3 Interesting Trends observe")

# Trend 1: Average rating across cost bands
print("\nTrend 1: Average Rating by Cost Segments")

# Create cost bins
cost_bins = pd.cut(df['cost'].dropna(), bins=5)
rating_by_cost = df.groupby(cost_bins)['rating_number'].agg(['mean', 'count', 'std']).round(2)

print("Rating statistics by cost segments:")
print(rating_by_cost)

# Display
plt.figure(figsize=(10, 6))
rating_by_cost['mean'].plot(kind='bar', color='lightcoral', alpha=0.7)
plt.title('Average Rating by Cost Segments')
plt.xlabel('Cost Range ($)')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Trend 2: Cost and rating by restaurant type
print("\nTrend 2: Cost and Rating by Restaurant Type")

top_types = df['type'].value_counts().head(5).index  # top 5 most common types

type_stats = df[df['type'].isin(top_types)].groupby('type').agg({
    'cost': ['mean', 'median', 'count'],
    'rating_number': ['mean', 'median']
}).round(2)

print("Statistics for top 5 restaurant types:")
print(type_stats)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Average cost comparison
type_stats[('cost', 'mean')].sort_values(ascending=False).plot(kind='bar', ax=ax1, color='lightblue', alpha=0.7)
ax1.set_title('Average Cost by Restaurant Type')
ax1.set_ylabel('Average Cost ($)')
ax1.tick_params(axis='x', rotation=45)

# Average rating comparison
type_stats[('rating_number', 'mean')].sort_values(ascending=False).plot(kind='bar', ax=ax2, color='lightgreen', alpha=0.7)
ax2.set_title('Average Rating by Restaurant Type')
ax2.set_ylabel('Average Rating')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Trend 3: Votes vs ratings
print("\nTrend 3: Relationship between Votes and Ratings")

valid_votes_rating = df[['votes', 'rating_number']].dropna()

if len(valid_votes_rating) > 0:
    plt.figure(figsize=(10, 6))
    plt.scatter(valid_votes_rating['votes'], valid_votes_rating['rating_number'], alpha=0.5, color='orange')
    plt.title('Relationship between Number of Votes and Rating Score')
    plt.xlabel('Number of Votes')
    plt.ylabel('Rating Score')
    plt.grid(True, alpha=0.3)
    
    correlation_votes_rating = valid_votes_rating['votes'].corr(valid_votes_rating['rating_number'])
    print(f"Correlation between votes and rating: {correlation_votes_rating:.3f}")
    
    if correlation_votes_rating > 0.2:
        print("→ Restaurants with more votes tend to have higher ratings")
    else:
        print("→ Weak relationship between number of votes and rating score")

# Trend 4: Missing values
print("\nTrend 4: Missing Values Pattern Analysis")

missing_summary = pd.DataFrame({
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
}).sort_values('Missing_Count', ascending=False)

print("Missing values summary:")
print(missing_summary[missing_summary['Missing_Count'] > 0])

# Key findings
print("\n=== Key Findings Summary ")
print("1. COST DISTRIBUTION:")
print(f"   - Average restaurant cost: ${df['cost'].mean():.2f}")
print(f"   - Cost distribution shows {('right' if df['cost'].skew() > 0 else 'left')}-skewed pattern")

print("\n2. RATING PATTERNS:")
print(f"   - Average rating: {df['rating_number'].mean():.2f}/5")
print(f"   - Rating distribution is relatively {'normal' if abs(df['rating_number'].skew()) < 1 else 'skewed'}")

print("\n3. CORRELATION INSIGHTS:")
if 'correlation' in locals():
    print(f"   - Cost–Votes correlation: {correlation:.3f} ({'positive' if correlation > 0 else 'negative'} relationship)")
if 'correlation_rating' in locals():
    print(f"   - Cost–Rating correlation: {correlation_rating:.3f}")

print("\n4. DATA QUALITY:")
print(f"   - Main concern: {df['rating_number'].isnull().sum()} missing ratings ({df['rating_number'].isnull().sum()/len(df)*100:.1f}% of data)")
print(f"   - Cost data missing: {df['cost'].isnull().sum()} records")

# Question 4 Geospatial Analysis:
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np

print("\n== 4.1 Loading Geojson Data")

sydney_gdf = gpd.read_file("/Users/stacey/Desktop/u3257317_assignment1 of Data Science technology and systems/sydney.geojson")
print(f"Suburbs loaded: {len(sydney_gdf)}")

print("\n== 4.2 Preparing Data")

# Prepare data
suburb_col = 'SSC_NAME'
df['subzone_clean'] = df['subzone'].str.lower().str.strip()
sydney_gdf['suburb_clean'] = sydney_gdf[suburb_col].str.lower().str.strip()

# Get all cuisines
all_cuisines = df['cuisine'].unique()
print(f"Total cuisine types: {len(all_cuisines)}")

print("\n== 4.3 Creating All Cuisines Distribution Map")

# Create a single map showing the overall distribution of all cuisines
fig, ax = plt.subplots(1, 1, figsize=(15, 12))

# Compute total number of restaurants per suburb (all cuisines)
suburb_totals = df.groupby('subzone_clean').size().reset_index(name='total_restaurants')

# Merge into the geospatial data
combined_gdf = sydney_gdf.merge(
    suburb_totals, 
    left_on='suburb_clean', 
    right_on='subzone_clean', 
    how='left'
)
combined_gdf['total_restaurants'] = combined_gdf['total_restaurants'].fillna(0)

# Plot the overall distribution map for all cuisines
if combined_gdf['total_restaurants'].max() > 0:
    # Use a gradient to show restaurant density
    plot = combined_gdf.plot(column='total_restaurants', 
                           ax=ax, 
                           legend=True,
                           cmap='YlOrRd',  # yellow-to-red gradient
                           edgecolor='white',
                           linewidth=0.5,
                           legend_kwds={'label': 'Total Number of Restaurants (All Cuisines)',
                                       'shrink': 0.8})
    
    # Highlight suburbs that have restaurants
    has_restaurants = combined_gdf[combined_gdf['total_restaurants'] > 0]
    if len(has_restaurants) > 0:
        has_restaurants.plot(ax=ax, color='none', edgecolor='black', linewidth=0.8)
else:
    combined_gdf.plot(ax=ax, color='lightgray', edgecolor='white')

ax.set_title('Distribution of All Restaurant Cuisines in Sydney Suburbs', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_axis_off()

# Add summary statistics
total_restaurants = len(df)
suburbs_with_restaurants = (combined_gdf['total_restaurants'] > 0).sum()

ax.text(0.02, 0.98, f'Total restaurants (all cuisines): {total_restaurants}', 
        transform=ax.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.text(0.02, 0.92, f'Suburbs with restaurants: {suburbs_with_restaurants}', 
        transform=ax.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.text(0.02, 0.86, f'Unique cuisine types: {len(all_cuisines)}', 
        transform=ax.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('sydney_all_cuisines_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n== 4.4 Cuisine Diversity Analysis")

# Create a second map showing cuisine diversity
fig2, ax2 = plt.subplots(1, 1, figsize=(15, 12))

# Calculate cuisine diversity per suburb (number of distinct cuisines)
cuisine_diversity = df.groupby('subzone_clean')['cuisine'].nunique().reset_index(name='cuisine_variety')

# Merge into the geospatial data
diversity_gdf = sydney_gdf.merge(
    cuisine_diversity, 
    left_on='suburb_clean', 
    right_on='subzone_clean', 
    how='left'
)
diversity_gdf['cuisine_variety'] = diversity_gdf['cuisine_variety'].fillna(0)

# Plot the cuisine diversity map
if diversity_gdf['cuisine_variety'].max() > 0:
    diversity_gdf.plot(column='cuisine_variety', 
                      ax=ax2, 
                      legend=True,
                      cmap='Blues',  # blue gradient
                      edgecolor='white',
                      linewidth=0.5,
                      legend_kwds={'label': 'Number of Different Cuisine Types',
                                  'shrink': 0.8})
else:
    diversity_gdf.plot(ax=ax2, color='lightgray', edgecolor='white')

ax2.set_title('Cuisine Diversity in Sydney Suburbs\n(Number of Different Cuisine Types Available)', 
              fontsize=16, fontweight='bold', pad=20)
ax2.set_axis_off()

plt.tight_layout()
plt.savefig('sydney_cuisine_diversity.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n== 4.5 Detailed Analysis Results")

# Overall statistics
print("OVERALL RESTAURANT DISTRIBUTION:")
print(f"Total restaurants: {total_restaurants}")
print(f"Total suburbs in Sydney: {len(sydney_gdf)}")
print(f"Suburbs with restaurants: {suburbs_with_restaurants}")
print(f"Suburbs without restaurants: {len(sydney_gdf) - suburbs_with_restaurants}")
print(f"Unique cuisine types: {len(all_cuisines)}")

# Restaurant density analysis
print(f"\nRESTAURANT DENSITY ANALYSIS:")
max_restaurants = combined_gdf['total_restaurants'].max()
avg_restaurants = combined_gdf['total_restaurants'].mean()
print(f"Maximum restaurants in one suburb: {max_restaurants}")
print(f"Average restaurants per suburb: {avg_restaurants:.1f}")
print(f"Suburbs with 10+ restaurants: {(combined_gdf['total_restaurants'] >= 10).sum()}")


# Show suburbs with the most restaurants
print(f"\nTOP 10 SUBURBS BY RESTAURANT COUNT:")
top_suburbs = combined_gdf.nlargest(10, 'total_restaurants')[[suburb_col, 'total_restaurants']]
for idx, row in top_suburbs.iterrows():
    if row['total_restaurants'] > 0:
        print(f"  {row[suburb_col]}: {int(row['total_restaurants'])} restaurants")

# Show suburbs with the most cuisine types
print(f"\nTOP 10 SUBURBS BY CUISINE DIVERSITY:")
diverse_suburbs = diversity_gdf.nlargest(10, 'cuisine_variety')[[suburb_col, 'cuisine_variety']]
for idx, row in diverse_suburbs.iterrows():
    if row['cuisine_variety'] > 0:
        print(f"  {row[suburb_col]}: {int(row['cuisine_variety'])} cuisine types")





# Question 5: Interactive Visualization with Plotly  
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Plotly not available, creating static visualizations instead")
    PLOTLY_AVAILABLE = False

print("\n---------- Question 5: Interactive Visualization  with plotly- Cost vs Rating Analysis")

# Prepare data
valid_data_rating = df[['cost', 'rating_number', 'type', 'subzone', 'cuisine', 'votes']].dropna()

if PLOTLY_AVAILABLE:
    #  Create interactive scatter plot - Cost vs Rating
    print("\n--- Creating Interactive Scatter Plot: Cost vs Rating")
    
    fig = px.scatter(
        valid_data_rating, 
        x='cost', 
        y='rating_number',
        color='type',
        size='votes',
        hover_data=['subzone', 'cuisine', 'votes'],
        title='Interactive Analysis: Restaurant Cost vs Rating Score',
        labels={'cost': 'Cost ($)', 'rating_number': 'Rating Score'},
        width=1000,
        height=600
    )
    
    # Add trend line
    z = np.polyfit(valid_data_rating['cost'], valid_data_rating['rating_number'], 1)
    p = np.poly1d(z)
    trend_x = np.linspace(valid_data_rating['cost'].min(), valid_data_rating['cost'].max(), 100)
    trend_y = p(trend_x)
    
    fig.add_trace(
        go.Scatter(
            x=trend_x,
            y=trend_y, 
            mode='lines', 
            name='Trend Line',
            line=dict(color='red', width=3, dash='dash')
        )
    )
    
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='closest'
    )
    
    fig.show()
    
    # Save interactive chart
    fig.write_html("interactive_cost_vs_rating.html")
    print("Interactive chart saved as interactive_cost_vs_rating.html")
    
else:
    # If Plotly is unavailable, create a static fallback
    print("Creating static visualization as fallback")
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        valid_data_rating['cost'],
        valid_data_rating['rating_number'], 
        c=valid_data_rating['votes'],
        alpha=0.6,
        cmap='viridis'
    )
    plt.colorbar(scatter, label='Number of Votes')
    plt.xlabel('Cost ($)')
    plt.ylabel('Rating Score')
    plt.title('Cost vs Rating (Static)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#  Explain why the interactive version is better
print("\n--- Explanation: Why the Interactive Version is Better")

print("""
Advantages of interactive visualization over static plots:

1. Detailed exploration: Hover to see each restaurant's details.
2.Multi-dimensional analysis: Simultaneously encode cost, rating, type, area, and votes.
3.Dynamic filtering: Click legend items to isolate specific restaurant types.
4. Zooming: Inspect dense regions for fine-grained patterns.
5. Better user experience: Users can explore patterns on their own.
""")

# Key statistics
correlation_cost_rating = valid_data_rating['cost'].corr(valid_data_rating['rating_number'])
print(f"\nKey finding: the correlation between cost and rating is {correlation_cost_rating:.3f}")




