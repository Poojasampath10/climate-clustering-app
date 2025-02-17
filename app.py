import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set the page title
st.title("🌍 Climate Change Clustering App")
st.write("Welcome to the Climate Change Clustering Analysis!")

# Load synthetic climate dataset
np.random.seed(42)
years = np.arange(1900, 2051)
global_temp_anomaly = 0.02 * (years - 1900) + np.random.normal(0, 0.1, len(years))
co2_emissions = 280 + 1.5 * (years - 1900) + np.random.normal(0, 10, len(years))
sea_level_rise = 1.5 * (years - 1900) + np.random.normal(0, 5, len(years))

# Create DataFrame
df = pd.DataFrame({
    'Year': years,
    'Global_Temperature_Anomaly': global_temp_anomaly,
    'CO2_Emissions': co2_emissions,
    'Sea_Level_Rise': sea_level_rise,
})

st.write("### 📊 Climate Dataset Sample")
st.dataframe(df.head())

# Feature Selection & Standardization
X = df.drop(columns=['Year'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans Clustering
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Display Clustering Result
st.write("### 🧩 Clustered Data Sample")
st.dataframe(df[['Year', 'Global_Temperature_Anomaly', 'CO2_Emissions', 'Sea_Level_Rise', 'Cluster']].head())

# Plotting Clusters
st.write("### 📈 CO₂ Emissions vs Temperature Anomaly (Clusters)")
plt.figure(figsize=(8, 5))
plt.scatter(
    df['CO2_Emissions'], df['Global_Temperature_Anomaly'], 
    c=df['Cluster'], cmap='plasma', alpha=0.6, edgecolors='k'
)
plt.xlabel("CO₂ Emissions (ppm)")
plt.ylabel("Global Temperature Anomaly (°C)")
plt.title("CO₂ Emissions vs Temperature Anomaly")
plt.grid(True)
st.pyplot(plt)

# Show Cluster Centers
st.write("### 🚩 Cluster Centers")
st.write(kmeans.cluster_centers_)
