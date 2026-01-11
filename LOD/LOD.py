import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
csv_filename = 'LOD_CHART.csv'

# --- Data Loading ---

# Check if the CSV file exists
if not os.path.exists(csv_filename):
    print(f"Error: The file '{csv_filename}' was not found in the current directory.")
    exit() # Exit the script if the file doesn't exist

# Load all data from your complete CSV file
try:
    df = pd.read_csv(csv_filename)
    # --- MODIFICATION: Sort by LOD in ascending order ---
    df = df.sort_values('LOD (uM)', ascending=True).reset_index(drop=True)
    
    print("Successfully loaded and sorted data for plotting:")
    print(df)
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")
    exit()

# --- Horizontal Bar Plot Generation ---

plt.figure(figsize=(10, 7))
# --- MODIFICATION: Use plt.barh for a horizontal plot ---
# Convert angles to string to treat them as categories
bars = plt.barh(df['Angle (deg)'].astype(str), df['LOD (uM)'],
                color='mediumseagreen',
                edgecolor='black')

# Invert the y-axis so the lowest LOD (first item) is at the top
plt.gca().invert_yaxis()

# --- Formatting the Plot ---

# --- MODIFICATION: Swap axis labels ---
plt.ylabel("Angle of Incidence (degrees)", fontsize=16)
plt.xlabel("Limit of Detection (LOD) [µM]", fontsize=16)
plt.grid(axis='x', linestyle='--', alpha=0.7) # Grid lines are now vertical
plt.tight_layout() # Adjusts plot to prevent labels from overlapping

# Set a slightly larger limit for the x-axis to make space for labels
plt.xlim(0, df['LOD (uM)'].max() * 1.15)

# Display the plot
plt.show()