import os
import pandas as pd

# Define the directory containing the treasury data
directory = "treasury"

# Initialize an empty list to store individual DataFrames
dataframes = []

# Loop through files in the directory and process only CSV files
for file in sorted(os.listdir(directory)):
    if file.endswith(".csv"):
        file_path = os.path.join(directory, file)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Convert "Date" column from mm/dd/yyyy to dd/mm/yyyy
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        
        # Append the DataFrame to the list
        dataframes.append(df)

# Combine all DataFrames into one
merged_df = pd.concat(dataframes, ignore_index=True)

# Sort the merged DataFrame by date
merged_df = merged_df.sort_values("Date")

# Generate the complete date range spanning all dates
full_date_range = pd.date_range(start=merged_df["Date"].min(), end=merged_df["Date"].max(), freq="D")

# Reindex the DataFrame to include all dates in the range
merged_df = merged_df.set_index("Date").reindex(full_date_range).reset_index()

# Rename the new index column back to "Date"
merged_df.rename(columns={"index": "Date"}, inplace=True)

# Forward-fill missing values
merged_df = merged_df.fillna(method="ffill")

# Extract the "Date" and "10 Yr" columns
filtered_df = merged_df[["Date", "10 Yr"]]

# Convert the "Date" column back to dd/mm/yyyy format for final output
filtered_df["Date"] = filtered_df["Date"].dt.strftime("%d/%m/%Y")

# Export the filtered DataFrame to a new CSV file
output_file = "merged_treasury_10yr_2.csv"
filtered_df.to_csv(output_file, index=False)

print(f"Merged, sorted, filled weekend dates, and filtered data saved to {output_file}")
