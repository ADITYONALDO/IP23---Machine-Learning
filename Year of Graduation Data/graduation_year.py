# importing the libraries
import pandas as pd
from datetime import datetime

# Load the data
df = pd.read_excel('Final_Lead_Data.xlsx')

# Remove duplicates based on 'Email'
df = df.drop_duplicates('Email')

# Select necessary columns
selected_columns = ["ID", "First Name", "Email", "Created", "Academic Year", "What is your current academic year?"]
df = df[selected_columns]

# Function to extract year from 'Created' column
def get_year(date_str):
    date = datetime.strptime(date_str, '%m/%d/%Y %I:%M:%S %p')
    return date.year

# Apply the function to 'Created' column
df['Year'] = df['Created'].astype(str).apply(get_year)

# Function to compute graduation year
def compute_grad_year(row):
    return row['Year'] + (4 - row['Academic Year'])

# Compute 'Graduation Year'
df['Graduation Year'] = df.apply(compute_grad_year, axis=1)

# Select output columns
output_df = df[['ID', 'First Name', 'Email', 'Graduation Year']]

# Save the output
output_df.to_excel('Graduation Year.xlsx', index=False)