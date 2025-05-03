import pandas as pd
import re
import numpy as np
from datetime import datetime, timedelta

# Function to load and clean one ECEC file
def load_and_clean_ecec_file(file_path, year):
    # Read with two header rows (dates + IN/OUT) and skip top metadata rows
    df_raw = pd.read_excel(file_path, header=[0, 1], skiprows=5, engine="openpyxl")

    # Split metadata (first 7 cols) vs. timestamps (rest)
    meta_df       = df_raw.iloc[:, :7].copy()
    timestamps_df = df_raw.iloc[:, 7:].copy()

    # Flatten metadata column names
    meta_df.columns = [
        '_'.join([str(c) for c in col if 'Unnamed' not in str(c)]).strip()
        for col in meta_df.columns
    ]
    # Flatten timestamp column names to "Mon 1_IN", "Mon 1_OUT", etc.
    timestamps_df.columns = [
        f"{str(c[0]).strip()}_{str(c[1]).strip()}"
        for c in timestamps_df.columns
    ]

    # Add row identifiers for re-merge
    meta_df['row_id']       = meta_df.index
    timestamps_df['row_id'] = timestamps_df.index

    # Melt wide timestamp table into long form
    long_df = timestamps_df.melt(
        id_vars='row_id',
        var_name='date_inout',
        value_name='timestamp'
    )
    # Extract the date string vs. IN/OUT flag
    long_df[['date_str', 'in_out']] = long_df['date_inout'].str.extract(r'(.*)_(IN|OUT)', expand=True)

    # Append year and parse into a proper date with explicit format
    long_df['date_str'] = long_df['date_str'].str.strip() + f" {year}"
    long_df['date']     = pd.to_datetime(
        long_df['date_str'],
        format='%b %d %Y',  # e.g., "Jan 05 2022"
        errors='coerce'
    )

    # Pivot so each row_id+date has one IN and one OUT column
    pivot_df = long_df.pivot_table(
        index=['row_id', 'date'],
        columns='in_out',
        values='timestamp',
        aggfunc='first'
    ).reset_index()

    # Ensure both IN and OUT exist
    for col in ['IN', 'OUT']:
        if col not in pivot_df.columns:
            pivot_df[col] = pd.NaT

    # Merge back metadata
    final_df = pivot_df.merge(meta_df, on='row_id', how='left')

    # Clean raw time strings (keep only "H:MM AM/PM")
    def clean_time_only(x):
        if pd.isna(x):
            return x
        m = re.match(r'^\s*\d{1,2}:\d{2}\s*(AM|PM)', str(x), re.IGNORECASE)
        return m.group(0) if m else x

    final_df['IN']  = final_df['IN'].apply(clean_time_only)
    final_df['OUT'] = final_df['OUT'].apply(clean_time_only)

    # Tag file year
    final_df['year'] = int(year)

    # Select and return core columns
    return final_df[['Record ID', 'Student Status', 'Room', 'Tags', 'date', 'IN', 'OUT', 'year']]

# Raw GitHub URLs converted to raw.githubusercontent.com
ecec_files = {
    "2022": "https://raw.githubusercontent.com/Reybolouri/Semester-Project-ECON8310-Reybolouri/main/data/ECEC%202022%20Student%20Sign%20In%20and%20Out.xlsx",
    "2023": "https://raw.githubusercontent.com/Reybolouri/Semester-Project-ECON8310-Reybolouri/main/data/ECEC%202023%20Student%20Sign%20In%20and%20Out.xlsx",
    "2024": "https://raw.githubusercontent.com/Reybolouri/Semester-Project-ECON8310-Reybolouri/main/data/ECEC%202024%20Student%20Sign%20In%20and%20Out.xlsx",
    "2025": "https://raw.githubusercontent.com/Reybolouri/Semester-Project-ECON8310-Reybolouri/main/data/ECEC%202025%2001012025-02282025%20Student%20Sign%20In%20and%20Out.xlsx"
}

# Load and combine all years
all_years_df = pd.concat(
    [load_and_clean_ecec_file(path, year) for year, path in ecec_files.items()],
    ignore_index=True
)

# Identify age group from Room
age_group_pattern = r'(Infants|Multi-Age|Toddlers|Preschool|Pre-K)'
all_years_df['age_group'] = all_years_df['Room'].str.extract(age_group_pattern, expand=False)

# Treat "--" as missing sign-out
all_years_df['OUT'] = all_years_df['OUT'].replace('--', pd.NA)

# Build full datetime objects for IN and OUT with explicit format
all_years_df['in_datetime'] = pd.to_datetime(
    all_years_df['date'].dt.strftime('%Y-%m-%d') + ' ' + all_years_df['IN'],
    format='%Y-%m-%d %I:%M %p',  # e.g., "2022-01-05 08:30 AM"
    errors='coerce'
)
all_years_df['out_datetime'] = pd.to_datetime(
    all_years_df['date'].dt.strftime('%Y-%m-%d') + ' ' + all_years_df['OUT'],
    format='%Y-%m-%d %I:%M %p',
    errors='coerce'
)

# Drop incomplete sessions
attended_df = all_years_df.dropna(subset=['in_datetime', 'out_datetime']).copy()

# Generate 30-minute blocks for each session
def generate_30min_blocks(start, end):
    return pd.date_range(start=start, end=end, freq='30min').tolist()

attended_df['time_blocks'] = attended_df.apply(
    lambda row: generate_30min_blocks(row['in_datetime'], row['out_datetime']),
    axis=1
)

# Explode so each row = one child in one 30-min block
expanded_df = attended_df.explode('time_blocks')
expanded_df['time_block'] = expanded_df['time_blocks'].dt.floor('30min')

# Build attendance grid
attendance_grid = expanded_df[['Record ID', 'Student Status', 'age_group', 'time_block']].copy()

# Count children present per age group & block
grouped = attendance_grid.groupby(
    ['age_group', 'time_block', 'Student Status']
).agg(
    children_present=('Record ID', 'nunique')
).reset_index()

# Student-to-staff ratios
ratio_table = pd.DataFrame({
    'age_group': ['Infants', 'Multi-Age', 'Toddlers', 'Preschool', 'Pre-K'],
    'student_to_staff': [4, 4, 6, 10, 12]
})

# Merge ratios and compute required staff
grouped = grouped.merge(ratio_table, on='age_group', how='left')
grouped['staff_required'] = np.ceil(grouped['children_present'] / grouped['student_to_staff']).astype(int)

# Save final staffing grid
grouped.to_csv("ecec_staffing_grouped.csv", index=False)

print("✅ ECEC data cleaned and staffing grid saved to ecec_staffing_grouped.csv")
