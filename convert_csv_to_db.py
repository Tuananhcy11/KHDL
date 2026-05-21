import sqlite3
import csv
import os

def convert_csv_to_sqlite(csv_filepath, db_filepath):
    print(f"Starting conversion from {csv_filepath} to {db_filepath}...")
    
    # Connect to SQLite database (will be created if it doesn't exist)
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()
    
    # Drop table if it already exists to ensure a clean slate
    cursor.execute("DROP TABLE IF EXISTS gold_d1")
    
    # Create the table with appropriate data types
    cursor.execute("""
        CREATE TABLE gold_d1 (
            Date TEXT PRIMARY KEY,
            Open REAL NOT NULL,
            High REAL NOT NULL,
            Low REAL NOT NULL,
            Close REAL NOT NULL,
            Volume INTEGER NOT NULL,
            Open_interest REAL NOT NULL
        )
    """)
    print("Table 'gold_d1' created successfully.")
    
    # Read CSV and insert into the database
    with open(csv_filepath, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader) # Skip header row
        
        insert_query = """
            INSERT INTO gold_d1 (Date, Open, High, Low, Close, Volume, Open_interest)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        row_count = 0
        batch = []
        for row in csv_reader:
            if not row:
                continue
            # Parse values to correct types
            date_val = row[0].strip()
            open_val = float(row[1])
            high_val = float(row[2])
            low_val = float(row[3])
            close_val = float(row[4])
            volume_val = int(row[5])
            open_interest_val = float(row[6])
            
            batch.append((date_val, open_val, high_val, low_val, close_val, volume_val, open_interest_val))
            row_count += 1
            
            # Insert in batches of 100
            if len(batch) >= 100:
                cursor.executemany(insert_query, batch)
                batch = []
        
        # Insert any remaining rows
        if batch:
            cursor.executemany(insert_query, batch)
            
    conn.commit()
    print(f"Successfully imported {row_count} rows of gold price data!")
    
    # Verify the import by printing some statistics
    cursor.execute("SELECT COUNT(*) FROM gold_d1")
    total_rows = cursor.fetchone()[0]
    
    cursor.execute("SELECT MIN(Date), MAX(Date) FROM gold_d1")
    min_date, max_date = cursor.fetchone()
    
    cursor.execute("SELECT Date, Close FROM gold_d1 ORDER BY Date DESC LIMIT 5")
    latest_rows = cursor.fetchall()
    
    print("\n--- Database Statistics ---")
    print(f"Total Rows: {total_rows}")
    print(f"Date Range: {min_date} to {max_date}")
    print("Latest 5 entries:")
    for r in latest_rows:
        print(f"  {r[0]}: Close = {r[1]}")
    print("----------------------------")
    
    conn.close()

if __name__ == "__main__":
    csv_file = "Gold_D1_Merged.csv"
    db_file = "Gold_D1_Merged.db"
    
    if os.path.exists(csv_file):
        convert_csv_to_sqlite(csv_file, db_file)
    else:
        print(f"Error: {csv_file} not found in the current directory.")
