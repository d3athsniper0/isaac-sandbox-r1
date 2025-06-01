#!/usr/bin/env python3
import csv
import argparse
import os
import sys
from typing import Dict, Set, List, Tuple

def read_phone_numbers(csv_path: str) -> Dict[str, str]:
    """
    Read phone numbers and usernames from a CSV file.
    Returns a dictionary with phone numbers as keys and usernames as values.
    """
    phone_users = {}
    
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Verify that the CSV has the expected columns
            if not reader.fieldnames or "Phone Number" not in reader.fieldnames:
                raise ValueError(f"CSV file {csv_path} missing required 'Phone Number' column")
            
            for row in reader:
                phone = row.get("Phone Number", "").strip()
                username = row.get("User Name", "").strip()
                
                if phone:  # Only add if phone number is not empty
                    phone_users[phone] = username
    
    except Exception as e:
        print(f"Error reading {csv_path}: {str(e)}")
        sys.exit(1)
        
    return phone_users

def subtract_csv_files(file_a: str, file_b: str, output_file: str) -> None:
    """
    Subtract phone numbers in file_b from file_a and write results to output_file.
    """
    print(f"Reading phone numbers from {file_a}...")
    phones_a = read_phone_numbers(file_a)
    print(f"Found {len(phones_a)} phone numbers in {file_a}")
    
    print(f"Reading phone numbers from {file_b}...")
    phones_b = read_phone_numbers(file_b)
    print(f"Found {len(phones_b)} phone numbers in {file_b}")
    
    # Get set of phone numbers in B
    phones_b_set = set(phones_b.keys())
    
    # Filter phones in A that are not in B
    result_phones = {phone: username for phone, username in phones_a.items() 
                    if phone not in phones_b_set}
    
    print(f"Found {len(result_phones)} phone numbers in {file_a} that are not in {file_b}")
    
    # Write results to output file
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Phone Number", "User Name"])
            
            for phone, username in result_phones.items():
                writer.writerow([phone, username])
                
        print(f"Successfully wrote {len(result_phones)} phone numbers to {output_file}")
        
    except Exception as e:
        print(f"Error writing to {output_file}: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Subtract phone numbers in B.csv from A.csv')
    parser.add_argument('file_a', help='Path to the first CSV file (A.csv)')
    parser.add_argument('file_b', help='Path to the second CSV file (B.csv)')
    parser.add_argument('-o', '--output', help='Path to the output CSV file (default: result.csv)', 
                        default='result.csv')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.file_a):
        print(f"Error: Input file '{args.file_a}' does not exist")
        sys.exit(1)
        
    if not os.path.exists(args.file_b):
        print(f"Error: Input file '{args.file_b}' does not exist")
        sys.exit(1)
    
    subtract_csv_files(args.file_a, args.file_b, args.output)

if __name__ == "__main__":
    main()
