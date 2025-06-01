#!/usr/bin/env python3
import csv
import re
import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional

def is_us_number(phone: str) -> bool:
    """
    Check if a phone number is a US number based on the length and pattern.
    US numbers typically have 10 digits when the country code is removed.
    """
    # Remove any non-digit characters for the check
    digits_only = re.sub(r'\D', '', phone)
    
    # If it starts with a 1 and has 11 digits, it's likely a US number with country code
    if digits_only.startswith('1') and len(digits_only) == 11:
        return True
    
    # If it's exactly 10 digits, it's likely a US number without country code
    if len(digits_only) == 10:
        return True
        
    return False

def format_phone_number(phone: str) -> str:
    """
    Format phone number to ensure it starts with a '+' sign.
    For US numbers without country code, add '+1'.
    For non-US numbers without '+', add '+'.
    """
    # Clean the phone number by removing spaces, dashes, parentheses
    cleaned_phone = re.sub(r'[\s\-\(\)]', '', phone)
    
    # If already starts with '+', return as is
    if cleaned_phone.startswith('+'):
        return cleaned_phone
    
    # If it's a US number without country code, add "+1"
    if is_us_number(cleaned_phone):
        # If it already starts with "1", just add the "+"
        if cleaned_phone.startswith('1') and len(re.sub(r'\D', '', cleaned_phone)) == 11:
            return "+" + cleaned_phone
        # If it's a 10-digit number, add "+1"
        if len(re.sub(r'\D', '', cleaned_phone)) == 10:
            return "+1" + cleaned_phone
    
    # For non-US numbers without '+', add '+'
    return "+" + cleaned_phone

def process_csv_file(input_file: str, output_file: str) -> None:
    """
    Process CSV file to standardize phone numbers.
    """
    try:
        rows = []
        formatted_count = 0
        already_formatted_count = 0
        us_numbers_count = 0
        
        print(f"Reading phone numbers from {input_file}...")
        
        # Read the input CSV
        with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Verify the CSV has the expected columns
            if not reader.fieldnames or "Phone Number" not in reader.fieldnames:
                raise ValueError(f"CSV file {input_file} missing required 'Phone Number' column")
            
            # Get all fieldnames to preserve structure
            fieldnames = reader.fieldnames
            
            for row in reader:
                # Get the original phone number
                original_phone = row.get("Phone Number", "").strip()
                
                if original_phone:
                    # Format the phone number
                    formatted_phone = format_phone_number(original_phone)
                    
                    # Track statistics
                    if formatted_phone == original_phone:
                        already_formatted_count += 1
                    else:
                        formatted_count += 1
                        
                    if formatted_phone.startswith("+1"):
                        us_numbers_count += 1
                    
                    # Update the row with formatted phone
                    row["Phone Number"] = formatted_phone
                
                rows.append(row)
        
        # Write the output CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        # Print summary
        total_numbers = len(rows)
        print(f"Successfully processed {total_numbers} phone numbers")
        print(f"  - {already_formatted_count} numbers were already correctly formatted")
        print(f"  - {formatted_count} numbers were reformatted")
        print(f"  - {us_numbers_count} numbers are US numbers (+1)")
        print(f"Results written to {output_file}")
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Format phone numbers in a CSV file to ensure they start with a "+" sign')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('-o', '--output', help='Path to the output CSV file (default: formatted_numbers.csv)', 
                        default='formatted_numbers.csv')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    process_csv_file(args.input_file, args.output)

if __name__ == "__main__":
    main()
