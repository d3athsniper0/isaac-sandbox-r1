import redis
import json
from typing import Dict, Any
import os
import dotenv

dotenv.load_dotenv()

def dump_redis_data(redis_url: str, output_file: str = "redis_dump.json") -> None:
    """
    Dump all Redis data to a JSON file.
    """
    # Initialize Redis connection
    redis_client = redis.from_url(redis_url, ssl_cert_reqs=None)
    
    # Container for all data
    all_data = {
        "keys": [],
        "data": {}
    }
    
    try:
        # Get all keys
        all_keys = redis_client.keys('*')
        print(f"Found {len(all_keys)} keys in Redis")
        all_data["keys"] = [k.decode('utf-8') for k in all_keys]
        
        # Get data for each key
        for key in all_keys:
            decoded_key = key.decode('utf-8')
            key_type = redis_client.type(key).decode('utf-8')
            print(f"Processing key: {decoded_key} (Type: {key_type})")
            
            if key_type == 'string':
                value = redis_client.get(key)
                try:
                    # Try to decode JSON string
                    decoded_value = json.loads(value)
                except json.JSONDecodeError:
                    # If not JSON, store as string
                    decoded_value = value.decode('utf-8')
                all_data["data"][decoded_key] = {
                    "type": "string",
                    "value": decoded_value
                }
            
            elif key_type == 'hash':
                hash_data = redis_client.hgetall(key)
                decoded_hash = {}
                for field, value in hash_data.items():
                    field_decoded = field.decode('utf-8')
                    try:
                        # Try to decode JSON string
                        value_decoded = json.loads(value.decode('utf-8'))
                    except json.JSONDecodeError:
                        # If not JSON, store as string
                        value_decoded = value.decode('utf-8')
                    decoded_hash[field_decoded] = value_decoded
                all_data["data"][decoded_key] = {
                    "type": "hash",
                    "value": decoded_hash
                }
            
            elif key_type == 'set':
                set_members = redis_client.smembers(key)
                decoded_set = [member.decode('utf-8') for member in set_members]
                all_data["data"][decoded_key] = {
                    "type": "set",
                    "value": decoded_set
                }
            
            elif key_type == 'list':
                list_items = redis_client.lrange(key, 0, -1)
                decoded_list = [item.decode('utf-8') for item in list_items]
                all_data["data"][decoded_key] = {
                    "type": "list",
                    "value": decoded_list
                }
            
            else:
                print(f"Unhandled type {key_type} for key {decoded_key}")
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2)
        
        print(f"\nDump completed successfully. Data written to {output_file}")
        
        # Print summary
        print("\nData Summary:")
        key_types = {}
        for key in all_data["keys"]:
            prefix = key.split(':')[0] if ':' in key else 'no_prefix'
            key_types[prefix] = key_types.get(prefix, 0) + 1
        
        print("\nKeys by prefix:")
        for prefix, count in key_types.items():
            print(f"{prefix}: {count} keys")
            
    except Exception as e:
        print(f"Error dumping Redis data: {str(e)}")

if __name__ == "__main__":
    redis_url = os.getenv("REDIS_URL")
    dump_redis_data(redis_url)
