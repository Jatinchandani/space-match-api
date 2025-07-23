import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import json

fake = Faker()

# Define property types and their characteristics
PROPERTY_TYPES = {
    'apartment': {'min_bedrooms': 1, 'max_bedrooms': 4, 'min_sqft': 500, 'max_sqft': 2000},
    'house': {'min_bedrooms': 2, 'max_bedrooms': 6, 'min_sqft': 1000, 'max_sqft': 4000},
    'condo': {'min_bedrooms': 1, 'max_bedrooms': 3, 'min_sqft': 600, 'max_sqft': 1800},
    'townhouse': {'min_bedrooms': 2, 'max_bedrooms': 4, 'min_sqft': 1200, 'max_sqft': 2500},
    'studio': {'min_bedrooms': 0, 'max_bedrooms': 1, 'min_sqft': 300, 'max_sqft': 700}
}

# City data with approximate rent multipliers
CITIES = {
    'Mumbai': {'state': 'MH', 'rent_multiplier': 3.2, 'zip_codes': ['400001', '400002', '400003', '400004', '400005']},
    'Delhi': {'state': 'DL', 'rent_multiplier': 2.8, 'zip_codes': ['110001', '110002', '110003', '110004', '110005']},
    'Bangalore': {'state': 'KA', 'rent_multiplier': 2.5, 'zip_codes': ['560001', '560002', '560003', '560004', '560005']},
    'Hyderabad': {'state': 'TG', 'rent_multiplier': 2.0, 'zip_codes': ['500001', '500002', '500003', '500004', '500005']},
    'Chennai': {'state': 'TN', 'rent_multiplier': 2.2, 'zip_codes': ['600001', '600002', '600003', '600004', '600005']},
    'Pune': {'state': 'MH', 'rent_multiplier': 2.1, 'zip_codes': ['411001', '411002', '411003', '411004', '411005']},
    'Kolkata': {'state': 'WB', 'rent_multiplier': 2.0, 'zip_codes': ['700001', '700002', '700003', '700004', '700005']},
    'Ahmedabad': {'state': 'GJ', 'rent_multiplier': 1.8, 'zip_codes': ['380001', '380002', '380003', '380004', '380005']},
    'Jaipur': {'state': 'RJ', 'rent_multiplier': 1.6, 'zip_codes': ['302001', '302002', '302003', '302004', '302005']},
    'Lucknow': {'state': 'UP', 'rent_multiplier': 1.5, 'zip_codes': ['226001', '226002', '226003', '226004', '226005']}
}


AMENITIES = [
    'Lift', 'Power Backup', 'Reserved Parking', 'Security Guard', 'Modular Kitchen',
    '24x7 Water Supply', 'Gated Society', 'CCTV Surveillance', 'Piped Gas',
    'Vaastu Compliant', 'Club House', 'Children\'s Play Area', 'Intercom Facility',
    'Gymnasium', 'Swimming Pool', 'Community Hall', 'Rain Water Harvesting',
    'Maintenance Staff', 'Visitor Parking', 'Shopping Centre', 'Park View',
    'Furnished', 'Semi-Furnished', 'Unfurnished', 'RO Water System'
]

def generate_property_listing():
    """Generate a single synthetic property listing"""
    
    # Choose random property type and city
    property_type = random.choice(list(PROPERTY_TYPES.keys()))
    city_name = random.choice(list(CITIES.keys()))
    city_info = CITIES[city_name]
    
    # Generate basic property details
    type_specs = PROPERTY_TYPES[property_type]
    bedrooms = random.randint(type_specs['min_bedrooms'], type_specs['max_bedrooms'])
    bathrooms = max(1, bedrooms + random.randint(-1, 1))
    sqft = random.randint(type_specs['min_sqft'], type_specs['max_sqft'])
    
    # Calculate rent based on city, size, and random factors
    base_rent = (sqft * 1.5) * city_info['rent_multiplier']
    rent_variation = random.uniform(0.8, 1.2)  # ±20% variation
    monthly_rent = int(base_rent * rent_variation)
    
    # Generate address
    street_address = fake.street_address()
    zip_code = random.choice(city_info['zip_codes'])
    
    # Select random amenities (3-8 amenities per property)
    num_amenities = random.randint(3, 8)
    selected_amenities = random.sample(AMENITIES, num_amenities)
    
    # Generate listing details
    available_date = fake.date_between(start_date='today', end_date='+90d')
    lease_term = random.choice([6, 12, 18, 24])  # months
    
    # Generate description
    descriptions = [
        f"Beautiful {property_type} in the heart of {city_name}",
        f"Modern {property_type} with stunning city views",
        f"Spacious {property_type} perfect for urban living",
        f"Newly renovated {property_type} in prime location",
        f"Luxury {property_type} with premium amenities"
    ]
    
    base_description = random.choice(descriptions)
    amenity_text = f"Features include: {', '.join(selected_amenities[:3])}"
    
    if bedrooms > 0:
        bedroom_text = f"{bedrooms} bedroom, {bathrooms} bathroom"
    else:
        bedroom_text = "Studio"
    
    description = f"{base_description}. {bedroom_text} spanning {sqft} sq ft. {amenity_text}."
    
    # Generate listing ID
    listing_id = f"SP{random.randint(100000, 999999)}"
    
    return {
        'listing_id': listing_id,
        'property_type': property_type,
        'title': f"{bedroom_text} {property_type.title()} in {city_name}",
        'description': description,
        'monthly_rent': monthly_rent,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft': sqft,
        'street_address': street_address,
        'city': city_name,
        'state': city_info['state'],
        'zip_code': zip_code,
        'available_date': available_date.strftime('%Y-%m-%d'),
        'lease_term_months': lease_term,
        'amenities': selected_amenities + [
            'pet_friendly' if 'Pet Friendly' in selected_amenities else None,
            'parking_available' if 'Parking' in selected_amenities else None,
            'furnished' if 'Furnished' in selected_amenities else None,
            'utilities_included' if 'Utilities Included' in selected_amenities else None
        ],
        'pet_friendly': 'Pet Friendly' in selected_amenities,
        'parking_available': 'Parking' in selected_amenities,
        'furnished': 'Furnished' in selected_amenities,
        'utilities_included': 'Utilities Included' in selected_amenities,
        'created_at': datetime.now().isoformat(),
        'contact_email': fake.email(),
        'contact_phone': fake.phone_number()
    }

def generate_dataset(num_listings=1000):
    """Generate a complete synthetic dataset"""
    print(f"Generating {num_listings} synthetic property listings...")
    
    listings = []
    for i in range(num_listings):
        if i % 100 == 0:
            print(f"Generated {i} listings...")
        
        listing = generate_property_listing()
        listings.append(listing)
    
    df = pd.DataFrame(listings)
    
    # Add some calculated fields
    df['price_per_sqft'] = df['monthly_rent'] / df['sqft']
    df['full_address'] = df['street_address'] + ', ' + df['city'] + ', ' + df['state'] + ' ' + df['zip_code']
    
    print(f"Dataset generation complete! Created {len(df)} listings.")
    print(f"Dataset shape: {df.shape}")
    print(f"Cities covered: {df['city'].nunique()}")
    print(f"Property types: {df['property_type'].value_counts().to_dict()}")
    
    return df

def save_dataset(df, filename='spacematch_properties.csv'):
    """Save dataset to CSV file"""
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    
    # Also save as JSON for easier processing
    json_filename = filename.replace('.csv', '.json')
    df.to_json(json_filename, orient='records', indent=2)
    print(f"Dataset also saved as {json_filename}")
    
    return filename, json_filename

def generate_summary_stats(df):
    """Generate and display summary statistics"""
    print("\n=== DATASET SUMMARY ===")
    print(f"Total listings: {len(df)}")
    print(f"Average rent: ${df['monthly_rent'].mean():.2f}")
    print(f"Rent range: ${df['monthly_rent'].min()} - ${df['monthly_rent'].max()}")
    print(f"Average sqft: {df['sqft'].mean():.0f}")
    print(f"Cities: {', '.join(df['city'].unique())}")
    print(f"Property types: {', '.join(df['property_type'].unique())}")
    print(f"Bedroom distribution:")
    print(df['bedrooms'].value_counts().sort_index())
    print(f"Most common amenities:")
    all_amenities = []
    for amenity_list in df['amenities']:
        all_amenities.extend(amenity_list)
    from collections import Counter
    amenity_counts = Counter(all_amenities)
    for amenity, count in amenity_counts.most_common(10):
        print(f"  {amenity}: {count} listings")

if __name__ == "__main__":
    # Generate the dataset
    df = generate_dataset(1000)
    
    # Save the dataset
    csv_file, json_file = save_dataset(df)
    
    # Generate summary statistics
    generate_summary_stats(df)
    
    print(f"\nFiles created:")
    print(f"- {csv_file}")
    print(f"- {json_file}")
    print(f"\nDataset is ready for vector store loading!")