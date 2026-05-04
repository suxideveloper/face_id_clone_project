import sys
import os
import random
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.core.models import User, FaceEncoding, Department

def generate_users():
    db = SessionLocal()
    
    first_names = ["Azamat", "Bekzod", "Sardor", "Dilshod", "Alisher", "Jasur", "Javohir", "Umid", "Shohruh", "Rustam",
                   "Madina", "Nigina", "Sevara", "Zilola", "Malika", "Guli", "Asal", "Shahzoda", "Dildora", "Nilufar"]
    last_names = ["Abdullayev", "Karimov", "Rahimov", "Nazarov", "Yusupov", "Ahmedov", "Qodirov", "Usmonov", "Ergashev",
                  "Tursunov", "Olimov", "Sattorov", "Umarov", "Zokirov", "Jalilov"]
    departments = ["IT", "HR", "Sales", "Marketing", "Finance", "Management", "Security"]
    positions = ["Developer", "Manager", "Specialist", "Analyst", "Director", "Designer", "Engineer"]
    
    # Ensure departments exist
    for dept_name in departments:
        if not db.query(Department).filter(Department.name == dept_name).first():
            db.add(Department(id=dept_name.lower(), name=dept_name, description=f"{dept_name} bo'limi"))
    db.commit()

    users_added = 0
    
    print("Generating 100 users...")
    for i in range(1, 101):
        username = f"user_{random.randint(10000, 99999)}_{i}"
        full_name = f"{random.choice(last_names)} {random.choice(first_names)}"
        phone = f"+998{random.choice(['90', '91', '93', '94', '97', '99'])}{random.randint(1000000, 9999999)}"
        department = random.choice(departments)
        position = random.choice(positions)
        
        # Random staff rate: 0.25, 0.5, 0.75, 1.0, 1.5
        staff_rate = random.choice([0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.5])
        is_doctorant = random.choice([True, False, False, False]) # 25% chance of being a doctorant

        new_user = User(
            username=username,
            full_name=full_name,
            phone=phone,
            department=department,
            position=position,
            registered=True,
            is_doctorant=is_doctorant,
            staff_rate=staff_rate
        )
        db.add(new_user)
        
        # Generate random 128d face encoding
        random_embedding = np.random.uniform(low=-0.1, high=0.1, size=(128,)).tolist()
        
        new_encoding = FaceEncoding(
            username=username,
            embedding=random_embedding
        )
        db.add(new_encoding)
        
        users_added += 1

    try:
        db.commit()
        print(f"Successfully generated and added {users_added} users with face data!")
    except Exception as e:
        db.rollback()
        print(f"Error occurred: {e}")
    finally:
        db.close()

if __name__ == '__main__':
    generate_users()
