"""
Generates demo student data for testing the College AI Assistant.
Run this once: python data/generate_demo_data.py
"""

import pandas as pd
import random
from datetime import date

random.seed(42)

FIRST_NAMES = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh",
               "Ananya", "Diya", "Ishita", "Kavya", "Myra", "Priya", "Riya", "Saanvi"]
LAST_NAMES = ["Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Reddy",
              "Mehta", "Joshi", "Nair", "Rao", "Chauhan", "Malhotra"]
PROGRAMS = ["Computer Science", "Electronics", "Mechanical", "Civil", "AI & ML", "Data Science"]
DEGREES = ["B.Tech", "M.Tech"]
SECTIONS = ["A", "B", "C"]
FEE_STATUS = ["Paid", "Pending", "Partially Paid"]
HOSTEL = ["Yes", "No"]

rows = []
for i in range(1, 201):  # 200 demo students
    student_id = f"MITRC{i:05d}"
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    program = random.choice(PROGRAMS)
    degree = random.choice(DEGREES)
    semester = random.randint(1, 8)
    year = (semester + 1) // 2
    section = random.choice(SECTIONS)
    email = f"{name.lower().replace(' ', '.')}{i}@mitrc.ac.in"
    phone = f"9{random.randint(100000000, 999999999)}"
    dob = date(random.randint(2001, 2006), random.randint(1, 12), random.randint(1, 28)).isoformat()
    attendance_pct = round(random.uniform(55, 99), 1)
    cgpa = round(random.uniform(5.5, 9.8), 2)
    fee_status = random.choice(FEE_STATUS)
    fee_due = 0 if fee_status == "Paid" else random.choice([15000, 30000, 45000, 60000])
    library_books = random.randint(0, 5)
    hostel = random.choice(HOSTEL)

    rows.append([
        student_id, name, program, degree, semester, year, section,
        email, phone, dob, attendance_pct, cgpa, fee_status, fee_due,
        library_books, hostel
    ])

df = pd.DataFrame(rows, columns=[
    "student_id", "name", "program", "degree", "semester", "year", "section",
    "email", "phone", "date_of_birth", "attendance_pct", "cgpa", "fee_status",
    "fee_due", "library_books", "hostel"
])

df.to_csv("data/students_demo.csv", index=False)
print(f"Generated {len(df)} demo students -> data/students_demo.csv")
