"""
Day 01 — Project Setup & Structure
Week 1: Foundations
"""
import os

FOLDERS = ["models", "data/raw", "data/processed", "notebooks", "utils", "tests"]

def setup():
    print("🚀 AICryptoPredictor — Day 01: Project Setup")
    print("=" * 50)
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)
        print(f"  ✅ Created: {folder}/")
    print("\n📦 Project structure ready!")
    print("💡 Next: pip install -r requirements.txt")

if __name__ == "__main__":
    setup()
