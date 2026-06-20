"""
AICryptoPredictor — main.py
"""
from datetime import date

PROJECT_START = date(2025, 6, 20)

def main():
    day = max(1, min((date.today() - PROJECT_START).days + 1, 28))
    week = ((day - 1) // 7) + 1
    print("=" * 50)
    print("🤖📈  AICryptoPredictor")
    print("=" * 50)
    print(f"  Day   : {day} / 28")
    print(f"  Week  : {week} / 4")
    print(f"  Today : {date.today()}")
    print("=" * 50)
    print("\nRun a specific day:")
    print("  python daily_progress/week1/day01setup.py")
    print("\nOr trigger the full daily update:")
    print("  python .github/scripts/daily_update.py\n")

if __name__ == "__main__":
    main()
