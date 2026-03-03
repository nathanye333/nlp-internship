import sys
import mysql.connector
import pandas as pd
import nltk
def test_python_version():
    assert sys.version_info >= (3, 11), "Python 3.11+ required"
    print("✓ Python version OK")
def test_packages():
    print("✓ All required packages installed")
def test_docker():
    try:
        conn = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='root',
        port=3308
        )
        conn.close()
        print("✓ MySQL connection successful")
    except Exception as e:
        print("⚠ MySQL connection failed:", e)
if __name__ == "__main__":
    test_python_version()
    test_packages()
    test_docker()
print("\nSetup verification complete!")