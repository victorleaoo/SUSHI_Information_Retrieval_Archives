import os

from data_loader import DataLoader

if __name__ == "__main__":
    dl = DataLoader(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))