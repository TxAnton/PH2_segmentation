#!/usr/bin/python3
import os

def _load():
    os.system("wget https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar")
    os.system("unrar x PH2Dataset.rar")

if __name__ == "__main__":
    _load()