import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("image")
args = parser.parse_args()
file = open(args.image, "rb")
r = requests.post("http://127.0.0.1:5000/predict", files={'file': file})
print(r.text)
