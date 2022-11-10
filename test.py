import requests

url = "http://localhost:9696/predict"

client = {"Temperature": 15.0, "Pressure": 30.47, "Humidity": 44, "WindDirection(Degrees)": 312.67, "Speed": 3.37}
response = requests.post(url, json=client).json()

print(response)