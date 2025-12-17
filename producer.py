import os, json, time, requests

API_KEY = "bed08bec321acd6d81b2168bbfd7b2f1"
STREAM_DIR = "stream_coba"
INTERVAL = 15

os.makedirs(STREAM_DIR, exist_ok=True)

while True:
    try:
        ts = int(time.time())

        weather = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?lat=-6.2088&lon=106.8456&appid={API_KEY}&units=metric",
            timeout=10
        ).json()

        air = requests.get(
            f"http://api.openweathermap.org/data/2.5/air_pollution?lat=-6.2088&lon=106.8456&appid={API_KEY}",
            timeout=10
        ).json()

        payload = {
            "timestamp": ts,
            "weather": weather,
            "air": air
        }

        filename = f"{STREAM_DIR}/owm_{ts}.json"
        with open(filename, "w") as f:
            json.dump(payload, f)

        print("WRITE:", filename)
        time.sleep(INTERVAL)

    except Exception as e:
        print("ERROR:", e)
        time.sleep(5)
