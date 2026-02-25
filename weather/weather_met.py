import requests
from datetime import datetime, timezone

MET_BASE_URL = "https://api.met.no/weatherapi/locationforecast/"
MET_COMPACT_URL = "2.0/compact"

def parse_iso8601_utc(timestamp):
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def get_current_weather(timeseries):
    now_utc = datetime.now(timezone.utc)
    parsed = []

    for item in timeseries:
        item_time = parse_iso8601_utc(item["time"])
        parsed.append((item_time, item))

    parsed.sort(key=lambda x: x[0])
    past_or_now = [entry for entry in parsed if entry[0] <= now_utc]
    if past_or_now:
        return past_or_now[-1][1]

    return parsed[0][1]


def fetchCompact(lat, lon, altitude):
    MET_PARAMS = {
        "lat": lat,
        "lon": lon,
        "altitude": altitude
    }
    MET_HEADERS = {
        "User-Agent": "EiT-TDT4861-G6/1.0 (student project)"
    }

    req = requests.get(
        MET_BASE_URL + MET_COMPACT_URL,
        params=MET_PARAMS,
        headers=MET_HEADERS,
        timeout=10
    )

    if not req.ok:
        print(f"[ERROR] Error retrieving request: {req.status_code}")
        return None

    payload = req.json()
    timeseries = payload["properties"]["timeseries"]
    current = get_current_weather(timeseries)

    print("Current weather timeslot:", current["time"])
    print("Details:", current["data"]["instant"]["details"])
    return current

def getTempHumid(lat,lon,altitude):
    res = fetchCompact(lat,lon,altitude)
    data = res["data"]["instant"]["details"]
    temp = data["air_temperature"]
    humidity = data["relative_humidity"]
    return {"temp": temp, "humidity": humidity}

getTempHumid(
    63.417833,
    10.407466,
    100
)
