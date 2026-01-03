from datetime import datetime
from .haversine import haversine_distance

PEAK_HOURS = {7, 8, 9, 16, 17, 18}

def build_features(
    pickup_date,
    pickup_time,
    pickup_lat,
    pickup_lon,
    dropoff_lat,
    dropoff_lon,
    passenger_count
):
    pickup_dt = datetime.combine(pickup_date, pickup_time)

    hour = pickup_dt.hour
    day_of_week = pickup_dt.weekday()
    is_weekend = int(day_of_week >= 5)
    is_peak_hour = int(hour in PEAK_HOURS)

    distance_km = haversine_distance(
        pickup_lat, pickup_lon,
        dropoff_lat, dropoff_lon
    )

    wide_features = [
        is_peak_hour,
        is_weekend,
        passenger_count
    ]

    deep_features = [
        distance_km,
        hour,
        day_of_week
    ]

    return wide_features, deep_features, distance_km
