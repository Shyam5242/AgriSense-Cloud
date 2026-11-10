"""
sensors.py - SIMULATED IoT field sensors for AgriSense Cloud.

Implements the paper's "IoT Sensor Simulation" section: readings follow
  * a daily temperature cycle (coolest near dawn, warmest mid-afternoon),
  * a seasonal temperature baseline,
  * humidity that falls as temperature rises,
  * soil moisture that dries out with heat and jumps after rain events,
    with rain more likely in the monsoon months (June-December).
Everything here is synthetic. Replace with a real sensor feed in production.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

COLUMNS = ["timestamp", "soil_moisture_pct", "temperature_c", "humidity_pct"]


def simulate_sensors(hours: int = 48, step_minutes: int = 60, start: datetime | None = None,
                     seed: int | None = None) -> pd.DataFrame:
    """Return a DataFrame with COLUMNS covering `hours` hours, one row every `step_minutes`."""
    if hours < 1 or step_minutes < 1:
        raise ValueError("hours and step_minutes must be at least 1")
    rng = np.random.default_rng(seed)
    n = max(1, int(hours * 60 / step_minutes))
    if start is None:
        start = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours)
    times = [start + timedelta(minutes=i * step_minutes) for i in range(n)]
    hour = np.array([t.hour + t.minute / 60 for t in times])
    doy = np.array([t.timetuple().tm_yday for t in times])
    month = np.array([t.month for t in times])
    dt_h = step_minutes / 60

    seasonal = 28 + 4 * np.cos(2 * np.pi * (doy - 135) / 365)        # warmest around mid-May
    diurnal = 5 * np.sin(2 * np.pi * (hour - 9) / 24)                 # peak at 15:00, minimum at 03:00
    temp = seasonal + diurnal + rng.normal(0, 0.4, n)
    humidity = np.clip(72 - 2.2 * (temp - seasonal) + rng.normal(0, 2.0, n), 25, 98)

    moisture = np.empty(n)
    m = float(rng.uniform(26, 36))
    monsoon = np.isin(month, [6, 7, 8, 9, 10, 11, 12])
    for i in range(n):
        evaporation = 0.012 * max(temp[i] - 15.0, 0.0) * dt_h
        rain = rng.uniform(2.0, 6.0) if rng.random() < (0.06 if monsoon[i] else 0.01) * dt_h else 0.0
        m = float(np.clip(m - evaporation + rain, 8.0, 50.0))
        moisture[i] = m

    return pd.DataFrame({
        "timestamp": times,
        "soil_moisture_pct": np.round(moisture, 1),
        "temperature_c": np.round(temp, 1),
        "humidity_pct": np.round(humidity, 1),
    })
