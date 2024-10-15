import numpy as np

class load_generator:
    @staticmethod
    def load():
        num_points=72
        x = np.linspace(0, 24, num_points)

        def gaussian(x, mu, sigma):
            return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        base_load = 15 + 5 * np.sin(np.pi * x / 12) 

        is_weekday = np.random.choice([True, False]) 
        if is_weekday:
            weekday_multiplier = np.random.uniform(1.05, 1.15) 
            base_load *= weekday_multiplier

        seasonal_effect = np.random.uniform(-2, 4)
        base_load += seasonal_effect * np.sin(np.pi * x / 12 - np.pi/2) 

        day_peak_time = np.random.uniform(11, 13)
        day_peak_magnitude = np.random.uniform(7, 10) 

        night_peak_time = np.random.uniform(20, 22)
        night_peak_magnitude = np.random.uniform(12, 15) 

        day_peak = gaussian(x, day_peak_time, 1) * day_peak_magnitude
        night_peak = gaussian(x, night_peak_time, 1.2) * night_peak_magnitude 

        morning_ramp = np.clip(x - 5, 0, 6) / 6 * 7
        evening_ramp = np.clip(18 - x, 0, 12) / 12 * 7 

        extreme_weather_event = np.random.choice([0, 1], p=[0.95, 0.05]) 
        if extreme_weather_event:
            base_load += np.random.uniform(3, 7)

        load_values = base_load + day_peak + night_peak + morning_ramp + evening_ramp

        high_freq_noise = np.random.uniform(-1.5, 1.5, num_points) 
        load_values += high_freq_noise

        min_load = np.min(load_values)
        max_load = np.max(load_values)
        desired_min = 50
        desired_max = 100
        scale_factor = (desired_max - desired_min) / (max_load - min_load)
        load_values = (load_values - min_load) * scale_factor + desired_min
        return load_values