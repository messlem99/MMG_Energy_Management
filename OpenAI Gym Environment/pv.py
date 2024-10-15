import numpy as np

class PV_generator:
    @staticmethod
    def pv():
        num_points = 72
        x = np.linspace(0, 24, num_points)
        ge = np.random.uniform(4.3, 7.3)  
        ga = np.random.uniform(17.3, 20)

        def gaussian(x, mu, sigma):
            return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        # Random start times, peak times, and end times
        start_time1 = np.random.uniform(4.3, 8)  
        peak_time1 = np.random.uniform(11, 15)  
        end_time1 = np.random.uniform(17, 20)  

        start_time2 = np.random.uniform(4.3, 8)
        peak_time2 = np.random.uniform(11, 15)
        end_time2 = np.random.uniform(17, 20)

        start_time3 = np.random.uniform(4.3, 8)
        peak_time3 = np.random.uniform(11, 15)
        end_time3 = np.random.uniform(17, 20)

        pv1_values = gaussian(x, peak_time1, 2) * (np.clip(x, start_time1, end_time1) - start_time1)
        pv2_values = gaussian(x, peak_time2, 2) * (np.clip(x, start_time2, end_time2) - start_time2)
        pv3_values = gaussian(x, peak_time3, 2) * (np.clip(x, start_time3, end_time3) - start_time3)

        # Scale: pv1 = 0-12, pv2 = 0-10, pv3 = 0-11 
        pv1_values *= 7 / np.max(pv1_values)
        pv2_values *= 9 / np.max(pv2_values)
        pv3_values *= 6 / np.max(pv3_values)

        # Varying noise amplitudes for each panel
        noise_amplitude_pv1 = np.random.uniform(0.5, 1.2)
        noise_amplitude_pv2 = np.random.uniform(0.3, 0.8)
        noise_amplitude_pv3 = np.random.uniform(0.2, 0.5)

        pv1_values += np.random.uniform(-noise_amplitude_pv1, noise_amplitude_pv1, num_points)
        pv2_values += np.random.uniform(-noise_amplitude_pv2, noise_amplitude_pv2, num_points)
        pv3_values += np.random.uniform(-noise_amplitude_pv3, noise_amplitude_pv3, num_points)

        # Random cloud shadows with varying depth and duration
        for _ in range(np.random.randint(0, 5)):  # 0 to 4 shadows possible
            shade_start = np.random.uniform(ge + 1, ga - 1)  # Shadows only during daylight
            shade_duration = np.random.uniform(0.25, 1.5)  # Shorter, more frequent shadows
            shade_depth = np.random.uniform(0.1, 0.9)  # Varying shadow depth
            shade_mask = (x >= shade_start) & (x <= shade_start + shade_duration)
            pv1_values[shade_mask] *= (1 - shade_depth)
            pv2_values[shade_mask] *= (1 - shade_depth)
            pv3_values[shade_mask] *= (1 - shade_depth)

        # Seasonal variation with wider range
        season_factor = np.random.uniform(0.7, 1.3) 
        pv1_values *= season_factor
        pv2_values *= season_factor
        pv3_values *= season_factor

        # Ensure zero output outside daylight hours
        pv1_values = np.where(((x < ge) | (x > ga)), 0, pv1_values) 
        pv2_values = np.where(((x < ge) | (x > ga)), 0, pv2_values)
        pv3_values = np.where(((x < ge) | (x > ga)), 0, pv3_values)

        # Clip values to their respective maximums and add a small offset to avoid zero values
        pv1_values = np.clip(pv1_values, 0, 10)
        pv2_values = np.clip(pv2_values, 0, 10)
        pv3_values = np.clip(pv3_values, 0, 10)

        return pv1_values, pv2_values, pv3_values