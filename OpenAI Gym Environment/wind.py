import numpy as np

class WindTurbineGenerator:

    @staticmethod
    def wind_turbine(num_points=72):
        x = np.linspace(0, 24, num_points)

        def wind_profile(x, amplitude, frequency, phase_shift):
            return amplitude * np.sin(frequency * x + phase_shift)

        def diurnal_pattern(x, peak_time=np.random.uniform(10, 16), peak_wind=np.random.uniform(3, 7), base_wind=np.random.uniform(1, 1.5)):
            return base_wind + (peak_wind - base_wind) * np.exp(-((x - peak_time) ** 2) / (2 * 3 ** 2))

        def seasonal_variation(month, max_amplitude):
            return max_amplitude * (1 + 0.5 * np.sin((month - 1) * np.pi / 6))

        month = np.random.choice(range(1, 13)) 
        amp1 = seasonal_variation(month, np.random.uniform(2, 7))
        freq1 = np.random.uniform(0.1, 0.3)
        phase1 = np.random.uniform(0, 2 * np.pi)

        amp2 = seasonal_variation(month, np.random.uniform(3, 13))
        freq2 = np.random.uniform(0.1, 0.3)
        phase2 = np.random.uniform(0, 2 * np.pi)

        amp3 = seasonal_variation(month, np.random.uniform(4, 12))
        freq3 = np.random.uniform(0.1, 0.3)
        phase3 = np.random.uniform(0, 2 * np.pi)

        wt1_values = wind_profile(x, amp1, freq1, phase1)
        wt2_values = wind_profile(x, amp2, freq2, phase2)
        wt3_values = wind_profile(x, amp3, freq3, phase3)

        diurnal_effect = diurnal_pattern(x)
        wt1_values *= diurnal_effect
        wt2_values *= diurnal_effect
        wt3_values *= diurnal_effect

        # Introducing more randomness with varying amplitudes
        noise_amplitude_wt1 = np.random.uniform(0.5, 1.2) 
        noise_amplitude_wt2 = np.random.uniform(0.8, 1.5) 
        noise_amplitude_wt3 = np.random.uniform(1.0, 2.0) 

        wt1_values += np.random.normal(0, noise_amplitude_wt1, num_points)
        wt2_values += np.random.normal(0, noise_amplitude_wt2, num_points)
        wt3_values += np.random.normal(0, noise_amplitude_wt3, num_points)

        # Introducing occasional spikes and dips
        spike_prob = 0.05
        dip_prob = 0.03

        for i in range(num_points):
            if np.random.rand() < spike_prob:
                wt1_values[i] *= np.random.uniform(1.2, 1.5)
                wt2_values[i] *= np.random.uniform(1.3, 1.6)
                wt3_values[i] *= np.random.uniform(1.4, 1.7)
            elif np.random.rand() < dip_prob:
                wt1_values[i] *= np.random.uniform(0.5, 0.8)
                wt2_values[i] *= np.random.uniform(0.4, 0.7)
                wt3_values[i] *= np.random.uniform(0.3, 0.6)

        wt1_values = np.clip(wt1_values, 0, None)
        wt2_values = np.clip(wt2_values, 0, None)
        wt3_values = np.clip(wt3_values, 0, None)

        if wt1_values.max() == wt1_values.min():
            wt1_values = np.full_like(wt1_values, 0.5) 
        else:
            wt1_values = ((wt1_values - wt1_values.min()) / (wt1_values.max() - wt1_values.min()) * (7 - 0.8) + 0.8) 

        if wt2_values.max() == wt2_values.min():
            wt2_values = np.full_like(wt2_values, 0.5) 
        else:
            wt2_values = ((wt2_values - wt2_values.min()) / (wt2_values.max() - wt2_values.min()) * (13 - 0.1) + 0.1) 

        if wt3_values.max() == wt3_values.min():
            wt3_values = np.full_like(wt3_values, 0.5)
        else:
            wt3_values = ((wt3_values - wt3_values.min()) / (wt3_values.max() - wt3_values.min()) * (12 - 0.9) + 0.9) 

        return wt1_values, wt2_values, wt3_values