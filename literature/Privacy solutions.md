

### 1. The Signal Layer: Time-Series Differential Privacy (TS-DP)

- **Objective:** To sanitize high-frequency physiological signals at the source, stripping unique biometric identifiers while preserving the utility of the data for fatigue and load analysis.
    
- **Target Data:** **Electromyography (EMG)** sensors (sampling at **4370 Hz**) and **Heart Rate Monitors**. The extreme sampling rate of the EMG sensors captures unique neuromuscular firing patterns that act as a biological fingerprint, necessitating rigorous signal masking.
    
- **Technical Mechanism:** Implementation of the **Fourier Perturbation Algorithm (FPA)**.
    
    1. **Transformation:** The raw time-series data is converted to the frequency domain using the Discrete Fourier Transform (DFT).
        
    2. **Perturbation:** Calibrated noise (Laplace) is injected _only_ into the first $k$ Fourier coefficients (low-frequency components). These coefficients represent the general trend and magnitude of muscle activation (the utility).
        
    3. **Truncation:** High-frequency coefficients, which contain the fine-grained "jitter" or noise specific to an individual's biology (the privacy risk), are discarded.
        
    4. **Reconstruction:** The signal is reconstructed via Inverse DFT.
        
- **Thesis Application:** This allows for the storage and analysis of muscle fatigue trends without retaining the raw, re-identifiable bio-signals, effectively decoupling the _performance metric_ from the _biological identity_.
    

### 2. The Training Layer: Federated Learning (FL)

- **Objective:** To train robust injury prediction and performance models across multiple athletes or teams without ever centralizing the raw, sensitive tracking data.
    
- **Target Data:** **Inertial Measurement Units (IMUs)** (74 sensors @ **240 Hz**) and **Markerless Optical Tracking** (48 cameras). The volume of this data (1.1 GB/s per camera) and its sensitivity (gait recognition) make centralized storage both computationally expensive and risky.
    
- **Technical Mechanism:** Deployment of a **Federated Averaging (FedAvg)** architecture.
    
    1. **Local Training:** A local model is initialized and trained on the edge device (or local team server) using the raw, private data.
        
    2. **Gradient Extraction:** Instead of sharing data, the system calculates the model _updates_ (gradients/weights) required to minimize error.
        
    3. **Aggregation:** Only these numerical weights are encrypted and transmitted to a central parameter server.
        
    4. **Global Update:** The server averages weights from multiple sources to update the Global Model, which is then redistributed to local devices.
        
- **Thesis Application:** This architecture ensures data minimization. The "Digital Twin" of the athlete remains in a local, secure environment (the JGHPI facility), while the collective intelligence of the model improves using data from the entire cohort.
    

### 3. The Comparative Layer: Secure Multi-Party Computation (SMPC)

- **Objective:** To enable competitive benchmarking and league-wide comparison of biomechanical metrics without revealing proprietary player data to competitors or third parties.
    
- **Target Data:** **Force Plates** (96 sensors @ **2000 Hz**) and derived metrics (e.g., Peak Ground Reaction Force, Asymmetry Indices). Teams consider this "scouting data" and are hesitant to share raw values.
    
- **Technical Mechanism:** Utilization of **Secret Sharing Schemes** (e.g., Shamir’s Secret Sharing).
    
    1. **Input Splitting:** A specific metric (e.g., a player’s Center of Pressure sway) is mathematically split into multiple random "shares." A single share reveals no information about the original value.
        
    2. **Distribution:** These shares are distributed across non-colluding compute nodes.
        
    3. **Blind Computation:** The nodes perform mathematical operations (addition, multiplication) on the shares directly.
        
    4. **Result Reconstruction:** The final outputs are combined to reveal the aggregate result (e.g., "League Average Sway") without any party ever seeing the individual inputs.
        
- **Thesis Application:** This resolves the "trust gap" in sports analytics, allowing for a "Blind Combine" where an athlete's biometrics can be ranked against the population without exposing their specific medical or physical vulnerabilities.