import spikeinterface.preprocessing as spr
from src.config import (
    RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EVENT_SUFFIXES, 
    EPOCH_T_PRE, EPOCH_T_POST, BASELINE_T_START, BASELINE_T_END, FREQ_BANDS
)
from src.io import load_binary_session, load_lfp_recording, load_envelopes
from scipy.signal import hilbert
import pywt
from scipy.ndimage import uniform_filter1d
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_and_save_lfp(subject, session, n_jobs=1):
    """
    Extracts Local Field Potentials (LFPs) following SpikeInterface best practices.
    Saves intermediate results to disk for efficiency.
    """
    recording = load_binary_session(subject, session)
    recording_uV = spr.scale(recording, gain=1e6)
    
    # 1. Bandpass filter (1-250 Hz)
    # ignore_low_freq_error=True bypasses the low frequency safety check in SI.
    # The margin is automatically set to 5 seconds.
    recording_bp = spr.bandpass_filter(
        recording_uV, 
        freq_min=1.0, 
        freq_max=250.0, 
        ignore_low_freq_error=True 
    )

    # 2. Downsample to an intermediate frequency (e.g., 1000 Hz)
    # Respects Nyquist theorem for the high_gamma band (250 Hz)
    recording_resampled = spr.resample(recording_bp, resample_rate=1000)

    # 3. Apply Common Median Reference (CMR) at 1000 Hz
    recording_cmr = spr.common_reference(
        recording_resampled, 
        reference='global', 
        operator='median'
    )

    # 4. SAVE TO DISK (Crucial step required by documentation)
    # Use 30-second chunks to minimize the 5-second margin overhead.
    lfp_folder = INTERIM_DATA_DIR / subject / session / "lfp_1000Hz"
    
    # If the folder exists, load directly to avoid recomputing
    if lfp_folder.exists():
        print(f"LFP for {subject}/{session} already existing, cancel the data to overwrite.")
    else:
        print(f"Computing and saving LFP in 30s chunks. Please wait...")
        recording_cmr.save(
            folder=lfp_folder,
            chunk_duration="30s", # Prevents memory overload
            n_jobs=n_jobs,            # Use all available CPU cores
            progress_bar=True
        )
    
    return 

def extract_and_save_envelopes(subject: str, session: str) -> None:
    """
    Extracts frequency band envelopes using Hilbert transform, smooths them, 
    downsamples to 100 Hz, and saves the output to a generic .npz file.
    """
    out_folder = INTERIM_DATA_DIR / subject / session
    out_path = out_folder / "band_envelopes_100Hz.npz"
    
    # If the file exists, skip to avoid recomputing
    if out_path.exists():
        print(f"Envelopes for {subject}/{session} already existing, cancel the data to overwrite.")
        return

    print(f"Computing and saving envelopes to .npz. Please wait...")
    
    # Load the intermediate LFP recording (0-300Hz)
    recording_lfp = load_lfp_recording(subject, session, "lfp_1000Hz")
    fs_lfp = recording_lfp.get_sampling_frequency()
    
    num_channels = recording_lfp.get_num_channels()
    num_samples = recording_lfp.get_num_samples()
    smooth_window_samples = int(0.5 * fs_lfp) # 500 ms window for smoothing
    target_fs = 100.0
    ds_factor = int(fs_lfp / target_fs) # Downsampling factor to reach the target 100 Hz for sampling the envelopes
    num_samples_ds = len(np.arange(0, num_samples, ds_factor))
    envelopes_100hz = {}
    
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        # 1. Lazy bandpass filter via SpikeInterface
        recording_bp = spr.bandpass_filter(
            recording_lfp,
            freq_min=fmin,
            freq_max=fmax,
            filter_order=5,
            ignore_low_freq_error=True
        )

        # Iterate over channels to save RAM and avoid Hilbert chunking artifacts
        band_envelope_ds = np.zeros((num_samples_ds, num_channels), dtype=np.float32)         
        for ch_idx in tqdm(range(num_channels), desc=f"Channels ({session}/{band_name})", leave=False):
            channel_id = recording_bp.channel_ids[ch_idx]
            trace_ch = recording_bp.get_traces(channel_ids=[channel_id], return_scaled=False)
            trace_ch = trace_ch[:, 0]
            analytic_signal = hilbert(trace_ch)
            envelope = np.abs(analytic_signal) # Envelope extraction (magnitude of the analytic signal via Hilbert)
            smoothed = uniform_filter1d(envelope, size=smooth_window_samples) # Smoothing (500-ms moving mean filter)
            band_envelope_ds[:, ch_idx] = smoothed[::ds_factor] - np.mean(smoothed[::ds_factor]) # Downsample to 100 Hz and mean centering
            
        envelopes_100hz[band_name] = band_envelope_ds
        
    # Save output to disk in .npz dict
    out_folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **envelopes_100hz)
    
    return

def epoch_and_normalize_envelopes(subject: str, session: str, event_type: str = 'grasp', thr_der: float = 3.0, fs: float = 100.0, max_bad_channels: float = 0.0) -> None:
    """
    Extracts epochs around events, computes ERD/ERS Z-score normalization based on a 
    pre-event baseline, removes artifactual trials using amplitude thresholds, 
    and saves the extracted 3D tensors.
    """
    envelopes = load_envelopes(subject, session)
    bands = list(envelopes.keys())
    
    events_dir = RAW_DATA_DIR / subject / session / "Events"
    csv_file = events_dir / f"{session}{EVENT_SUFFIXES[event_type]}"
    
    if not csv_file.exists():
        print(f"No '{event_type}' events found for {subject}/{session}.")
        return
        
    df_events = pd.read_csv(csv_file)
    
    # Extract event timestamps and construct labels for decoding
    if event_type == 'grasp':
        timestamps = df_events['EventTime'].values
        labels = df_events['Target'].fillna('unknown').astype(str) + "_" + df_events['Hand'].fillna('unknown').astype(str)
        labels = labels.values
        baseline_timestamps = timestamps
    elif event_type == 'steps':
        # Every step is an event, but use the start of the walking sequence as the baseline reference for all steps in that sequence
        timestamps = df_events['StepTime'].values
        labels = df_events['StepType'].fillna('unknown').astype(str) + "_" + \
                 df_events['Hand'].fillna('unknown').astype(str) + "_" + \
                 df_events['Surface'].fillna('unknown').astype(str)
        labels = labels.values
        baseline_timestamps = df_events.groupby('WalkNumber')['StepTime'].transform('min').values

    # Load and apply manual artifact mask
    bad_trials_file = PROCESSED_DATA_DIR / subject / session / f"bad_trials_{event_type}.csv"
    if bad_trials_file.exists():
        df_bad = pd.read_csv(bad_trials_file)
        valid_mask = ~df_bad['is_artifact'].values.astype(bool)
        
        timestamps = timestamps[valid_mask]
        labels = labels[valid_mask]
        baseline_timestamps = baseline_timestamps[valid_mask]

    # Pre-compute indices 
    samples_pre = int(EPOCH_T_PRE * fs)
    samples_post = int(EPOCH_T_POST * fs)

    raw_epochs = []
    valid_labels_pass1 = []
    baseline_means = {band: {} for band in bands}
    baseline_stds = {band: {} for band in bands}
    
    
    for i, (t, t_base) in enumerate(zip(timestamps, baseline_timestamps)):
        idx = int(t * fs)
        idx_base = int(t_base * fs)
        base_start_idx = idx_base + int(BASELINE_T_START * fs)
        base_end_idx = idx_base + int(BASELINE_T_END * fs)
        # Discard epochs or baselines extending beyond the recording boundaries
        if (idx - samples_pre < 0 or idx + samples_post > envelopes[bands[0]].shape[0] or
            base_start_idx < 0 or base_end_idx > envelopes[bands[0]].shape[0]):
            continue
            
        epoch_dict = {}
        for band in bands:
            # Extract temporal window [Samples, Channels]
            epoch = envelopes[band][idx - samples_pre : idx + samples_post, :].copy()
            epoch_dict[band] = epoch
            
            if t_base not in baseline_means[band]:
                baseline = envelopes[band][base_start_idx:base_end_idx, :]
                baseline_means[band][t_base] = np.mean(baseline, axis=0)
                baseline_stds[band][t_base] = np.std(baseline, axis=0)
            
        raw_epochs.append(epoch_dict)
        valid_labels_pass1.append(labels[i])
        
    if not raw_epochs:
        print("No valid epochs extracted.")
        return

    # 2. Compute robust session-level baseline metrics (average across all trials)
    session_base_mean = {}
    session_base_std = {}
    
    for band in bands:
        session_base_mean[band] = np.mean(list(baseline_means[band].values()), axis=0)
        
        # Average the standard deviations and prevent division by zero
        mean_std = np.mean(list(baseline_stds[band].values()), axis=0)
        mean_std[mean_std == 0] = 1.0
        session_base_std[band] = mean_std
        
    # 3. Second pass: normalize with global baseline and reject artifacts
    epoched_data = {band: [] for band in bands}
    final_labels = []
    num_channels = raw_epochs[0][bands[0]].shape[1]
    for i, epoch_dict in enumerate(raw_epochs):
        is_artifact = False
        norm_epoch_dict = {}
        bad_channels_mask = np.zeros(num_channels, dtype=bool)
        for band in bands:
            # Z-score using session-level metrics
            epoch_z = (epoch_dict[band] - session_base_mean[band]) / session_base_std[band]
            norm_epoch_dict[band] = epoch_z
            
            # First-derivative artifact rejection (difference across the time axis)
            derivative = np.diff(epoch_z, axis=0)
            bad_in_this_band = np.any(np.abs(derivative) >= thr_der, axis=0)
            bad_channels_mask = bad_channels_mask | bad_in_this_band
        total_bad_channels = np.sum(bad_channels_mask)/num_channels
        
        # Reject trial if the number of bad channels exceeds the threshold
        if total_bad_channels <= max_bad_channels:
            for band in bands:
                epoched_data[band].append(norm_epoch_dict[band])
            final_labels.append(valid_labels_pass1[i])
            
    # Cast lists into continuous numpy arrays
    for band in bands:
        epoched_data[band] = np.array(epoched_data[band], dtype=np.float32)
        
    rejected_count = len(timestamps) - len(final_labels)
    print(f"Extracted {len(final_labels)} valid epochs. Rejected {rejected_count} (exceeded more than {max_bad_channels}% bad channels).")
    
    out_folder = PROCESSED_DATA_DIR / subject / session
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / f"epoched_{event_type}_zscored.npz"
    np.savez_compressed(out_path, labels=np.array(final_labels), **epoched_data)
    
    return

def extract_cwt_epochs(subject: str, session: str, event_type: str = 'grasp', pad_s: float = 2.0, target_fs: float = 100.0, n_freqs: int = 50) -> None:
    """
    Extracts epoched LFP data, computes Continuous Wavelet Transform (Morlet) 
    on padded windows to prevent edge artifacts, and downsamples the power.
    Saves the 4D tensor (trials, freqs, time, channels) to an .npz file.
    """
    events_dir = RAW_DATA_DIR / subject / session / "Events"
    csv_file = events_dir / f"{session}{EVENT_SUFFIXES.get(event_type)}"
    
    if not csv_file or not csv_file.exists():
        print(f"No '{event_type}' events found for {subject}/{session}.")
        return

    df_events = pd.read_csv(csv_file)
    
    # Construct timestamps and labels
    if event_type == 'grasp':
        timestamps = df_events['EventTime'].values
        labels = df_events['Target'].fillna('unknown').astype(str) + "_" + df_events['Hand'].fillna('unknown').astype(str)
        labels = labels.values
        baseline_timestamps = timestamps
    elif event_type == 'steps':
        timestamps = df_events['StepTime'].values
        labels = df_events['StepType'].fillna('unknown').astype(str) + "_" + \
                 df_events['Hand'].fillna('unknown').astype(str) + "_" + \
                 df_events['Surface'].fillna('unknown').astype(str)
        labels = labels.values
        baseline_timestamps = df_events.groupby('WalkNumber')['StepTime'].transform('min').values

    # Load and apply manual artifact mask
    bad_trials_file = PROCESSED_DATA_DIR / subject / session / f"bad_trials_{event_type}.csv"
    if bad_trials_file.exists():
        df_bad = pd.read_csv(bad_trials_file)
        valid_mask = ~df_bad['is_artifact'].values.astype(bool)
        
        timestamps = timestamps[valid_mask]
        labels = labels[valid_mask]
        baseline_timestamps = baseline_timestamps[valid_mask]

    # Load 1000Hz LFP data
    recording_lfp = load_lfp_recording(subject, session, "lfp_1000Hz")
    fs_lfp = recording_lfp.get_sampling_frequency()
    num_channels = recording_lfp.get_num_channels()
    total_samples = recording_lfp.get_num_samples()

    # CWT Parameters setup (Logarithmic spacing from 1 Hz to 250 Hz)
    freqs = np.logspace(np.log10(2.0), np.log10(250.0), num=n_freqs)
    
    # Time and downsampling parameters
    samples_pre_pad = int((EPOCH_T_PRE + pad_s) * fs_lfp)
    samples_post_pad = int((EPOCH_T_POST + pad_s) * fs_lfp)
    ds_factor = int(fs_lfp / target_fs)
    pad_ds = int(pad_s * target_fs)
    smooth_window = int(0.05 * fs_lfp) # smoothing window of 50 ms for the power time series after CWT

    # Define Complex Morlet wavelet: Bandwidth=1.5, Center Frequency=1.0 Hz
    wavelet_name = 'cmor1.5-1.0'
    sampling_period = 1.0 / fs_lfp

    # Back-calculate wavelet scales from target frequencies (2 to 250 Hz)
    center_freq = pywt.central_frequency(wavelet_name)
    scales = center_freq / (freqs * sampling_period)

    raw_trial_tfrs = []
    valid_labels_pass1 = []
    baseline_means = {}
    baseline_stds = {}

    for i, (t, t_base) in enumerate(tqdm(zip(timestamps, baseline_timestamps), desc=f"Processing {event_type} CWT", total=len(timestamps))):
        idx = int(t * fs_lfp)
        
        # Calculate baseline boundaries
        base_start_idx = int(t_base * fs_lfp) + int(BASELINE_T_START * fs_lfp)
        base_end_idx = int(t_base * fs_lfp) + int(BASELINE_T_END * fs_lfp)
        
        # Boundary check for both epoch and baseline padded windows
        if (idx - samples_pre_pad < 0) or (idx + samples_post_pad > total_samples) or \
           (base_start_idx - samples_pre_pad < 0) or (base_end_idx + samples_post_pad > total_samples):
            continue
            
        # Compute CWT for Baseline
        if t_base not in baseline_means:
            pad_base_epoch = recording_lfp.get_traces(
                start_frame=base_start_idx - pad_s * fs_lfp, 
                end_frame=base_end_idx + pad_s * fs_lfp, 
                return_scaled=False
            )
            n_times_pad_ds = len(np.arange(0, pad_base_epoch.shape[0], ds_factor))
            base_tfr = np.zeros((n_freqs, n_times_pad_ds, num_channels), dtype=np.float32)
            
            for ch in range(num_channels):
                cwt_mat, frequencies = pywt.cwt(pad_base_epoch[:, ch], scales, wavelet_name, sampling_period=sampling_period)
                power = np.abs(cwt_mat)**2
                base_tfr[:, :, ch] = uniform_filter1d(power, size=smooth_window, axis=1)[:, ::ds_factor]
                
            base_tfr_trimmed = base_tfr[:, pad_ds:-pad_ds, :]
            # Compute mean/std keeping dims for broadcasting -> (n_freqs, 1, num_channels)
            baseline_means[t_base] = np.mean(base_tfr_trimmed, axis=1, keepdims=True)
            baseline_stds[t_base] = np.std(base_tfr_trimmed, axis=1, keepdims=True)

        # --- 2. Compute CWT for Epoch ---
        padded_epoch = recording_lfp.get_traces(
            start_frame=idx - samples_pre_pad, 
            end_frame=idx + samples_post_pad, 
            return_scaled=False
        )
        n_times_pad_ds = len(np.arange(0, padded_epoch.shape[0], ds_factor))
        trial_tfr = np.zeros((n_freqs, n_times_pad_ds, num_channels), dtype=np.float32)

        for ch in range(num_channels):
            cwt_mat, _ = pywt.cwt(padded_epoch[:, ch], scales, wavelet_name, sampling_period=sampling_period)
            power = np.abs(cwt_mat)**2
            trial_tfr[:, :, ch] = uniform_filter1d(power, size=smooth_window, axis=1)[:, ::ds_factor]
            
        trial_tfr_trimmed = trial_tfr[:, pad_ds:-pad_ds, :]
        
        raw_trial_tfrs.append(trial_tfr_trimmed)
        valid_labels_pass1.append(labels[i])

    if not raw_trial_tfrs:
        print("No valid epochs extracted.")
        return
        
    # --- PASS 2: Robust session-level baseline and Z-score ---
    # Average baseline metrics across all unique baseline periods
    session_base_mean = np.mean(list(baseline_means.values()), axis=0) # Shape: (n_freqs, 1, num_channels)
    session_base_std = np.mean(list(baseline_stds.values()), axis=0)
    session_base_std[session_base_std == 0] = 1.0 # Prevent div by zero
    
    epoched_cwt = []
    for trial_tfr in raw_trial_tfrs:
        # Broadcasting automatically handles the temporal dimension
        trial_zscored = (trial_tfr - session_base_mean) / session_base_std
        epoched_cwt.append(trial_zscored)

    # Tensor shape: (n_trials, n_freqs, n_times, n_channels)
    epoched_cwt_arr = np.stack(epoched_cwt)
    valid_labels_arr = np.array(valid_labels_pass1)

    # Save output
    out_folder = PROCESSED_DATA_DIR / subject / session
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / f"epoched_cwt_{event_type}.npz"
    
    np.savez_compressed(
        out_path, 
        cwt_tensor=epoched_cwt_arr, 
        labels=valid_labels_arr,
        freqs=frequencies
    )
    print(f"Saved CWT tensor {epoched_cwt_arr.shape} to {out_path}")