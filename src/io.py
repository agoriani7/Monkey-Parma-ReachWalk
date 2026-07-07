import spikeinterface as si
from src.config import RAW_DATA_DIR, FS_ORIGINAL, NUM_CHANNELS, DTYPE, INTERIM_DATA_DIR, EVENT_SUFFIXES, PROCESSED_DATA_DIR
import numpy as np
import pandas as pd

def inspect_behavior(subject: str, session: str) -> None:
    """
    Reads behavioral CSV files (Grasp and Steps) and prints a descriptive summary.
    Adapts dynamically to the available columns in the datasets.
    """
    events_dir = RAW_DATA_DIR / subject / session / "Events"
    steps_file = events_dir / f"{session}{EVENT_SUFFIXES['steps']}"
    grasp_file = events_dir / f"{session}{EVENT_SUFFIXES['grasp']}"
    
    print(f"=== Behavior Summary: {subject} | {session} ===\n")
    
    # 1. Grasp Analysis
    if grasp_file.exists():
        df_grasp = pd.read_csv(grasp_file)
        num_grasps = len(df_grasp)
        print("[GRASP]")
        print(f"  - Total grasps recorded: {num_grasps}")
            
        if 'Target' in df_grasp.columns and 'Hand' in df_grasp.columns:
            print("  - Breakdown by Target and Hand:")
            breakdown = df_grasp.groupby(['Target', 'Hand']).size().to_string(header=False)
            print(f"    {breakdown.replace(chr(10), chr(10) + '    ')}")
    else:
        print("[GRASP]\n  - File not found.")
        
    print("\n" + "-"*45 + "\n")
    
    # 2. Walk/Steps Analysis
    if steps_file.exists():
        df_steps = pd.read_csv(steps_file)
        print("[WALK]")
        
        num_walks = df_steps['WalkNumber'].nunique()
        total_steps = len(df_steps)
        
        print(f"  - Total walks: {num_walks}")
        print(f"  - Total individual steps: {total_steps}")
        
        if num_walks > 0:
            # Calculate mean steps per walk (total steps / unique walks)
            mean_steps = total_steps / num_walks
            print(f"  - Mean steps per walk: {mean_steps:.1f}")
            
            # Calculate walk duration: max(StepTime) - min(StepTime) for each WalkNumber
            walk_durations = df_steps.groupby('WalkNumber')['StepTime'].agg(lambda x: x.max() - x.min())
            mean_walk_dur = walk_durations.mean()
            print(f"  - Mean walk duration: {mean_walk_dur:.2f} s")
            
            # Breakdown by Surface
            if 'Surface' in df_steps.columns:
                print("  - Walks breakdown by Surface:")
                # Group by walk and take the first surface to avoid counting every step
                surface_counts = df_steps.groupby('WalkNumber')['Surface'].first().value_counts()
                for surface, count in surface_counts.items():
                    print(f"    - {surface}: {count} walks")
    else:
        print("[WALK]\n  - File not found.")

def load_binary_session(subject: str, session: str) -> si.core.BinaryRecordingExtractor:
    """
    Loads a multiplexed binary file into SpikeInterface.
    
    Parameters
    ----------
    subject : str
        Subject name (e.g., 'Router').
    session : str
        Session name (e.g., 'Router_20220211').
        
    Returns
    -------
    recording : sc.BinaryRecordingExtractor
        SpikeInterface recording object ready for preprocessing.
    """
    # Construct the exact path mapped by the MATLAB script
    bin_path = RAW_DATA_DIR / subject / session / 'Wideband' / f"{session}_raw.bin"
    
    if not bin_path.exists():
        raise FileNotFoundError(f"Binary file not found at {bin_path}")

    # Load into SpikeInterface as interleaved (C-order by default)
    recording = si.core.read_binary(
        file_paths=bin_path,
        sampling_frequency=FS_ORIGINAL,
        num_channels=NUM_CHANNELS,
        dtype=DTYPE, 
        is_filtered=False
    )
    
    return recording

def load_lfp_recording(subject: str, session: str, folder_name: str) -> si.core.BaseRecording:
    """
    Load a SpikeInterface recording object previously saved to disk.
    """
    recording_path = INTERIM_DATA_DIR / subject / session / folder_name

    if not recording_path.exists():
        raise FileNotFoundError(f"Saved recording not found at {recording_path}")

    return si.load(recording_path)

def load_envelopes(subject: str, session: str) -> dict:
    """
    Utility function to load the saved envelopes for downstream analyses.
    """
    file_path = INTERIM_DATA_DIR / subject / session / "band_envelopes_100Hz.npz"

    if not file_path.exists():
        raise FileNotFoundError(f"Envelope file not found at {file_path}")

    with np.load(file_path) as data:
        return {band: data[band] for band in data.files}
    
def load_epochs(subject: str, session: str, event_type: str) -> dict:
    """
    Loads the normalized epoched tensors and corresponding labels from disk.
    Returns a dictionary containing the frequency bands and the 'labels' array.
    """
    # Construct the exact path for the processed data
    file_path = PROCESSED_DATA_DIR / subject / session / f"epoched_{event_type}_zscored.npz"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Epoched data not found at {file_path}")
        
    # Load the compressed .npz file
    with np.load(file_path, allow_pickle=True) as data:
        # Convert the NpzFile object to a standard Python dictionary to avoid lazy-loading issues
        epochs_data = {key: data[key] for key in data.files}
        
    return epochs_data

def load_cwt_epochs(subject: str, session: str, event_type: str) -> dict:
    """
    Loads the CWT epoched tensors, corresponding labels, and frequency vectors from disk.
    Returns a dictionary containing 'cwt_tensor', 'labels', and 'freqs'.
    """
    # Construct the exact path for the CWT processed data
    file_path = PROCESSED_DATA_DIR / subject / session / f"epoched_cwt_{event_type}.npz"
    
    if not file_path.exists():
        raise FileNotFoundError(f"CWT epoched data not found at {file_path}")
        
    # Load the compressed .npz file
    with np.load(file_path, allow_pickle=True) as data:
        # Convert the NpzFile object to a standard Python dictionary to avoid lazy-loading issues
        cwt_data = {key: data[key] for key in data.files}
        
    return cwt_data