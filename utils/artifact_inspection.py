import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, EVENT_SUFFIXES, EPOCH_T_PRE, EPOCH_T_POST
from src.io import load_lfp_recording

def inspect_artifacts(subject: str, session: str, event_type: str = 'grasp', offset_uv: float = 200.0) -> None:
    """
    Interactive GUI for manual artifact rejection on 1000Hz LFP data.
    Displays 128 channels grouped into 4 physical arrays (32 channels each) 
    with a vertical offset for clear inspection.
    """
    # 1. Load Events and initialize tracking
    events_dir = RAW_DATA_DIR / subject / session / "Events"
    csv_file = events_dir / f"{session}{EVENT_SUFFIXES.get(event_type)}"
    
    if not csv_file or not csv_file.exists():
        print(f"No events found at {csv_file}")
        return
        
    df_events = pd.read_csv(csv_file)
    if event_type == 'grasp':
        timestamps = df_events['EventTime'].values
        labels = df_events['Target'].fillna('unknown').astype(str) + "_" + df_events['Hand'].fillna('unknown').astype(str)
        labels = labels.values
    elif event_type == 'steps':
        timestamps = df_events['StepTime'].values
        labels = df_events['StepType'].fillna('unknown').astype(str) + "_" + \
                 df_events['Hand'].fillna('unknown').astype(str) + "_" + \
                 df_events['Surface'].fillna('unknown').astype(str)
        labels = labels.values
        
    num_trials = len(timestamps)
    
    # Check for existing artifact annotations to allow resuming
    out_dir = PROCESSED_DATA_DIR / subject / session
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"bad_trials_{event_type}.csv"
    
    if out_file.exists():
        df_bad = pd.read_csv(out_file)
        is_artifact = df_bad['is_artifact'].values.astype(bool)
    else:
        is_artifact = np.zeros(num_trials, dtype=bool)

    # 2. Load 1000Hz LFP infrastructure (without reading into RAM)
    recording_lfp = load_lfp_recording(subject, session, "lfp_1000Hz")
    fs_lfp = recording_lfp.get_sampling_frequency()
    total_samples = recording_lfp.get_num_samples()
    
    samples_pre = int(EPOCH_T_PRE * fs_lfp)
    samples_post = int(EPOCH_T_POST * fs_lfp)
    time_vec = np.linspace(-EPOCH_T_PRE, EPOCH_T_POST, samples_pre + samples_post)
    
    # 3. Setup the static Figure and pre-allocate Line2D objects
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(hspace=0.3)
    
    lines = []
    for a in range(4):
        ax = axes[a]
        ax.set_title(f"Array {a+1} (Ch {a*32} - {a*32+31})", loc='left', fontsize=10)
        ax.set_yticks([]) # Hide Y-axis as it's offset-based
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        
        # Pre-allocate 32 empty lines for the current array
        array_lines = [ax.plot([], [], color='black', linewidth=0.6, alpha=0.8)[0] for _ in range(32)]
        lines.append(array_lines)
        
        # Set static limits
        ax.set_xlim(-EPOCH_T_PRE, EPOCH_T_POST)
        ax.set_ylim(-offset_uv, 33 * offset_uv)
        
    axes[-1].set_xlabel("Time [s]")
    
    # 4. GUI Widgets
    trial_slider = widgets.IntSlider(min=0, max=num_trials-1, step=1, value=0, description='Trial:')
    btn_prev = widgets.Button(description='◀ Prev', button_style='info')
    btn_next = widgets.Button(description='Next ▶', button_style='info')
    toggle_art = widgets.ToggleButton(description='MARK ARTIFACT', button_style='success', value=False)
    btn_save = widgets.Button(description='💾 Save to Disk', button_style='warning')
    lbl_info = widgets.Label(value="")
    
    # 5. Core Update Logic
    def update_view(change=None) -> None:
        idx = trial_slider.value
        t = timestamps[idx]
        label = labels[idx]
        start_idx = int(t * fs_lfp) - samples_pre
        end_idx = int(t * fs_lfp) + samples_post
        
        # Sync toggle button with the array state
        toggle_art.unobserve(on_toggle_change, names='value')
        toggle_art.value = bool(is_artifact[idx])
        update_toggle_style()
        toggle_art.observe(on_toggle_change, names='value')
        
        lbl_info.value = f" Time: {t:.3f} s | Status: {'ARTIFACT' if is_artifact[idx] else 'CLEAN'}"
        
        # Handle boundaries
        if start_idx < 0 or end_idx > total_samples:
            fig.suptitle(f"Trial {idx} out of recording bounds", color='red')
            fig.canvas.draw_idle()
            return
            
        fig.suptitle(f"{subject} / {session} | Event: {event_type} | Trial {idx}/{num_trials-1} | {label}", fontsize=14)
        
        # Dynamically extract only the current window
        data_window = recording_lfp.get_traces(start_frame=start_idx, end_frame=end_idx, return_scaled=False)
        
        for a in range(4):
            for ch in range(32):
                global_ch = a * 32 + ch
                trace = data_window[:, global_ch]
                
                # Zero-center the trace and add fixed spatial offset
                trace_centered = trace - np.mean(trace)
                trace_offset = trace_centered + (ch * offset_uv)
                
                lines[a][ch].set_data(time_vec, trace_offset)
                
        fig.canvas.draw_idle()

    # 6. Event Handlers
    def on_prev(b) -> None:
        if trial_slider.value > 0:
            trial_slider.value -= 1

    def on_next(b) -> None:
        if trial_slider.value < num_trials - 1:
            trial_slider.value += 1
            
    def update_toggle_style() -> None:
        if toggle_art.value:
            toggle_art.button_style = 'danger'
            toggle_art.icon = 'times'
        else:
            toggle_art.button_style = 'success'
            toggle_art.icon = 'check'

    def on_toggle_change(change) -> None:
        idx = trial_slider.value
        is_artifact[idx] = change.new
        update_toggle_style()
        lbl_info.value = f" Time: {timestamps[idx]:.3f} s | Status: {'ARTIFACT' if is_artifact[idx] else 'CLEAN'}"

    def on_save(b) -> None:
        df_out = pd.DataFrame({'timestamp': timestamps, 'is_artifact': is_artifact})
        df_out.to_csv(out_file, index=False)
        lbl_info.value = f" Saved successfully to {out_file.name}!"

    # Bindings
    btn_prev.on_click(on_prev)
    btn_next.on_click(on_next)
    btn_save.on_click(on_save)
    trial_slider.observe(update_view, names='value')
    toggle_art.observe(on_toggle_change, names='value')
    
    # Layout and Initialization
    controls = widgets.HBox([btn_prev, trial_slider, btn_next, toggle_art, btn_save])
    display(widgets.VBox([controls, lbl_info]))
    
    update_view()
    plt.show()