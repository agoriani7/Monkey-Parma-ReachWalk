import numpy as np
import math
from utils.get_channel_geometry_mapping import get_channel_geometry_mapping
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Tuple, Optional
import ipywidgets as widgets
from IPython.display import display
from src.io import load_envelopes, load_epochs, load_cwt_epochs
from src.config import EPOCH_T_POST, EPOCH_T_PRE, FREQ_BANDS

def plot_interactive_envelopes(
    subject: str,
    session: str,
    fs: float = 100.0,
    time_window: Optional[Tuple[float, float]] = (0.0, 10.0)
) -> None:
    """
    Renders an interactive time-frequency heatmap of the envelopes.
    Uses a scrollable list for channel selection placed next to the plot.
    """
    envelopes = load_envelopes(subject, session)
    bands = list(envelopes.keys())
    num_channels = envelopes[bands[0]].shape[1]
    num_bands = len(bands)
    # Determine sample indices based on the requested time window
    if time_window is not None:
        start_sample = int(time_window[0] * fs)
        end_sample = int(time_window[1] * fs)
    else:
        start_sample = 0
        end_sample = envelopes[bands[0]].shape[0]
        
    # Add + 1 to end_sample to generate the rightmost boundary required by shading='flat'
    time_vector = np.arange(start_sample, end_sample) / fs
    
    # Create a scrollable list (Select widget) for channels
    channel_options = [(f"{i}", i) for i in range(num_channels)]
    channel_selector = widgets.Select(
        options=channel_options,
        value=0,
        description='Ch:',
        rows=20,  # Number of visible rows determines the height and scrollability
        layout=widgets.Layout(width='150px')
    )
    
    # Output widget to host the plot
    plot_output = widgets.Output()
    
    def update_plot(change) -> None:
        # Get the new channel index from the event or fallback to the current value
        channel_idx = change.new if change is not None else channel_selector.value
        
        with plot_output:
            plot_output.clear_output(wait=True)
            
            fig, axes = plt.subplots(nrows=num_bands, ncols=1, figsize=(10, num_bands), sharex=True)
            if num_bands == 1:
                axes = [axes]
            for ax, band in zip(axes, bands):
                # Extract temporal trace for the specific band and channel
                trace = envelopes[band][start_sample:end_sample, channel_idx]
                # Define Y-axis limits with a 10% margin
                ymin, ymax = np.min(trace), np.max(trace)
                margin = (ymax - ymin) * 0.1 if ymax != ymin else 1.0
                ymin, ymax = ymin - margin, ymax + margin
                # Plot the colormap background reflecting the amplitude
                ax.imshow(
                    trace[np.newaxis, :], 
                    aspect='auto', 
                    cmap='viridis', 
                    extent=[time_vector[0], time_vector[-1], ymin, ymax],
                    alpha=0.7,  # Add transparency to make the line clearly visible
                    origin='lower'
                )
                
                # Plot the line plot
                ax.plot(time_vector, trace, color='black', linewidth=1.2)
                ax.set_ylim(ymin, ymax)
                ax.set_ylabel(f"{band}\n[$\\mu$V]")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            axes[-1].set_xlabel("Time [s]")
            axes[-1].set_xlim(time_vector[0], time_vector[-1])
            fig.suptitle(f"{subject} / {session} - Channel {channel_idx}", fontsize=14, y=0.98)
            fig.tight_layout()
            plt.show()
    # Bind the selector to the update function
    channel_selector.observe(update_plot, names='value')
    update_plot(None)
    display(widgets.HBox([channel_selector, plot_output]))

def plot_interactive_epochs(subject: str, session: str, event_type: str) -> None:
    """
    Renders an interactive plot of trial-averaged epochs across all frequency bands.
    Uses a scrollable list for channel selection placed next to the plot.
    """
    # Expected shape for each band: (num_trials, num_samples, num_channels)
    epochs_dict = load_epochs(subject, session, event_type)
    labels = epochs_dict.pop("labels")
    bands = list(epochs_dict.keys())
    unique_labels = np.unique(labels)
    
    num_trials, num_samples, num_channels = epochs_dict[bands[0]].shape
    time_vector = np.linspace(-EPOCH_T_PRE, EPOCH_T_POST, num_samples)
    
    # Create a scrollable list (Select widget) for channels
    channel_options = [(f"{i}", i) for i in range(num_channels)]
    channel_selector = widgets.Select(
        options=channel_options,
        value=0,
        description='Ch:',
        rows=20,
        layout=widgets.Layout(width='150px')
    )
    visibility_state = {}
    # Output widget to host the plot
    plot_output = widgets.Output()
    
    def update_plot(change) -> None:
        # Get the new channel index from the event or fallback to the current value
        channel_idx = change.new if change is not None else channel_selector.value
        
        with plot_output:
            plot_output.clear_output(wait=True)
            
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(17, 8))
            axes = axes.flatten()                
            artists_dict = {}
            for i, ax in enumerate(axes):
                # Hide unused subplots if bands are less than 16
                if i >= len(bands):
                    ax.set_visible(False)
                    continue
                    
                band = bands[i]
                
                for col_idx, label in enumerate(unique_labels):
                    # Filter trials corresponding to the current label
                    trial_mask = (labels == label)
                    num_trials_label = np.sum(trial_mask)
                    
                    if num_trials_label == 0:
                        continue

                    legend_label = f"{label} (N={num_trials_label})"
                    if legend_label not in artists_dict:
                        artists_dict[legend_label] = []

                    # Initialize state if not present
                    if legend_label not in visibility_state:
                        visibility_state[legend_label] = True
                    is_visible = visibility_state[legend_label]

                    # Extract temporal traces for specific band, filtered trials, and specific channel
                    channel_data = epochs_dict[band][trial_mask, :, channel_idx]
                    
                    # Compute mean and standard error of the mean (SEM)
                    mean_trace = np.mean(channel_data, axis=0)
                    std_trace = np.std(channel_data, axis=0)
                    sem_trace = std_trace / np.sqrt(num_trials_label)
                    
                    # Plot mean line and shaded SEM area applying the persisted visibility
                    line, = ax.plot(time_vector, mean_trace, linewidth=1.5, label=legend_label, visible=is_visible)
                    fill = ax.fill_between(
                        time_vector, 
                        mean_trace - sem_trace, 
                        mean_trace + sem_trace, 
                        alpha=0.2, 
                        edgecolor='none',
                        visible=is_visible
                    )
                    
                    # Store artists to toggle visibility later
                    artists_dict[legend_label].extend([line, fill])
                
                # Mark event onset
                ax.axvline(x=0.0, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_title(f"{band}")
                ax.set_ylabel("[$\\mu$V]")
                ax.set_xlabel("Time [s]")
                ax.set_xlim(time_vector[0], time_vector[-1])
                
            # Extract handles and labels from the first subplot
            handles, leg_labels = axes[0].get_legend_handles_labels()
            
            # Repurpose the last axis of the 4x4 grid for the legend
            legend_ax = axes[-1]
            legend_ax.set_visible(True)
            legend_ax.axis('off')
            leg = legend_ax.legend(
                handles, 
                leg_labels, 
                loc='center', 
                frameon=False, 
                fontsize='small', 
                ncol=1
            )
            
            # Make legend lines clickable
            for leg_line, label_key in zip(leg.get_lines(), leg_labels):
                leg_line.set_picker(True)
                leg_line.set_pickradius(5)
                leg_line._associated_artists = artists_dict[label_key]
                leg_line._label_key = label_key  # Store key for state update
                leg_line.set_alpha(1.0 if visibility_state[label_key] else 0.2)

            def on_pick(event) -> None:
                leg_line = event.artist
                label_key = leg_line._label_key
                
                # Toggle and save visibility state
                visibility_state[label_key] = not visibility_state[label_key]
                is_visible = visibility_state[label_key]
                
                # Update legend alpha to indicate state
                leg_line.set_alpha(1.0 if is_visible else 0.2)
                
                # Toggle visibility of associated plot lines and fills
                for artist in leg_line._associated_artists:
                    artist.set_visible(is_visible)
                    
                fig.canvas.draw_idle()

            # Connect the click event to the figure
            fig.canvas.mpl_connect('pick_event', on_pick)

            # Connect the click event to the figure
            fig.canvas.mpl_connect('pick_event', on_pick)          
            fig.suptitle(f"{subject} / {session} | Event: {event_type} - Channel {channel_idx}", fontsize=14, y=1.02)
            fig.tight_layout()
            plt.show()

    # Bind the selector to the update function
    channel_selector.observe(update_plot, names='value')
    update_plot(None)
    display(widgets.HBox([channel_selector, plot_output]))

def plot_spatiotemporal_video(subject: str, session: str, event_type: str, label_filter: str) -> None:
    """
    Renders an interactive player showing the spatiotemporal activation of 128 channels 
    (arranged in a 2x2 grid of 8x4 matrices) across 8 frequency bands.
    Trials are filtered by checking if `label_filter` is a substring of the trial label.
    """
    epochs_dict = load_epochs(subject, session, event_type)
    labels = epochs_dict.pop("labels")
    bands = list(epochs_dict.keys())
    
    # Filter trials based on substring
    trial_mask = np.array([label_filter in str(lbl) for lbl in labels])
    num_trials = np.sum(trial_mask)
    
    if num_trials == 0:
        print(f"No trial found for '{label_filter}'.")
        return
        
    num_samples = epochs_dict[bands[0]].shape[1]
    
    # Compute mean across filtered trials for each band
    # Expected shape after mean: (num_samples, 128)
    mean_data = {band: np.mean(epochs_dict[band][trial_mask, :, :], axis=0) for band in bands}
    
    # Determine symmetric global min and max across time for divergent colormapping per band
    vlims = {}
    for band in bands:
        p_min = np.percentile(mean_data[band], 5)
        p_max = np.percentile(mean_data[band], 95)
        v_limit = max(abs(p_min), abs(p_max))
        vlims[band] = (-v_limit, v_limit)
    
    # Setup the figure
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
    axes = axes.flatten()
    
    im_artists = []
    
    for i, ax in enumerate(axes):
        if i >= len(bands):
            ax.set_visible(False)
            continue
            
        band = bands[i]
        
        # Initialize an empty 16x8 matrix for the 4 arrays
        im = ax.imshow(
            np.zeros((16, 8)), 
            aspect='auto', 
            cmap='RdBu_r', 
            vmin=vlims[band][0], 
            vmax=vlims[band][1], 
            origin='upper'
        )
        ax.set_title(f"{band}")
        ax.axis('off')
        
        # Draw separators for the 2x2 macro-grid
        ax.axhline(7.5, color='white', linewidth=3)
        ax.axvline(3.5, color='white', linewidth=3)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('[$\\mu$V]', fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        
        im_artists.append((band, im))
        
    fig.suptitle(f"{subject} / {session} | Event: {event_type} | Filter: '{label_filter}' (N={num_trials})", fontsize=14)
    fig.tight_layout()
    
    # Create interactive widgets
    play = widgets.Play(
        value=0,
        min=0,
        max=num_samples - 1,
        step=1,
        interval=50, # Update interval in milliseconds (20 fps)
        description="Press play"
    )
    time_slider = widgets.IntSlider(min=0, max=num_samples - 1, step=1, description='Sample:')
    widgets.jslink((play, 'value'), (time_slider, 'value'))
    
    # Pre-calculate the physical geometry mapping
    channel_mapping = get_channel_geometry_mapping()

    def update(change) -> None:
        t = change.new if change is not None else time_slider.value
        
        for band, im in im_artists:
            data_t = mean_data[band][t, :] # Shape: (128,)
            
            # Instantly map the 1D 128-channel array to the 16x8 physical grid
            grid = data_t[channel_mapping]
            
            im.set_data(grid)
            
        fig.canvas.draw_idle()

    # Bind and display
    time_slider.observe(update, names='value')
    update(None) # Trigger first frame
    
    display(widgets.HBox([play, time_slider]))
    plt.show()

def plot_average_cwt(subject: str, session: str, event_type: str) -> None:
    """
    Renders a static Time-Frequency Representation (Spectrogram) of the CWT data.
    Averages the Z-scored power across trials and clusters the 128 channels into 
    4 separate arrays (32 channels each), displayed in a 2x2 grid per label.
    """
    # Load CWT dictionary 
    cwt_dict = load_cwt_epochs(subject, session, event_type)
    cwt_tensor = cwt_dict['cwt_tensor']  # Shape: (n_trials, n_freqs, n_times, n_channels)
    labels = cwt_dict['labels']
    freqs = cwt_dict['freqs']
    
    unique_labels = np.unique(labels)
    num_trials_total, num_freqs, num_times, num_channels = cwt_tensor.shape
    
    time_vector = np.linspace(-EPOCH_T_PRE, EPOCH_T_POST, num_times)
    
    # Determine outer grid layout based on number of unique labels
    ncols_out = 2 if len(unique_labels) > 1 else 1
    nrows_out = math.ceil(len(unique_labels) / ncols_out)
    
    fig = plt.figure(figsize=(8 * ncols_out, 6 * nrows_out))
    # wspace and hspace control the padding between different label blocks
    outer_gs = gridspec.GridSpec(nrows_out, ncols_out, figure=fig, wspace=0.3, hspace=0.4)
    
    # Precompute mean power and global color limits across all conditions and arrays
    label_data_db = {}
    global_min, global_max = np.inf, -np.inf
    
    for label in unique_labels:
        trial_mask = (labels == label)
        num_trials_label = np.sum(trial_mask)
        
        if num_trials_label == 0:
            continue
            
        # Average Z-scored power across trials (axis=0) -> shape: (n_freqs, n_times, 128)
        mean_trials = np.mean(cwt_tensor[trial_mask, :, :, :], axis=0)
        
        # Split into 4 physical arrays of 32 channels each and average spatially (axis=2)
        arrays_mean = [
            np.mean(mean_trials[:, :, 0:32], axis=2),
            np.mean(mean_trials[:, :, 32:64], axis=2),
            np.mean(mean_trials[:, :, 64:96], axis=2),
            np.mean(mean_trials[:, :, 96:128], axis=2)
        ]
        
        label_data_db[label] = (arrays_mean, num_trials_label)
        
        # Calculate percentiles for robust colormap limits
        for arr in arrays_mean:
            global_min = min(global_min, np.percentile(arr, 5))
            global_max = max(global_max, np.percentile(arr, 95))
            
    v_limit = max(abs(global_min), abs(global_max))
    im = None
    
    for i, label in enumerate(unique_labels):
        if label not in label_data_db:
            continue
            
        arrays_mean, n_trials = label_data_db[label]
        
        row_out = i // ncols_out
        col_out = i % ncols_out
        
        # Create a 2x2 inner grid for the 4 arrays inside the current label cell
        inner_gs = outer_gs[row_out, col_out].subgridspec(2, 2, wspace=0.08, hspace=0.08)
        
        # Create a hidden axis to set a centralized title for the 2x2 block
        ax_hidden = fig.add_subplot(outer_gs[row_out, col_out], frameon=False)
        ax_hidden.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        ax_hidden.set_title(f"Label: {label} (N={n_trials})", pad=20, fontsize=12, fontweight='bold')
        
        for j in range(4):
            ax = fig.add_subplot(inner_gs[j // 2, j % 2])
            
            # Plot TFR using a divergent colormap
            im = ax.pcolormesh(
                time_vector, 
                freqs, 
                arrays_mean[j], 
                cmap='RdBu_r', 
                vmin=-v_limit, 
                vmax=v_limit, 
                shading='gouraud'
            )
            
            ax.axvline(x=0.0, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
            # ax.set_yscale('log')
            
            # Keep labels clean by showing them only on the outer edges of the 2x2 grid
            if j // 2 == 1:
                ax.set_xlabel("Time [s]")
            else:
                ax.tick_params(labelbottom=False)
                
            if j % 2 == 0:
                ax.set_ylabel("Frequency [Hz]")
            else:
                ax.tick_params(labelleft=False)
                
            # Add Array ID overlay
            ax.text(
                0.05, 0.95, f"Arr {j+1}", transform=ax.transAxes, 
                ha='left', va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
            )
    
    # Attach a shared colorbar to the figure
    if im is not None:
        cbar = fig.colorbar(im, ax=fig.axes, fraction=0.02, pad=0.04)
        cbar.set_label('Power\n[Z-score]', fontsize=11)
        
    fig.suptitle(f"{subject} / {session} | Event: {event_type} - Average per Array", fontsize=14, y=1.02)
    plt.show()

def plot_interactive_cwt(subject: str, session: str, event_type: str) -> None:
    """
    Renders an interactive Time-Frequency Representation (Spectrogram) of the CWT data.
    Uses a scrollable list for channel selection. Displays a grid comparing all unique labels.
    """
    # Load CWT dictionary 
    cwt_dict = load_cwt_epochs(subject, session, event_type)
    cwt_tensor = cwt_dict['cwt_tensor']  # Shape: (n_trials, n_freqs, n_times, n_channels)
    labels = cwt_dict['labels']
    freqs = cwt_dict['freqs']
    
    unique_labels = np.unique(labels)
    num_trials_total, num_freqs, num_times, num_channels = cwt_tensor.shape
    
    time_vector = np.linspace(-EPOCH_T_PRE, EPOCH_T_POST, num_times)
    
    # Create a scrollable list for channels
    channel_options = [(f"{i}", i) for i in range(num_channels)]
    channel_selector = widgets.Select(
        options=channel_options,
        value=0,
        description='Ch:',
        rows=20,
        layout=widgets.Layout(width='150px')
    )
    
    plot_output = widgets.Output()
    
    def update_plot(change) -> None:
        channel_idx = change.new if change is not None else channel_selector.value
        
        with plot_output:
            plot_output.clear_output(wait=True)
            
            # Determine grid layout based on number of unique labels
            ncols = 2 if len(unique_labels) > 1 else 1
            nrows = math.ceil(len(unique_labels) / ncols)
            
            fig, axes = plt.subplots(
                nrows=nrows, 
                ncols=ncols, 
                figsize=(6 * ncols, 3.5 * nrows), 
                sharex=True, 
                sharey=True
            )
            
            # Flatten axes for unified 1D iteration
            if isinstance(axes, np.ndarray):
                axes_flat = axes.flatten()
            else:
                axes_flat = [axes]
                
            # Precompute mean power in dB and global color limits for the current channel
            label_data_db = {}
            global_min, global_max = np.inf, -np.inf
            
            for label in unique_labels:
                trial_mask = (labels == label)
                num_trials_label = np.sum(trial_mask)
                
                if num_trials_label == 0:
                    continue
                    
                # Extract and average Z-scored power across trials
                mean_power_z = np.mean(cwt_tensor[trial_mask, :, :, channel_idx], axis=0)
                
                label_data_db[label] = (mean_power_z, num_trials_label)
                
                global_min = min(global_min, np.percentile(mean_power_z, 5))
                global_max = max(global_max, np.percentile(mean_power_z, 95))
            
            im = None
            for i, ax in enumerate(axes_flat):
                if i >= len(unique_labels):
                    ax.set_visible(False)
                    continue
                    
                label = unique_labels[i]
                if label not in label_data_db:
                    continue
                    
                mean_power_z, n_trials = label_data_db[label]
                
                # Make color limits symmetric around 0 for Z-scores
                v_limit = max(abs(global_min), abs(global_max))
                
                # Plot TFR using a divergent colormap
                im = ax.pcolormesh(
                    time_vector, 
                    freqs, 
                    mean_power_z, 
                    cmap='RdBu_r', 
                    vmin=-v_limit, 
                    vmax=v_limit, 
                    shading='gouraud'
                )
                
                # Visual markers and axis settings
                ax.axvline(x=0.0, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
                ax.set_yscale('log')
                ax.set_title(f"Label: {label} (N={n_trials})", fontsize=11)
                
                if i % ncols == 0:
                    ax.set_ylabel("Frequency [Hz]")
                if i >= len(unique_labels) - ncols:
                    ax.set_xlabel("Time [s]")
            
            # Attach a shared colorbar to the figure
            if im is not None:
                cbar = fig.colorbar(im, ax=axes_flat.tolist(), fraction=0.02, pad=0.04)
                cbar.set_label('Power\n[Z-score]', fontsize=11)
                
            fig.suptitle(f"{subject} / {session} | Event: {event_type} - Channel {channel_idx}", fontsize=14, y=1.02)
            plt.show()

    # Bind the selector to the update function
    channel_selector.observe(update_plot, names='value')
    update_plot(None)
    
    display(widgets.HBox([channel_selector, plot_output]))

def plot_spatiotemporal_cwt_video(subject: str, session: str, event_type: str, label_filter: str) -> None:
    """
    Renders an interactive player showing the spatiotemporal activation of 128 channels 
    (arranged in a 2x2 grid of 8x4 matrices). 
    Computes the mean CWT power across defined frequency bands and filtered trials.
    """
    cwt_dict = load_cwt_epochs(subject, session, event_type)
    cwt_tensor = cwt_dict['cwt_tensor']  # Shape: (n_trials, n_freqs, n_times, n_channels)
    labels = cwt_dict['labels']
    freqs = cwt_dict['freqs']
    
    # Filter trials based on substring
    trial_mask = np.array([label_filter in str(lbl) for lbl in labels])
    num_trials = np.sum(trial_mask)
    
    if num_trials == 0:
        print(f"No trial found for '{label_filter}'.")
        return
        
    # Compute mean power across filtered trials -> Shape: (n_freqs, n_times, n_channels)
    mean_trial_cwt = np.mean(cwt_tensor[trial_mask, :, :, :], axis=0)
    
    # Aggregate power across frequencies for each band defined in config
    mean_data = {}
    valid_bands = []
    
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        # Find frequency bins that fall within the current band limits
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        if np.any(freq_mask):
            # Average power within the specific frequency band
            mean_data[band_name] = np.mean(mean_trial_cwt[freq_mask, :, :], axis=0)
            valid_bands.append(band_name)
            
    if not valid_bands:
        print("No frequency bands matched the computed CWT frequencies.")
        return

    # Restrict to a maximum of 8 bands to fit the 2x4 grid layout
    valid_bands = valid_bands[:8]
    num_samples = cwt_tensor.shape[2]
    
    # Determine symmetric global min and max across time for divergent colormapping per band
    vlims = {}
    for band in valid_bands:
        p_min = np.percentile(mean_data[band], 5)
        p_max = np.percentile(mean_data[band], 95)
        v_limit = max(abs(p_min), abs(p_max))
        vlims[band] = (-v_limit, v_limit)
    
    # Setup the figure
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 6))
    axes = axes.flatten()
    
    im_artists = []
    
    for i, ax in enumerate(axes):
        if i >= len(valid_bands):
            ax.set_visible(False)
            continue
            
        band = valid_bands[i]
        
        # Initialize an empty 16x8 matrix for the 4 arrays
        im = ax.imshow(
            np.zeros((16, 8)), 
            aspect='auto', 
            cmap='RdBu_r', 
            vmin=vlims[band][0], 
            vmax=vlims[band][1], 
            origin='upper'
        )
        ax.set_title(f"{band}")
        ax.axis('off')
        
        # Draw separators for the 2x2 macro-grid
        ax.axhline(7.5, color='white', linewidth=3)
        ax.axvline(3.5, color='white', linewidth=3)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Power\n[Z-score]', fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        
        im_artists.append((band, im))
        
    fig.suptitle(f"{subject} / {session} | Event: {event_type} | Filter: '{label_filter}' (N={num_trials})", fontsize=14)
    fig.tight_layout()
    
    # Create interactive widgets
    play = widgets.Play(
        value=0,
        min=0,
        max=num_samples - 1,
        step=1,
        interval=50, # Update interval in milliseconds (20 fps)
        description="Press play"
    )
    time_slider = widgets.IntSlider(min=0, max=num_samples - 1, step=1, description='Sample:')
    widgets.jslink((play, 'value'), (time_slider, 'value'))

    # Pre-calculate the physical geometry mapping
    channel_mapping = get_channel_geometry_mapping()
    
    def update(change) -> None:
        # Pre-calculate the physical geometry mapping
        channel_mapping = get_channel_geometry_mapping()
        t = change.new if change is not None else time_slider.value
        
        for band, im in im_artists:
            data_t = mean_data[band][t, :] # Shape: (128,)
            
            # Instantly map the 1D 128-channel array to the 16x8 physical grid
            grid = data_t[channel_mapping]
            
            im.set_data(grid)
            
        fig.canvas.draw_idle()

    # Bind and display
    time_slider.observe(update, names='value')
    update(None) # Trigger first frame
    
    display(widgets.HBox([play, time_slider]))
    plt.show()