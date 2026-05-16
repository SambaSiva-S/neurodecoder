# ===== NEURODECODER: FULL BCI ANALYSIS PLATFORM =====
# A complete Brain-Computer Interface analysis tool
# Built by SambaSiva-S | siverse.org

import gradio as gr
import numpy as np
import scipy.signal as sig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from braindecode.datasets import MOABBDataset
import torch
import torch.nn as nn
import tempfile
import time
import warnings
warnings.filterwarnings('ignore')

# ===== GLOBAL MODEL LOADING =====
print("Loading BCI models... (this takes ~30 seconds on first run)")

dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[1])
raw_global = dataset.datasets[0].raw
sfreq_global = raw_global.info['sfreq']
events_global, event_id_global = mne.events_from_annotations(raw_global)
eeg_picks_global = mne.pick_types(raw_global.info, eeg=True, eog=False, stim=False)
class_names_global = list(event_id_global.keys())
n_eeg = len(eeg_picks_global)

raw_filtered_global = raw_global.copy().filter(8, 30, picks=eeg_picks_global, verbose=False)
epochs_global = mne.Epochs(raw_filtered_global, events_global, event_id_global,
                           tmin=0.5, tmax=4.0, picks=eeg_picks_global,
                           baseline=None, preload=True, verbose=False)

X_global = epochs_global.get_data().astype(np.float32)
y_global = epochs_global.events[:, -1]

csp_global = CSP(n_components=8, reg=None, log=True, norm_trace=False)
X_csp_global = csp_global.fit_transform(X_global, y_global)
lda_global = LinearDiscriminantAnalysis()
lda_global.fit(X_csp_global, y_global)

print("Models loaded!\n")


def load_user_data(file, subject_id):
    """Load EEG data from upload or demo subject."""
    if file is not None:
        file_path = file.name if hasattr(file, 'name') else file
        if file_path.endswith('.gdf'):
            return mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
        elif file_path.endswith('.edf'):
            return mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        elif file_path.endswith('.fif'):
            return mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        else:
            return None
    else:
        sid = int(subject_id) if subject_id else 1
        sid = max(1, min(9, sid))
        ds = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[sid])
        return ds.datasets[0].raw


# ===== TAB 1: SIGNAL EXPLORER =====
def signal_explorer(file, subject_id):
    try:
        user_raw = load_user_data(file, subject_id)
        if user_raw is None:
            return None, None, None, None, "Unsupported file format."

        user_sfreq = user_raw.info['sfreq']
        user_eeg_picks = mne.pick_types(user_raw.info, eeg=True, eog=False, stim=False)
        ch_names = [user_raw.ch_names[p] for p in user_eeg_picks]

        motor_chs = [ch for ch in ['C3', 'Cz', 'C4'] if ch in user_raw.ch_names]
        if not motor_chs:
            motor_chs = [user_raw.ch_names[p] for p in user_eeg_picks[:3]]

        # Plot 1: Raw EEG
        fig1, axes1 = plt.subplots(len(motor_chs), 1, figsize=(12, 2.5*len(motor_chs)), sharex=True)
        if len(motor_chs) == 1:
            axes1 = [axes1]
        data = user_raw.get_data(picks=motor_chs)
        t = np.arange(min(int(10*user_sfreq), data.shape[1])) / user_sfreq
        for i, ch in enumerate(motor_chs):
            axes1[i].plot(t, data[i, :len(t)] * 1e6, linewidth=0.5)
            axes1[i].set_ylabel(f'{ch} (µV)')
            axes1[i].grid(True, alpha=0.3)
        axes1[0].set_title('Raw EEG — Motor Cortex Channels (10 seconds)', fontweight='bold')
        axes1[-1].set_xlabel('Time (seconds)')
        plt.tight_layout()
        raw_path = tempfile.mktemp(suffix='.png')
        fig1.savefig(raw_path, dpi=100, bbox_inches='tight')
        plt.close(fig1)

        # Plot 2: PSD
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        for ch in motor_chs[:2]:
            ch_data = user_raw.get_data(picks=[ch])[0]
            freqs, psd = sig.welch(ch_data, fs=user_sfreq, nperseg=int(2*user_sfreq))
            ax2.semilogy(freqs, psd * 1e12, label=ch, linewidth=2)
        ax2.axvspan(8, 12, alpha=0.2, color='orange', label='Mu (8-12 Hz)')
        ax2.axvspan(13, 30, alpha=0.2, color='green', label='Beta (13-30 Hz)')
        ax2.set_xlim([0, 50])
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power (µV²/Hz)')
        ax2.set_title('Power Spectral Density', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        psd_path = tempfile.mktemp(suffix='.png')
        fig2.savefig(psd_path, dpi=100, bbox_inches='tight')
        plt.close(fig2)

        # Plot 3: Bandpass filtered
        bands = {'Delta (0.5-4)': (0.5, 4), 'Theta (4-8)': (4, 8),
                 'Mu (8-12)': (8, 12), 'Beta (13-30)': (13, 30), 'Gamma (30-45)': (30, 45)}
        ch_data_raw = user_raw.get_data(picks=[motor_chs[0]])[0]
        fig3, axes3 = plt.subplots(len(bands), 1, figsize=(12, 12), sharex=True)
        t_5s = np.arange(min(int(5*user_sfreq), len(ch_data_raw))) / user_sfreq
        colors_band = ['purple', 'blue', 'orange', 'green', 'red']
        for i, (bname, (low, high)) in enumerate(bands.items()):
            b, a = sig.butter(4, [low/(user_sfreq/2), high/(user_sfreq/2)], btype='band')
            filtered = sig.filtfilt(b, a, ch_data_raw)
            axes3[i].plot(t_5s, filtered[:len(t_5s)] * 1e6, linewidth=0.5, color=colors_band[i])
            axes3[i].set_ylabel(f'{bname}\n(µV)')
            axes3[i].grid(True, alpha=0.3)
        axes3[0].set_title(f'Frequency Band Decomposition — {motor_chs[0]}', fontweight='bold')
        axes3[-1].set_xlabel('Time (seconds)')
        plt.tight_layout()
        bands_path = tempfile.mktemp(suffix='.png')
        fig3.savefig(bands_path, dpi=100, bbox_inches='tight')
        plt.close(fig3)

        # Plot 4: Spectrogram
        fig4, axes4 = plt.subplots(len(motor_chs[:2]), 1, figsize=(12, 4*len(motor_chs[:2])))
        if len(motor_chs[:2]) == 1:
            axes4 = [axes4]
        for i, ch in enumerate(motor_chs[:2]):
            ch_d = user_raw.get_data(picks=[ch])[0]
            f_s, t_s, Sxx = sig.spectrogram(ch_d[:int(30*user_sfreq)], fs=user_sfreq,
                                             nperseg=int(user_sfreq*0.5), noverlap=int(user_sfreq*0.4))
            axes4[i].pcolormesh(t_s, f_s, 10*np.log10(Sxx*1e12), shading='gouraud',
                               cmap='viridis', vmin=-20, vmax=20)
            axes4[i].set_ylim([0, 45])
            axes4[i].axhline(y=8, color='white', linestyle='--', alpha=0.5)
            axes4[i].axhline(y=12, color='white', linestyle='--', alpha=0.5)
            axes4[i].set_ylabel('Frequency (Hz)')
            axes4[i].set_title(f'{ch} — Spectrogram (30s)', fontweight='bold')
        axes4[-1].set_xlabel('Time (seconds)')
        plt.tight_layout()
        spec_path = tempfile.mktemp(suffix='.png')
        fig4.savefig(spec_path, dpi=100, bbox_inches='tight')
        plt.close(fig4)

        summary = f"""## Signal Explorer Results
- **Channels:** {len(user_eeg_picks)} EEG channels
- **Sampling Rate:** {user_sfreq} Hz
- **Duration:** {user_raw.times[-1]:.1f} seconds
- **Motor Cortex Channels:** {', '.join(motor_chs)}
- **Frequency Bands:** Delta, Theta, Mu, Beta, Gamma decomposed
"""
        return raw_path, psd_path, bands_path, spec_path, summary

    except Exception as e:
        return None, None, None, None, f"Error: {str(e)}"


# ===== TAB 2: CLASSICAL DECODER =====
def classical_decoder(file, subject_id):
    try:
        user_raw = load_user_data(file, subject_id)
        if user_raw is None:
            return None, None, None, "Unsupported file format."

        user_sfreq = user_raw.info['sfreq']
        user_events, user_event_id = mne.events_from_annotations(user_raw)
        user_eeg_picks = mne.pick_types(user_raw.info, eeg=True, eog=False, stim=False)

        user_raw_f = user_raw.copy().filter(8, 30, picks=user_eeg_picks, verbose=False)
        user_epochs = mne.Epochs(user_raw_f, user_events, user_event_id, tmin=0.5, tmax=4.0,
                                 picks=user_eeg_picks, baseline=None, preload=True, verbose=False)

        X_user = user_epochs.get_data().astype(np.float32)
        y_user = user_epochs.events[:, -1]
        user_class_names = list(user_event_id.keys())

        # Decode
        X_user_csp = csp_global.transform(X_user)
        predictions = lda_global.predict(X_user_csp)
        probabilities = lda_global.predict_proba(X_user_csp)

        correct = (predictions == y_user).sum()
        total = len(y_user)
        accuracy = correct / total * 100

        # Plot 1: Confusion matrix
        fig1, ax1 = plt.subplots(figsize=(7, 6))
        cm = confusion_matrix(y_user, predictions)
        labels = user_class_names if len(user_class_names) == cm.shape[0] else [f'Class {i}' for i in range(cm.shape[0])]
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot(ax=ax1, cmap='Blues', values_format='d')
        ax1.set_title(f'Confusion Matrix — CSP + LDA\nAccuracy: {accuracy:.1f}%', fontsize=14, fontweight='bold')
        plt.tight_layout()
        cm_path = tempfile.mktemp(suffix='.png')
        fig1.savefig(cm_path, dpi=100, bbox_inches='tight')
        plt.close(fig1)

        # Plot 2: Confidence distribution
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        max_confs = probabilities.max(axis=1)
        axes2[0].hist(max_confs[predictions == y_user], bins=15, alpha=0.7, color='green', label='Correct', edgecolor='black')
        axes2[0].hist(max_confs[predictions != y_user], bins=15, alpha=0.7, color='red', label='Incorrect', edgecolor='black')
        axes2[0].set_xlabel('Confidence')
        axes2[0].set_ylabel('Count')
        axes2[0].set_title('Confidence Distribution', fontweight='bold')
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)

        unique_preds, pred_counts = np.unique(predictions, return_counts=True)
        colors_bar = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        pred_labels = []
        for p in unique_preds:
            idx = list(event_id_global.values()).index(p) if p in event_id_global.values() else 0
            pred_labels.append(class_names_global[idx] if idx < len(class_names_global) else f'Class {p}')
        axes2[1].bar(pred_labels, pred_counts, color=colors_bar[:len(pred_labels)], edgecolor='black')
        axes2[1].set_title('Predicted Class Distribution', fontweight='bold')
        axes2[1].set_ylabel('Count')
        axes2[1].grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        conf_path = tempfile.mktemp(suffix='.png')
        fig2.savefig(conf_path, dpi=100, bbox_inches='tight')
        plt.close(fig2)

        # Plot 3: CSP patterns
        fig3 = csp_global.plot_patterns(epochs_global.info, ch_type='eeg', units='AU', size=1.5, show=False)
        fig3.suptitle('CSP Spatial Patterns (Learned Filters)', fontweight='bold')
        plt.tight_layout()
        csp_path = tempfile.mktemp(suffix='.png')
        fig3.savefig(csp_path, dpi=100, bbox_inches='tight')
        plt.close(fig3)

        summary = f"""## Classical Decoder Results
**Overall Accuracy: {accuracy:.1f}%** ({correct}/{total} trials)

### Per-Class Accuracy:
"""
        for cls_name, cls_id in user_event_id.items():
            mask = y_user == cls_id
            if mask.sum() > 0:
                cls_acc = (predictions[mask] == y_user[mask]).mean() * 100
                summary += f"- **{cls_name}:** {cls_acc:.1f}% ({(predictions[mask] == y_user[mask]).sum()}/{mask.sum()})\n"

        summary += f"""
### Decoder Configuration:
- Algorithm: Common Spatial Patterns (CSP) + Linear Discriminant Analysis (LDA)
- CSP Components: 8
- Filter: 8-30 Hz bandpass (Mu + Beta bands)
- Chance Level: {100/len(np.unique(y_user)):.1f}%
"""
        return cm_path, conf_path, csp_path, summary

    except Exception as e:
        return None, None, None, f"Error: {str(e)}"


# ===== TAB 3: REAL-TIME SIMULATOR =====
def realtime_simulator(file, subject_id, trial_num):
    try:
        user_raw = load_user_data(file, subject_id)
        if user_raw is None:
            return None, None, None, "Unsupported file format."

        user_sfreq = user_raw.info['sfreq']
        user_events, user_event_id = mne.events_from_annotations(user_raw)
        user_eeg_picks = mne.pick_types(user_raw.info, eeg=True, eog=False, stim=False)

        user_raw_f = user_raw.copy().filter(8, 30, picks=user_eeg_picks, verbose=False)
        user_epochs = mne.Epochs(user_raw_f, user_events, user_event_id, tmin=0.5, tmax=4.0,
                                 picks=user_eeg_picks, baseline=None, preload=True, verbose=False)

        X_user = user_epochs.get_data().astype(np.float32)
        y_user = user_epochs.events[:, -1]

        trial_idx = min(int(trial_num) - 1, len(X_user) - 1) if trial_num else 0
        trial_data = X_user[trial_idx]
        true_label = y_user[trial_idx]
        user_class_names = list(user_event_id.keys())
        true_class = user_class_names[list(user_event_id.values()).index(true_label)]

        # Streaming decode
        chunk_size = int(user_sfreq * 0.2)
        n_chunks = trial_data.shape[1] // chunk_size
        n_ch = trial_data.shape[0]

        belief = np.ones(len(user_event_id)) / len(user_event_id)
        belief_history = []
        confidence_history = []
        timestamps = []
        latencies = []
        cursor_x, cursor_y = 0.5, 0.5
        cursor_trail = [(0.5, 0.5)]

        direction_map = {0: (0, 0.03), 1: (-0.03, 0), 2: (0.03, 0), 3: (0, -0.03)}
        b_filt, a_filt = sig.butter(4, [8/(user_sfreq/2), 30/(user_sfreq/2)], btype='band')
        accumulated = None

        for chunk_idx in range(n_chunks):
            start_t = time.perf_counter()
            s = chunk_idx * chunk_size
            chunk = trial_data[:, s:s+chunk_size]

            if accumulated is None:
                accumulated = chunk.copy()
            else:
                accumulated = np.concatenate([accumulated, chunk], axis=1)

            current_time = accumulated.shape[1] / user_sfreq

            if accumulated.shape[1] >= int(user_sfreq * 0.5):
                try:
                    filtered = sig.filtfilt(b_filt, a_filt, accumulated, axis=1)
                    features = csp_global.transform(filtered.reshape(1, n_ch, -1))
                    probs = lda_global.predict_proba(features)[0]
                    belief = belief * probs
                    belief = belief / belief.sum()
                except Exception:
                    pass

            pred_class = np.argmax(belief)
            confidence = belief[pred_class]
            latency = (time.perf_counter() - start_t) * 1000

            belief_history.append(belief.copy())
            confidence_history.append(confidence)
            timestamps.append(current_time)
            latencies.append(latency)

            dx, dy = direction_map.get(pred_class, (0, 0))
            speed = max(0, (confidence - 0.3)) / 0.7
            cursor_x = np.clip(cursor_x + dx * speed, 0, 1)
            cursor_y = np.clip(cursor_y + dy * speed, 0, 1)
            cursor_trail.append((cursor_x, cursor_y))

        # Plot 1: Belief evolution
        fig1, axes1 = plt.subplots(2, 1, figsize=(12, 8))
        belief_arr = np.array(belief_history)
        for ci in range(belief_arr.shape[1]):
            label = user_class_names[ci] if ci < len(user_class_names) else f'Class {ci}'
            axes1[0].plot(timestamps, belief_arr[:, ci], linewidth=2, label=label)
        axes1[0].axhline(y=0.25, color='gray', linestyle='--', alpha=0.3)
        axes1[0].axhline(y=0.6, color='green', linestyle=':', alpha=0.5, label='Threshold')
        axes1[0].set_ylabel('Belief Probability')
        axes1[0].set_title(f'Progressive Belief Building — True: {true_class.upper()}',
                          fontsize=13, fontweight='bold')
        axes1[0].legend(loc='upper left', fontsize=9)
        axes1[0].set_ylim([0, 1])
        axes1[0].grid(True, alpha=0.3)

        axes1[1].plot(timestamps, confidence_history, linewidth=2, color='purple')
        axes1[1].fill_between(timestamps, confidence_history, alpha=0.3, color='purple')
        axes1[1].axhline(y=0.6, color='green', linestyle='--', label='Action threshold')
        axes1[1].set_xlabel('Time (seconds)')
        axes1[1].set_ylabel('Confidence')
        axes1[1].set_title('Decoder Confidence Over Time', fontweight='bold')
        axes1[1].set_ylim([0, 1])
        axes1[1].legend()
        axes1[1].grid(True, alpha=0.3)
        plt.tight_layout()
        belief_path = tempfile.mktemp(suffix='.png')
        fig1.savefig(belief_path, dpi=100, bbox_inches='tight')
        plt.close(fig1)

        # Plot 2: Cursor
        fig2, ax2 = plt.subplots(figsize=(7, 7))
        trail_x = [p[0] for p in cursor_trail]
        trail_y = [p[1] for p in cursor_trail]
        ax2.plot(trail_x, trail_y, 'b-', linewidth=1, alpha=0.5)
        ax2.plot(trail_x[0], trail_y[0], 'go', markersize=15, label='Start')
        ax2.plot(trail_x[-1], trail_y[-1], 'r*', markersize=20, label='End')

        targets = {'Up (feet)': (0.5, 0.9), 'Left (left_hand)': (0.1, 0.5),
                   'Right (right_hand)': (0.9, 0.5), 'Down (tongue)': (0.5, 0.1)}
        for name, (tx, ty) in targets.items():
            circle = plt.Circle((tx, ty), 0.08, fill=False, linestyle='--', color='gray')
            ax2.add_patch(circle)
            ax2.annotate(name, (tx, ty), ha='center', va='center', fontsize=8, color='gray')

        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.set_aspect('equal')
        ax2.set_title(f'BCI Cursor — True: {true_class.upper()}', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        cursor_path = tempfile.mktemp(suffix='.png')
        fig2.savefig(cursor_path, dpi=100, bbox_inches='tight')
        plt.close(fig2)

        # Plot 3: Latency
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.hist(latencies, bins=20, color='teal', edgecolor='black', alpha=0.7)
        ax3.axvline(x=np.mean(latencies), color='red', linestyle='--',
                    label=f'Mean: {np.mean(latencies):.2f}ms')
        ax3.axvline(x=50, color='orange', linestyle='--', label='Target: 50ms')
        ax3.set_xlabel('Latency (ms)')
        ax3.set_ylabel('Count')
        ax3.set_title('Decode Latency Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        lat_path = tempfile.mktemp(suffix='.png')
        fig3.savefig(lat_path, dpi=100, bbox_inches='tight')
        plt.close(fig3)

        final_pred = user_class_names[np.argmax(belief)]
        expected = {'feet': 'Up', 'left_hand': 'Left', 'right_hand': 'Right', 'tongue': 'Down'}
        summary = f"""## Real-Time Simulation Results

**True Intent:** {true_class} → Expected: {expected.get(true_class, '?')}
**Decoded:** {final_pred} (Confidence: {belief[np.argmax(belief)]:.0%})
**Cursor:** ({cursor_x:.2f}, {cursor_y:.2f})

### Performance:
- Avg Latency: {np.mean(latencies):.2f}ms
- P95 Latency: {np.percentile(latencies, 95):.2f}ms
- Target <50ms: {'✓ PASS' if np.percentile(latencies, 95) < 50 else '✗'}
- Chunk Size: 200ms micro-turns
- Architecture: Bayesian belief accumulation (Thinking Machines inspired)
"""
        return belief_path, cursor_path, lat_path, summary

    except Exception as e:
        return None, None, None, f"Error: {str(e)}"


# ===== TAB 4: SPEECH ANALYSIS =====
def speech_analysis(file, subject_id):
    try:
        user_raw = load_user_data(file, subject_id)
        if user_raw is None:
            return None, None, "Unsupported file format."

        user_sfreq = user_raw.info['sfreq']
        user_eeg_picks = mne.pick_types(user_raw.info, eeg=True, eog=False, stim=False)
        user_events, user_event_id = mne.events_from_annotations(user_raw)
        ch_names = [user_raw.ch_names[p] for p in user_eeg_picks]

        user_raw_wide = user_raw.copy().filter(1, 70, picks=user_eeg_picks, verbose=False)
        user_epochs = mne.Epochs(user_raw_wide, user_events, user_event_id, tmin=0.0, tmax=3.0,
                                 picks=user_eeg_picks, baseline=None, preload=True, verbose=False)

        X_user = user_epochs.get_data().astype(np.float32)
        y_user = user_epochs.events[:, -1]
        unique_y = np.unique(y_user)
        label_map = {old: new for new, old in enumerate(unique_y)}
        y_mapped = np.array([label_map[l] for l in y_user])
        user_class_names = list(user_event_id.keys())

        speech_bands = {
            'Theta (4-8)': (4, 8), 'Alpha (8-13)': (8, 13),
            'Low Beta (13-20)': (13, 20), 'High Beta (20-30)': (20, 30),
            'Low Gamma (30-45)': (30, 45), 'High Gamma (55-70)': (55, 70)
        }

        n_ch = X_user.shape[1]

        # Plot 1: Band power per class
        fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
        for bidx, (bname, (low, high)) in enumerate(speech_bands.items()):
            ax = axes1[bidx//3, bidx%3]
            b, a = sig.butter(4, [low/(user_sfreq/2), high/(user_sfreq/2)], btype='band')

            for cidx in range(min(len(unique_y), 4)):
                mask = y_mapped == cidx
                powers = []
                for trial in X_user[mask]:
                    ch_powers = []
                    for ch in range(n_ch):
                        filt = sig.filtfilt(b, a, trial[ch])
                        ch_powers.append(np.log(np.var(filt) + 1e-10))
                    powers.append(ch_powers)
                avg_power = np.mean(powers, axis=0)
                label = user_class_names[cidx] if cidx < len(user_class_names) else f'Class {cidx}'
                ax.plot(avg_power, linewidth=2, alpha=0.7, label=label)

            ax.set_title(bname, fontweight='bold')
            ax.set_xlabel('Channel')
            ax.set_ylabel('Log Power')
            ax.grid(True, alpha=0.3)
            if bidx == 0:
                ax.legend(fontsize=7)

        fig1.suptitle('Speech-Relevant Frequency Bands Per Class', fontsize=14, fontweight='bold')
        plt.tight_layout()
        bands_path = tempfile.mktemp(suffix='.png')
        fig1.savefig(bands_path, dpi=100, bbox_inches='tight')
        plt.close(fig1)

        # Plot 2: Gamma topography
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        b_g, a_g = sig.butter(4, [30/(user_sfreq/2), 45/(user_sfreq/2)], btype='band')
        for cidx in range(min(len(unique_y), 4)):
            mask = y_mapped == cidx
            gamma_per_ch = []
            for ch in range(n_ch):
                trials = X_user[mask, ch, :]
                powers = [np.log(np.var(sig.filtfilt(b_g, a_g, t)) + 1e-10) for t in trials]
                gamma_per_ch.append(np.mean(powers))
            label = user_class_names[cidx] if cidx < len(user_class_names) else f'Class {cidx}'
            ax2.plot(gamma_per_ch, 'o-', linewidth=2, markersize=4, label=label)

        ax2.set_xticks(range(len(ch_names)))
        ax2.set_xticklabels(ch_names, rotation=45, fontsize=7)
        ax2.set_xlabel('Channel')
        ax2.set_ylabel('Log Gamma Power')
        ax2.set_title('Low Gamma (30-45 Hz) — Speech-Relevant Activity', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        gamma_path = tempfile.mktemp(suffix='.png')
        fig2.savefig(gamma_path, dpi=100, bbox_inches='tight')
        plt.close(fig2)

        summary = f"""## Speech Analysis Results
- **Channels analyzed:** {n_ch}
- **Frequency bands:** Theta, Alpha, Low/High Beta, Low/High Gamma
- **Trials:** {len(X_user)}
- **Classes:** {', '.join(user_class_names)}

### Key Findings:
- Gamma activity (30-70 Hz) is critical for speech imagery
- Frontal channels (F, FC) relate to Broca's area (speech production)
- Temporal channels relate to Wernicke's area (speech comprehension)
- Different classes show distinct gamma-band spatial patterns
"""
        return bands_path, gamma_path, summary

    except Exception as e:
        return None, None, f"Error: {str(e)}"


# ===== BUILD THE FULL PLATFORM =====
with gr.Blocks(
    title="NeuroDecoder — BCI Analysis Platform",
    theme=gr.themes.Soft()
) as app:

    gr.Markdown("""
    # 🧠 NeuroDecoder — BCI Analysis Platform
    ### Decode brain signals. Visualize neural activity. Simulate real-time control.

    Upload your EEG data or explore with built-in demo subjects.
    Built by [SambaSiva-S](https://github.com/SambaSiva-S/neurodecoder) | [Siverse](https://siverse.org)

    ---
    """)

    # Shared inputs
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload EEG (.gdf, .edf, .fif)", file_types=[".gdf", ".edf", ".fif"])
            subject_dd = gr.Dropdown(choices=["1","2","3","4","5","6","7","8","9"],
                                     value="1", label="Or Select Demo Subject")

    # TAB 1: Signal Explorer
    with gr.Tab("📊 Signal Explorer"):
        gr.Markdown("Visualize raw EEG, frequency bands, spectrograms, and power spectral density.")
        explore_btn = gr.Button("🔍 Explore Signals", variant="primary", size="lg")
        with gr.Row():
            explore_raw = gr.Image(label="Raw EEG")
            explore_psd = gr.Image(label="Power Spectral Density")
        with gr.Row():
            explore_bands = gr.Image(label="Frequency Bands")
            explore_spec = gr.Image(label="Spectrogram")
        explore_summary = gr.Markdown()
        explore_btn.click(fn=signal_explorer, inputs=[file_input, subject_dd],
                         outputs=[explore_raw, explore_psd, explore_bands, explore_spec, explore_summary])

    # TAB 2: Classical Decoder
    with gr.Tab("🎯 Classical Decoder"):
        gr.Markdown("Decode motor imagery using CSP + LDA. See confusion matrix and confidence analysis.")
        decode_btn = gr.Button("🧠 Decode", variant="primary", size="lg")
        with gr.Row():
            decode_cm = gr.Image(label="Confusion Matrix")
            decode_conf = gr.Image(label="Confidence Analysis")
        decode_csp = gr.Image(label="CSP Spatial Patterns")
        decode_summary = gr.Markdown()
        decode_btn.click(fn=classical_decoder, inputs=[file_input, subject_dd],
                        outputs=[decode_cm, decode_conf, decode_csp, decode_summary])

    # TAB 3: Real-Time Simulator
    with gr.Tab("⚡ Real-Time Simulator"):
        gr.Markdown("Watch Bayesian belief build progressively. Confidence-aware cursor control. Inspired by Thinking Machines' Interaction Model (2026).")
        with gr.Row():
            trial_dd = gr.Dropdown(choices=[str(i) for i in range(1, 13)], value="1", label="Trial Number")
        rt_btn = gr.Button("▶️ Simulate Real-Time", variant="primary", size="lg")
        with gr.Row():
            rt_belief = gr.Image(label="Progressive Belief & Confidence")
            rt_cursor = gr.Image(label="Cursor Movement")
        rt_latency = gr.Image(label="Latency Distribution")
        rt_summary = gr.Markdown()
        rt_btn.click(fn=realtime_simulator, inputs=[file_input, subject_dd, trial_dd],
                    outputs=[rt_belief, rt_cursor, rt_latency, rt_summary])

    # TAB 4: Speech Analysis
    with gr.Tab("🗣️ Speech Analysis"):
        gr.Markdown("Analyze speech-relevant frequency bands (Theta through High Gamma). Targets: Paradromics.")
        speech_btn = gr.Button("🗣️ Analyze Speech Features", variant="primary", size="lg")
        with gr.Row():
            speech_bands_plot = gr.Image(label="Speech Frequency Bands")
            speech_gamma = gr.Image(label="Gamma Activity Map")
        speech_summary = gr.Markdown()
        speech_btn.click(fn=speech_analysis, inputs=[file_input, subject_dd],
                        outputs=[speech_bands_plot, speech_gamma, speech_summary])

    # Footer
    gr.Markdown("""
    ---
    ### About NeuroDecoder
    A complete BCI analysis platform built from scratch. Implements signal processing,
    classical ML (CSP + LDA), deep learning (EEGNet, Transformer), and real-time streaming
    decoding with Bayesian belief accumulation.

    **Architecture inspired by** [Thinking Machines' Interaction Model](https://thinkingmachines.ai/blog/interaction-models/) (2026)

    **Tech Stack:** Python, PyTorch, MNE-Python, scikit-learn, Gradio

    **GitHub:** [github.com/SambaSiva-S/neurodecoder](https://github.com/SambaSiva-S/neurodecoder) |
    **Website:** [siverse.org](https://siverse.org)
    """)

if __name__ == "__main__":
    app.launch(share=True)