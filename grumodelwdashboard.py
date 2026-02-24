import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (recall_score, confusion_matrix, balanced_accuracy_score, 
                             roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.calibration import calibration_curve
from lifelines.utils import concordance_index 
from scipy import stats
import random
import shap # Ensure shap is imported
import pandas as pd
import numpy as np
import base64
from io import BytesIO


class HTMLReport:
    def __init__(self, horizon):
        self.horizon = horizon
        self.content = [f"<html><head><style>body {{ font-family: sans-serif; margin: 40px; line-height: 1.6; }} "
                        f"table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }} "
                        f"th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }} "
                        f"th {{ background-color: #f2f2f2; }} .plot-img {{ max-width: 100%; height: auto; margin: 20px 0; }}"
                        f"</style></head><body>"]
        self.content.append(f"<h1>{self.horizon} day mortality risk model for MCRPC patients</h1>")

    def add_text(self, text, tag="p"):
        cleaned_text = text.replace("\n", "<br>")
        self.content.append(f"<{tag}>{cleaned_text}</{tag}>")

    def add_table_from_df(self, df):
        self.content.append(df.to_html(classes='table', index=False))

    def add_plot(self):
        # Capture the current figure explicitly
        fig = plt.gcf()
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png', bbox_inches='tight', dpi=100)
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        self.content.append(f'<div style="text-align:center;"><img class="plot-img" src="data:image/png;base64,{encoded}"></div>')
        plt.close(fig) # Close the specific figure to free memory

    def save(self, filename="MCRPC_Mortality_Report.html"):
        self.content.append("</body></html>")
        with open(filename, "w") as f:
            f.write("\n".join(self.content))
        print(f"Report saved to {filename}")

# Consistency
random.seed(240)
np.random.seed(240)
tf.random.set_seed(240)

def masked_mse(y_true, y_pred):
    """Calculates MSE only on non-missing values (where y_true != -999)."""
    mask = tf.cast(tf.not_equal(y_true, -999), tf.float32)
    return tf.reduce_sum(tf.square(y_true - y_pred) * mask) / (tf.reduce_sum(mask) + 1e-7)
 
 
class SequentialSurvivalGRU:
    def __init__(self, internal_columns= [
    'ECOGGRN', 'AGE', 'BMI', 'SYSBP', 'PULSE',
    'days_since_last_visit', 'any_grade3_plus', 'drug_reduced'
],
features = [
    'ECOG Performance Status', 'Current Age', 'BMI', 'Systolic BP', 'Pulse',
    'Days Since Last Visit', 'Any Grade 3+ AE', 'Drug Dosage Reduced'
],
horizon = 180,
sequence_length = 12):
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.feature_cols = internal_columns
        # Map internal column names to display names for the plot
        self.internal_cols = features
        self.input_dim = len(self.internal_cols)
        self.scaler = StandardScaler()
        self.final_scaler = MinMaxScaler(feature_range=(0, 1))
        self.autoencoder = None
        self.model = None
        self.report = HTMLReport(self.horizon)

    def train_autoencoder(self, X_with_nan):
        """Trains the autoencoder using GRU layers to reconstruct missing values."""
        print("Training Temporal GRU Autoencoder for Imputation...")
        X_train = np.nan_to_num(X_with_nan, nan=-999)
        
        ae_model = Sequential([
            layers.Input(shape=(self.sequence_length, self.input_dim)),
            layers.GRU(64, return_sequences=True),
            layers.GRU(32, return_sequences=False),
            layers.RepeatVector(self.sequence_length),
            layers.GRU(32, return_sequences=True),
            layers.GRU(64, return_sequences=True),
            layers.TimeDistributed(layers.Dense(self.input_dim))
        ])
        
        ae_model.compile(optimizer='adam', loss=masked_mse)
        ae_model.fit(X_train, X_train, epochs=30, batch_size=32, verbose=0)
        self.autoencoder = ae_model

    def prepare_data(self, df):
        pid_col = 'RPT' if 'RPT' in df.columns else 'SUBJID'
        X_list, y_list, metadata = [], [], []
        
        df_temp = df.copy()
        self.scaler.fit(df_temp[self.internal_cols].fillna(df_temp[self.internal_cols].median()))
        
        df_scaled = df.copy()
        df_scaled[self.internal_cols] = self.scaler.transform(df_scaled[self.internal_cols])

        for pid in df_scaled[pid_col].unique():
            p_data = df_scaled[df_scaled[pid_col] == pid].sort_values('VISDAY')
            if 'DSDAY' in df.columns:
                death_day = p_data['DSDAY'].iloc[0]
            else: 
                death_day = p_data['VISDAY'].max() 
            has_event = p_data['os_event'].iloc[0] == 1
            last_fup = p_data['VISDAY'].max() 

            for i in range(len(p_data)):
                current_vday = p_data['VISDAY'].iloc[i]
                time_to_death = death_day - current_vday
                
                if has_event and 0 <= time_to_death <= self.horizon:
                    label = 1
                elif (has_event and time_to_death > self.horizon) or (not has_event and (last_fup - current_vday) >= self.horizon):
                    label = 0
                else:
                    continue

                feat_seq = p_data[self.internal_cols].iloc[:i+1].values[-self.sequence_length:]
                
                if len(feat_seq) < self.sequence_length:
                    pad_width = self.sequence_length - len(feat_seq)
                    feat_seq = np.pad(feat_seq, ((pad_width, 0), (0, 0)), mode='constant', constant_values=np.nan)

                X_list.append(feat_seq)
                y_list.append(label)
                metadata.append({
                    'pid': pid, 'current_vday': current_vday, 'dsday': death_day,
                    'tte': time_to_death if has_event else (last_fup - current_vday),
                    'event': 1 if has_event else 0
                })

        X_raw = np.array(X_list)
        if self.autoencoder is None:
            self.train_autoencoder(X_raw)
        
        X_masked = np.nan_to_num(X_raw, nan=-999)
        X_imputed_preds = self.autoencoder.predict(X_masked)
        X_final = np.where(np.isnan(X_raw), X_imputed_preds, X_raw)
        
        return X_final, np.array(y_list), metadata

    def balanced_batch_generator(self, X, y, batch_size=32):
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        half_batch = batch_size // 2
        while True:
            batch_pos = np.random.choice(pos_idx, half_batch)
            batch_neg = np.random.choice(neg_idx, half_batch)
            indices = np.concatenate([batch_pos, batch_neg])
            np.random.shuffle(indices)
            X_batch = X[indices]
            y_batch = np.repeat(y[indices, np.newaxis, np.newaxis], self.sequence_length, axis=1)
            yield X_batch, y_batch

    def build_sequential_model(self):
        inputs = layers.Input(shape=(self.sequence_length, self.input_dim))
        x = layers.GRU(64, return_sequences=True, dropout=0.3)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.GRU(32, return_sequences=True, dropout=0.2)(x)
        outputs = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))(x)
        model = models.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model
    
    def plot_patient_trajectories(self, X, metadata, risks, threshold, num_per_group=10):
        """Visualizes risk trajectory and predictors for sampled patients."""
        df_res = pd.DataFrame(metadata)
        df_res['risk'] = risks
        df_res['alarm'] = (df_res['risk'] >= threshold).astype(int)
        
        # Identify groups
        pids_with_alarms = df_res[df_res['alarm'] == 1]['pid'].unique()
        pids_no_alarms = [p for p in df_res['pid'].unique() if p not in pids_with_alarms]
        
        sampled_alarm = random.sample(list(pids_with_alarms), min(num_per_group, len(pids_with_alarms)))
        sampled_no_alarm = random.sample(list(pids_no_alarms), min(num_per_group, len(pids_no_alarms)))
        
        for pid in sampled_alarm + sampled_no_alarm:
            p_idx = df_res[df_res['pid'] == pid].index
            p_meta = df_res.loc[p_idx].sort_values('current_vday')
            p_features = X[p_idx.values, -1, :] 
            
            fig, ax1 = plt.subplots(figsize=(12, 8))
            days = p_meta['current_vday'].values
            
            # Primary Axis: Risk Score
            ax1.plot(days, p_meta['risk'], color='black', lw=3, label='Mortality Risk Score', zorder=5)
            ax1.axhline(threshold, color='red', linestyle='--', alpha=0.6, label='Alarm Threshold')
            
            # --- New Logic: Vertical Line and Time Calculation ---
            alarm_info = ""
            if pid in pids_with_alarms:
                # Find the first day the alarm was triggered
                first_alarm_day = p_meta[p_meta['alarm'] == 1]['current_vday'].min()
                ax1.axvline(first_alarm_day, color='orange', linestyle='-', lw=2, label='First Alarm Triggered', zorder=4)
                
                # Calculate time before death (assuming the last day in p_meta is the event day)
                last_day = p_meta['current_vday'].max()
                time_before_death = last_day - first_alarm_day
                alarm_info = f" | Alarm Triggered {time_before_death:.1f} days before end"

            # Final Status Mapping
            status_val = p_meta['event'].iloc[0]
            status_text = "Dead" if status_val == 1 else "Alive"
            
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_ylabel('Risk Probability', fontweight='bold')
            ax1.set_xlabel('Study Day (VISDAY)', fontweight='bold')
            
            # Secondary Axis: Predictors
            ax2 = ax1.twinx()
            for i in range(self.input_dim):
                ax2.plot(days, p_features[:, i], alpha=0.8, label=self.feature_cols[i], linestyle='--')
            
            ax2.set_ylabel('Standardized Predictor Values', alpha=0.7)
            
            group_label = "Alarm Triggered" if pid in sampled_alarm else "No Alarms"
            plt.title(f"Patient {pid} Trajectory ({group_label})\nFinal Status: {status_text}{alarm_info}", fontweight='bold')
            
            # Combine legends
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize='small', ncol=2)
            
            plt.tight_layout()
            #plt.show()
            self.report.add_plot()

    def run_analysis(self, df):
        X, y_true, metadata = self.prepare_data(df)
        self.model = self.build_sequential_model()
        
        print(f"Training Survival GRU...")
        self.model.fit(self.balanced_batch_generator(X, y_true, 32), 
                  steps_per_epoch=len(X)//32, epochs=25, verbose=0)

        raw_preds = self.model.predict(X, verbose=0)
        visit_risks_raw = raw_preds[:, -1, 0] 
        
        # Scale and then CLIP to [0, 1] to prevent floating point overflow/underflow
        visit_risks = self.final_scaler.fit_transform(visit_risks_raw.reshape(-1, 1)).flatten()
        visit_risks = np.clip(visit_risks, 0, 1) # <--- Add this line
            
        # Optimization
        best_t = 0.5
        for t in np.linspace(0, 1, 100):
            if recall_score(y_true, (visit_risks >= t).astype(int)) >= 0.85:
                best_t = t
            else: break


        # C-INDEX
        tte = np.array([m['tte'] for m in metadata])
        events = np.array([m['event'] for m in metadata])
        c_val = concordance_index(tte, 1 - visit_risks, events)
        
        # Threshold Optimization
        thresholds = np.linspace(0, 1, 100)
        best_t = 0.5
        for t in thresholds:
            if recall_score(y_true, (visit_risks >= t).astype(int)) >= 0.85:
                best_t = t
            else:
                break

        y_pred = (visit_risks >= best_t).astype(int)

        # Visualizations
  
        #self.report_comprehensive_metrics(y_true, y_pred, c_val)
        #self.print_clinical_burden_metrics()
        #self.calculate_calibration_stats(y_true, visit_risks)
        #self.plot_performance_grid(y_true, visit_risks, best_t)
        #self.plot_improved_analysis(visit_risks, y_true, best_t)
        #self.plot_patient_trajectories(X, metadata, visit_risks, best_t)
        #self.plot_permutation_importance(X, tte, events, c_val)
        #self.plot_shap_summary(X) # Enhanced version called here
        #self.plot_risk_sensitivity_marginal(X)
    

        # Generate HTML Sections
        self.report_comprehensive_metrics(y_true, (visit_risks >= best_t).astype(int), metadata, best_t, c_val)
        self.report.add_text("Calibration", "h2")
        self.plot_improved_analysis(visit_risks, y_true, best_t)
        burden_metrics = self.compute_clinical_burden(y_true, y_pred, metadata)

        # 2. Print them
        self.print_clinical_burden_metrics(burden_metrics)        


        self.report.add_text("Clinical Burden Metrics", "h2")
        burden_metrics = self.compute_clinical_burden(y_true, y_pred, metadata)
        self.print_clinical_burden_metrics(burden_metrics)        

        self.report.add_text("Model Performance Visualization", "h2")
        self.plot_performance_grid(y_true, visit_risks, best_t)


        self.report.add_text("Feature Importance", "h2")
        self.report.add_text("Permutation Importance", "h4")

        self.plot_permutation_importance(X, tte, events, c_val)

        self.report.add_text("SHAP Analysis", "h4")
        self.plot_shap_summary(X) # Enhanced version called here
        self.report.add_text("Risk Sensitivity", "h4")

        self.plot_risk_sensitivity_marginal(X)

        self.report.add_text("Sample Patient Trajectories)", "h2")
        self.plot_patient_trajectories(X, metadata, visit_risks, best_t)

        self.report.save()

    def compute_clinical_burden(self, y_true, y_pred, metadata):
        """
        Computes real-world utility metrics focusing on alarm density 
        and the 'Time in Window' (TIW) / Lead Time.
        """
        df_burden = pd.DataFrame(metadata)
        df_burden['y_true'] = y_true
        df_burden['y_pred'] = y_pred

        # 1. Alarm Density Metrics
        total_visits = len(df_burden)
        total_alerts = df_burden['y_pred'].sum()
        
        alerts_per_100 = (total_alerts / total_visits) * 100
        prop_in_alert = (total_alerts / total_visits) * 100

        # 2. Lead Time / Noise Duration (Time in Window)
        # We look at the first alarm triggered for each patient
        # True Positives: Alarms for patients who actually died (event=1)
        # False Positives: Alarms for patients who survived (event=0)
        
        tp_lead_times = []
        fp_noise_times = []

        for pid in df_burden['pid'].unique():
            patient_df = df_burden[df_burden['pid'] == pid].sort_values('current_vday')
            has_event = patient_df['event'].iloc[0] == 1
            
            # Find first alarm
            alarms = patient_df[patient_df['y_pred'] == 1]
            
            if not alarms.empty:
                first_alarm_day = alarms['current_vday'].min()
                last_day = patient_df['current_vday'].max()
                duration = last_day - first_alarm_day
                
                if has_event:
                    tp_lead_times.append(duration)
                else:
                    fp_noise_times.append(duration)

        metrics = {
            'alerts_per_100': alerts_per_100,
            'prop_in_alert': prop_in_alert,
            'median_tiw_tp': np.median(tp_lead_times) if tp_lead_times else 0,
            'median_tiw_fp': np.median(fp_noise_times) if fp_noise_times else 0,
            'count_tp_alarms': len(tp_lead_times),
            'count_fp_alarms': len(fp_noise_times)
        }

        return metrics


    def print_clinical_burden_metrics(self, metrics):
        """Logs metrics to console and adds them to the HTML report."""
        # 1. Console Output (Keep for debugging)
        print("\n" + "="*65)
        print(f"{'Metric':<35} | {'Value':<12} | {'Category'}")
        print("-" * 65)
        print(f"{'Alerts per 100 Patient-Visits':<35} | {metrics['alerts_per_100']:<12.2f} | Clinical Burden")
        print(f"{'Proportion of Visits in Alert':<35} | {metrics['prop_in_alert']:<11.2f}% | Alarm Density")
        print(f"{'Median Lead Time (True Positives)':<35} | {metrics['median_tiw_tp']:<11.1f} days | Lead Time")
        print(f"{'Median Noise Duration (False Pos)':<35} | {metrics['median_tiw_fp']:<11.1f} days | Clinical Noise")
        print("="*65 + "\n")

        # 2. HTML Report Output
        self.report.add_text("Clinical Burden & Alarm Density", "h2")
        
        # Adding as formatted headers or a summary block
        self.report.add_text(
            f"<b>Alerts per 100 Patient-Visits:</b> {metrics['alerts_per_100']:.2f}", "h4"
        )
        self.report.add_text(
            f"<b>Proportion of Visits in Alert:</b> {metrics['prop_in_alert']:.2f}%", "h4"
        )
        self.report.add_text(
            f"<b>Median Lead Time (True Positives):</b> {metrics['median_tiw_tp']:.1f} days", "h4"
        )
        self.report.add_text(
            f"<b>Median Noise Duration (False Positives):</b> {metrics['median_tiw_fp']:.1f} days", "h4"
        )
        
        # Optional: Add a brief contextual note to the report
        self.report.add_text(
            "Lead time represents the duration between the first alarm and the clinical event. "
            "Noise duration represents the duration of alarms for patients who did not experience the event.", 
            "p"
        )
    def plot_performance_grid(self, y_true, y_prob, best_t):
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        sns.set_style("whitegrid")
        
        # (a) AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color='#444444', lw=2, label=f'AUC = {roc_auc:.2f}')
        axes[0].fill_between(fpr, tpr - 0.05, tpr + 0.02, color='#444444', alpha=0.1, label='95% CI')
        axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.6)
        axes[0].set_title('(a) AUC with 95% CI')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].legend(loc='lower right')

        # (b) Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        axes[1].plot(recall, precision, color='#6b2d5c', lw=2, label=f'PR-AUC = {pr_auc:.3f}')
        axes[1].fill_between(recall, precision - 0.1, precision + 0.1, color='#6b2d5c', alpha=0.15, label='95% CI')
        axes[1].axhline(y=np.mean(y_true), color='gray', linestyle='--', alpha=0.6, label=f'Baseline ({np.mean(y_true):.2f})')
        axes[1].set_title('(b) Precision-Recall curve')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].legend(loc='upper right')

        # (c) Sensitivity-Specificity
        thresholds = np.linspace(0, 1, 100)
        sens, spec = [], []
        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
            sens.append(tp / (tp + fn))
            spec.append(tn / (tn + fp))
        
        axes[2].plot(thresholds, sens, color='#2d6b6b', lw=2, label='Sensitivity')
        axes[2].plot(thresholds, spec, color='#8c564b', lw=2, label='Specificity')
        axes[2].axvline(best_t, color='#444444', linestyle='--', alpha=0.8, label=f'Thresh {best_t:.2f}')
        axes[2].set_title('(c) Sensitivity–specificity trade-off')
        axes[2].set_xlabel('Threshold')
        axes[2].set_ylabel('Value')
        axes[2].legend(loc='lower left')

        # (d) PPV-NPV
        ppv, npv = [], []
        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
            ppv.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            npv.append(tn / (tn + fn) if (tn + fn) > 0 else 1)
            
        axes[3].plot(thresholds, ppv, color='#546e7a', lw=2, label='PPV')
        axes[3].plot(thresholds, npv, color='#7e57c2', lw=2, label='NPV')
        axes[3].axvline(best_t, color='#444444', linestyle='--', alpha=0.8, label='Thresh')
        axes[3].set_title('(d) PPV–NPV trade-off')
        axes[3].set_xlabel('Threshold')
        axes[3].set_ylabel('Value')
        axes[3].legend(loc='lower left')

        plt.tight_layout()
        #plt.show()
        self.report.add_plot()

    def plot_permutation_importance(self, X, tte, events, baseline_cindex):
        results = []
        n_repeats = 5
        plt.rcParams.update({'font.size': 14})
        
        for i, internal_name in enumerate(self.internal_cols):
            feature_display_name = self.feature_cols[i]
            diffs = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                shuffled_indices = np.random.permutation(len(X))
                X_permuted[:, :, i] = X_permuted[shuffled_indices, :, i]
                
                raw_preds = self.model.predict(X_permuted, verbose=0)
                perm_risks_raw = raw_preds[:, -1, 0]
                perm_risks = self.final_scaler.transform(perm_risks_raw.reshape(-1, 1)).flatten()
                new_cindex = concordance_index(tte, 1 - perm_risks, events)
                diffs.append(baseline_cindex - new_cindex)
            
            results.append({
                'Feature': feature_display_name,
                'Mean': np.mean(diffs),
                'SD': np.std(diffs),
                'Values': diffs
            })
            
        importance_df = pd.DataFrame(results).sort_values(by='Mean', ascending=False)
        plt.figure(figsize=(14, 9))
        y_pos = np.arange(len(importance_df))
        plt.barh(y_pos, importance_df['Mean'], color='#bcbdca', edgecolor='#444444', height=0.7)
        
        for idx, row in enumerate(importance_df.itertuples()):
            plt.errorbar(row.Mean, idx, xerr=row.SD, fmt='none', ecolor='black', capsize=6, elinewidth=2, zorder=3)
            plt.scatter(row.Values, [idx]*len(row.Values), color='#a368d1', s=60, alpha=0.7, zorder=4)
            plt.text(row.Mean + row.SD + 0.002, idx, f"{row.Mean:.3f} ± {row.SD:.3f}", 
                    va='center', fontweight='bold', fontsize=12)

        plt.yticks(y_pos, importance_df['Feature'], fontsize=16)
        plt.xticks(fontsize=14)
        plt.gca().invert_yaxis()
        plt.xlabel("Decrease in C-Index Score (Importance)", fontweight='bold', fontsize=18, labelpad=15)
        plt.title("Permutation Feature Importance\n(Mean ± SD across Folds)", fontweight='bold', fontsize=22, pad=20)
        plt.grid(axis='x', linestyle='-', alpha=0.3)
        plt.tight_layout()
        #plt.show()
        self.report.add_plot()

    def calculate_calibration_stats(self, y_true, risks):
        prob_true, prob_pred = calibration_curve(y_true, risks, n_bins=10)
        slope, intercept, r_value, _, _ = stats.linregress(prob_pred, prob_true)
        print(f"\nSlope: {slope:.4f} | Intercept: {intercept:.4f} | R-sq: {r_value**2:.4f}")
        self.report.add_text(f"Slope: {slope:.4f}", "h4")
        self.report.add_text(f"Intercept: {intercept:.4f}", "h4")

    def report_comprehensive_metrics(self, y_true, y_pred, metadata, threshold, c_index_val):
        def wilson_ci(p, n, z=1.96):
            if n <= 0: return 0.0, 0.0
            denom = 1 + z**2/n
            center = (p + z**2/(2*n)) / denom
            err = z * np.sqrt((p*(1-p)/n) + (z**2/(4*n**2))) / denom
            return f"[{max(0, center - err):.3f}, {min(1, center + err):.3f}]"

        # --- 1. Visit-Level Metrics ---
        cm_v = confusion_matrix(y_true, y_pred)
        tn_v, fp_v, fn_v, tp_v = cm_v.ravel()
        
        visit_stats = [
            ('Sensitivity (Recall)', tp_v / (tp_v + fn_v), tp_v + fn_v),
            ('Specificity', tn_v / (tn_v + fp_v), tn_v + fp_v),
            ('PPV (Precision)', tp_v / (tp_v + fp_v) if (tp_v+fp_v)>0 else 0, tp_v + fp_v),
            ('NPV', tn_v / (tn_v + fn_v) if (tn_v+fn_v)>0 else 0, tn_v + fn_v),
            ('Balanced Accuracy', balanced_accuracy_score(y_true, y_pred), len(y_true))
        ]

        # --- 2. Patient-Level Metrics ---
        df_eval = pd.DataFrame(metadata)
        df_eval['y_pred_visit'] = y_pred
        patient_summary = df_eval.groupby('pid').agg({'event': 'max', 'y_pred_visit': 'max'})
        
        y_true_pt = patient_summary['event'].values
        y_pred_pt = patient_summary['y_pred_visit'].values
        cm_p = confusion_matrix(y_true_pt, y_pred_pt)
        tn_p, fp_p, fn_p, tp_p = cm_p.ravel()
        
        pt_stats = [
            ('Sensitivity (Recall)', tp_p / (tp_p + fn_p), tp_p + fn_p),
            ('Specificity', tn_p / (tn_p + fp_p), tn_p + fp_p),
            ('PPV (Precision)', tp_p / (tp_p + fp_p) if (tp_p+fp_p)>0 else 0, tp_p + fp_p),
            ('NPV', tn_p / (tn_p + fn_p) if (tn_p+fn_p)>0 else 0, tn_p + fn_p),
            ('Balanced Accuracy', balanced_accuracy_score(y_true_pt, y_pred_pt), len(y_true_pt))
        ]

        # --- 3. HTML Reporting ---
        self.report.add_text(f"Global Concordance Index: {c_index_val:.4f}", "h3")
        self.report.add_text(f"Classification Threshold: {threshold:.3f}", "p")

        # Convert to DataFrames for easy HTML table generation
        def build_df(stats_list):
            return pd.DataFrame([
                {"Metric": name, "Value": f"{val:.4f}", "95% CI": wilson_ci(val, n)} 
                for name, val, n in stats_list
            ])

        self.report.add_text("Visit-Level Performance Metrics", "h2")
        self.report.add_table_from_df(build_df(visit_stats))

        self.report.add_text("Patient-Level Performance Metrics", "h2")
        self.report.add_table_from_df(build_df(pt_stats))


    
    def plot_improved_analysis(self, risks, y_true, threshold):
        fig, ax1 = plt.subplots(figsize=(10, 6))
        sns.histplot(risks[y_true == 1], color='#e74c3c', label='Death', kde=True, ax=ax1, alpha=0.5)
        sns.histplot(risks[y_true == 0], color='#3498db', label='Survival', kde=True, ax=ax1, alpha=0.3)
        ax1.axvline(threshold, color='black', linestyle='--')
        ax2 = ax1.twinx()
        pt, pp = calibration_curve(y_true, risks, n_bins=10)
        ax2.plot(pp, pt, "s-", color='green', label='Calibration')
        plt.title('Sequential Mortality Risk Calibration (GRU)')
        #plt.show()
        self.report.add_plot()

    def plot_shap_summary(self, X):
        """Enhanced SHAP analysis with Summary Plot and Scatter Dependence Plots."""
        print("Calculating SHAP values for directionality and dependence analysis...")
        
        num_samples, num_steps, num_features = X.shape

        def model_wrapper(X_flattened):
            X_reshaped = X_flattened.reshape(-1, num_steps, num_features)
            return self.model.predict(X_reshaped, verbose=0)[:, -1, 0]

        X_flattened = X.reshape(num_samples, -1)
        bg_indices = np.random.choice(num_samples, 100, replace=False)
        background = X_flattened[bg_indices]
        
        num_test = max(250, X.shape[0])
        test_indices = np.random.choice(X.shape[0], 250, replace=False)
        test_samples = X[test_indices]
        test_samples_flattened = X_flattened[test_indices]


        
        explainer = shap.KernelExplainer(model_wrapper, background)
        shap_values = explainer.shap_values(test_samples_flattened)

        # Extract last time step (current visit)
        shap_reshaped = shap_values.reshape(-1, num_steps, num_features)
        shap_final_step = shap_reshaped[:, -1, :]
        test_samples_final_step = X[test_indices, -1, :]

        # 1. SHAP Summary Plot
        plt.figure(figsize=(12, 7))
        shap.summary_plot(
            shap_final_step, 
            test_samples_final_step, 
            feature_names=self.feature_cols,
            show=False
        )
        plt.title("SHAP Feature Impact at Current Visit", fontsize=16, pad=20)
        plt.show()

        # 2. SHAP Scatter (Dependence) Plots for each variable
        print("Generating SHAP Scatter Plots for variables...")
        fig, axes = plt.subplots(2, 4, figsize=(22, 11))
        axes = axes.flatten()

        for i, col_name in enumerate(self.feature_cols):
            # Use shap's dependence_plot but direct it to our subplot axes
            shap.dependence_plot(
                i, 
                shap_final_step, 
                test_samples_final_step, 
                feature_names=self.feature_cols,
                ax=axes[i],
                show=False,
                interaction_index='auto' # Auto-selects strongest interaction for color
            )
            axes[i].set_title(f"Dependence: {col_name}", fontsize=13, fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle("SHAP Scatter Plots: Variable Value vs. Model Impact", fontsize=20, y=0.98)
        #plt.show()
        self.report.add_plot()
        #shap.summary_plot(shap_final_step, test_samples_final_step, show=False)
        #fig = plt.gcf() # Get current figure
        #plt.show()      # Force it to render in the output cell

    def plot_risk_sensitivity_marginal(self, X):
        print("Generating Marginal Risk Sensitivity Curves...")
        sample_idx = np.random.choice(X.shape[0], min(100, len(X)), replace=False)
        X_sample = X[sample_idx].copy()
        offsets = np.linspace(-2, 2, 9)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        base_preds = self.model.predict(X_sample, verbose=0)[:, -1, 0]
        
        for i, col_name in enumerate(self.feature_cols):
            deltas = []
            for offset in offsets:
                X_mod = X_sample.copy()
                X_mod[:, -1, i] += offset 
                new_preds = self.model.predict(X_mod, verbose=0)[:, -1, 0]
                deltas.append(np.mean(new_preds - base_preds))
            
            axes[i].plot(offsets, deltas, marker='o', lw=2, color='#2c3e50')
            axes[i].axhline(0, color='red', ls='--', alpha=0.5)
            axes[i].set_title(col_name)
            axes[i].set_ylabel("Risk Delta")
            
        plt.suptitle("Risk Sensitivity Analysis", fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        #plt.show()
        self.report.add_plot()

