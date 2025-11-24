import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import re

def create_ablation_plots(metrics_dir, output_dir=None, val_color='#1F3E74', test_color='#4469AD', 
                         fixed_scales=None, experiment_order=None):
    """
    Erstellt Säulendiagramme für Ablation Study Ergebnisse.
    
    Args:
        metrics_dir (str): Pfad zum Directory mit den CSV-Dateien
        output_dir (str): Pfad zum Output Directory (optional)
        val_color (str): Hex-Farbe für Validation Balken
        test_color (str): Hex-Farbe für Test Balken
        fixed_scales (dict): Feste Y-Achsen Bereiche pro Metrik, z.B. {'Accuracy': (0, 1), 'MSE': (0, 20)}
        experiment_order (list): Gewünschte Reihenfolge der Experimente, z.B. ['Baseline', 'NoCoarsePreds', ...]
    """
    
    if output_dir is None:
        output_dir = metrics_dir
    
    # CSV-Dateien finden
    csv_pattern = os.path.join(metrics_dir, "overall_metrics_*.csv")
    csv_files = glob(csv_pattern)
    
    if not csv_files:
        print(f"Keine CSV-Dateien gefunden in {metrics_dir}")
        return
    
    # Daten sammeln
    data = {}
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        
        # Experiment Name und Dataset aus Dateiname extrahieren
        match = re.match(r'overall_metrics_(.+)_(val|test)\.csv', filename)
        if not match:
            print(f"Dateiname {filename} folgt nicht dem erwarteten Muster")
            continue
            
        experiment_name = match.group(1)
        dataset_type = match.group(2)
        
        # CSV laden
        df = pd.read_csv(csv_file, index_col=0)
        
        # Daten speichern
        if experiment_name not in data:
            data[experiment_name] = {}
        data[experiment_name][dataset_type] = df
    
    if not data:
        print("Keine gültigen Daten gefunden")
        return
    
    # Metriken aus dem ersten Dataset extrahieren
    first_experiment = next(iter(data.values()))
    first_dataset = next(iter(first_experiment.values()))
    metrics = first_dataset.columns.tolist()
    error_types = first_dataset.index.tolist()
    
    print(f"Gefundene Experimente: {list(data.keys())}")
    print(f"Gefundene Metriken: {metrics}")
    print(f"Gefundene Fehlertypen: {error_types}")
    
    # Für jede Metrik ein Plot erstellen
    for metric in metrics:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Ablation Study Results - {metric}', fontsize=16, fontweight='bold')
        
        # Globale Skala für diese Metrik berechnen (falls nicht manuell gesetzt)
        if fixed_scales and metric in fixed_scales:
            y_min, y_max = fixed_scales[metric]
        else:
            # Automatische Skalierung basierend auf allen Werten für diese Metrik
            all_values = []
            for exp_data in data.values():
                for dataset_type in ['val', 'test']:
                    if dataset_type in exp_data:
                        for error_type in error_types:
                            all_values.append(exp_data[dataset_type].loc[error_type, metric])
            
            y_min = 0  # Immer bei 0 beginnen
            y_max = max(all_values) * 1.1  # 10% Puffer über dem Maximum
        
        axes = axes.flatten()
        
        for idx, error_type in enumerate(error_types):
            ax = axes[idx]
            
            # Daten für diesen Fehlertyp und Metrik sammeln
            experiments = []
            val_values = []
            test_values = []
            
            # Experiment-Reihenfolge bestimmen
            if experiment_order:
                # Verwendete Reihenfolge aus experiment_order, falls verfügbar
                exp_names = [exp for exp in experiment_order if exp in data and 'val' in data[exp] and 'test' in data[exp]]
                # Füge alle anderen Experimente hinzu, die nicht in experiment_order sind
                exp_names.extend([exp for exp in data.keys() if exp not in exp_names and 'val' in data[exp] and 'test' in data[exp]])
            else:
                # Standardmäßig alphabetische Sortierung
                exp_names = sorted([exp for exp in data.keys() if 'val' in data[exp] and 'test' in data[exp]])
            
            for exp_name in exp_names:
                exp_data = data[exp_name]
                experiments.append(exp_name.replace('_', '\n'))  # Zeilenumbruch für bessere Lesbarkeit
                val_values.append(exp_data['val'].loc[error_type, metric])
                test_values.append(exp_data['test'].loc[error_type, metric])
            
            # Balkendiagramm erstellen
            x = np.arange(len(experiments))
            width = 0.4  # Breitere Balken
            
            bars1 = ax.bar(x - width/2, val_values, width, label='Validation', color=val_color)
            bars2 = ax.bar(x + width/2, test_values, width, label='Test', color=test_color)
            
            # Plot formatieren
            ax.set_title(f'{error_type.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(experiments, rotation=45, ha='right', fontsize=9)
            
            # Feste Y-Achsen Skalierung
            ax.set_ylim(y_min, y_max)
            
            # Legende bei jedem Subplot
            ax.legend(fontsize=9, loc='upper right')
            
            ax.grid(axis='y', alpha=0.3)
            
            # Werte auf den Balken anzeigen (optional)
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Plot speichern
        output_path = os.path.join(output_dir, f'ablation_study_{metric.lower().replace(" ", "_")}.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot gespeichert: {output_path}")
        
        plt.show()

def create_error_type_comparison_plots(metrics_dir, output_dir=None, experiment_order=None, 
                                      bar_width=0.18, fixed_scales=None):
    """
    Erstellt Säulendiagramme zum Vergleich verschiedener Error Types.
    
    Args:
        metrics_dir (str): Pfad zum Directory mit den CSV-Dateien
        output_dir (str): Pfad zum Output Directory (optional)
        experiment_order (list): Gewünschte Reihenfolge der Experimente
        bar_width (float): Breite der einzelnen Balken (Standard: 0.18)
        fixed_scales (dict): Feste Y-Achsen Bereiche pro Metrik
    """
    
    if output_dir is None:
        output_dir = metrics_dir
    
    # Farben für die verschiedenen Error Types
    colors = {
        'coarse_error_val': '#80350E',
        'coarse_error_test': '#C04F15', 
        'refined_error_val': '#112C57',
        'refined_error_test': '#4469AD',
        'coarse_proj_error_val': '#80350E',
        'coarse_proj_error_test': '#C04F15',
        'refined_proj_error_val': '#112C57', 
        'refined_proj_error_test': '#4469AD'
    }
    
    # CSV-Dateien finden und laden
    csv_pattern = os.path.join(metrics_dir, "overall_metrics_*.csv")
    csv_files = glob(csv_pattern)
    
    if not csv_files:
        print(f"Keine CSV-Dateien gefunden in {metrics_dir}")
        return
    
    # Daten sammeln
    data = {}
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        match = re.match(r'overall_metrics_(.+)_(val|test)\.csv', filename)
        if not match:
            continue
            
        experiment_name = match.group(1)
        dataset_type = match.group(2)
        df = pd.read_csv(csv_file, index_col=0)
        
        if experiment_name not in data:
            data[experiment_name] = {}
        data[experiment_name][dataset_type] = df
    
    if not data:
        print("Keine gültigen Daten gefunden")
        return
    
    # Metriken extrahieren
    first_experiment = next(iter(data.values()))
    first_dataset = next(iter(first_experiment.values()))
    metrics = first_dataset.columns.tolist()
    
    print(f"Gefundene Experimente: {list(data.keys())}")
    print(f"Gefundene Metriken: {metrics}")
    
    # Experiment-Reihenfolge bestimmen
    if experiment_order:
        exp_names = [exp for exp in experiment_order if exp in data and 'val' in data[exp] and 'test' in data[exp]]
        exp_names.extend([exp for exp in data.keys() if exp not in exp_names and 'val' in data[exp] and 'test' in data[exp]])
    else:
        exp_names = sorted([exp for exp in data.keys() if 'val' in data[exp] and 'test' in data[exp]])
    
    # Für jede Metrik zwei Plots erstellen
    for metric in metrics:
        # Plot 1: coarse_error vs refined_error
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot 2: coarse_proj_error vs refined_proj_error  
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        
        # Globale Skala für diese Metrik berechnen
        if fixed_scales and metric in fixed_scales:
            y_min, y_max = fixed_scales[metric]
        else:
            all_values = []
            for exp_data in data.values():
                for dataset_type in ['val', 'test']:
                    if dataset_type in exp_data:
                        for error_type in ['coarse_error', 'refined_error', 'coarse_proj_error', 'refined_proj_error']:
                            all_values.append(exp_data[dataset_type].loc[error_type, metric])
            y_min = 0
            y_max = max(all_values) * 1.1
        
        # Daten sammeln
        experiments_display = [exp.replace('_', '\n') for exp in exp_names]
        x = np.arange(len(exp_names))
        
        # Plot 1: coarse_error vs refined_error
        coarse_val = [data[exp]['val'].loc['coarse_error', metric] for exp in exp_names]
        coarse_test = [data[exp]['test'].loc['coarse_error', metric] for exp in exp_names]
        refined_val = [data[exp]['val'].loc['refined_error', metric] for exp in exp_names]
        refined_test = [data[exp]['test'].loc['refined_error', metric] for exp in exp_names]
        
        bars1 = ax1.bar(x - 1.5*bar_width, coarse_val, bar_width, 
                       label='Coarse Error - Val', color=colors['coarse_error_val'])
        bars2 = ax1.bar(x - 0.5*bar_width, coarse_test, bar_width, 
                       label='Coarse Error - Test', color=colors['coarse_error_test'])
        bars3 = ax1.bar(x + 0.5*bar_width, refined_val, bar_width, 
                       label='Refined Error - Val', color=colors['refined_error_val'])
        bars4 = ax1.bar(x + 1.5*bar_width, refined_test, bar_width, 
                       label='Refined Error - Test', color=colors['refined_error_test'])
        
        # Plot 1 formatieren
        ax1.set_title(f'Error Type Comparison - {metric} (Coarse vs Refined)', fontsize=14, fontweight='bold')
        ax1.set_ylabel(metric, fontsize=12)
        ax1.set_xlabel('Experiments', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(experiments_display, rotation=45, ha='right')
        ax1.set_ylim(y_min, y_max)
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Werte auf Balken anzeigen (Plot 1)
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: coarse_proj_error vs refined_proj_error
        coarse_proj_val = [data[exp]['val'].loc['coarse_proj_error', metric] for exp in exp_names]
        coarse_proj_test = [data[exp]['test'].loc['coarse_proj_error', metric] for exp in exp_names]
        refined_proj_val = [data[exp]['val'].loc['refined_proj_error', metric] for exp in exp_names]
        refined_proj_test = [data[exp]['test'].loc['refined_proj_error', metric] for exp in exp_names]
        
        bars5 = ax2.bar(x - 1.5*bar_width, coarse_proj_val, bar_width, 
                       label='Coarse Proj Error - Val', color=colors['coarse_proj_error_val'])
        bars6 = ax2.bar(x - 0.5*bar_width, coarse_proj_test, bar_width, 
                       label='Coarse Proj Error - Test', color=colors['coarse_proj_error_test'])
        bars7 = ax2.bar(x + 0.5*bar_width, refined_proj_val, bar_width, 
                       label='Refined Proj Error - Val', color=colors['refined_proj_error_val'])
        bars8 = ax2.bar(x + 1.5*bar_width, refined_proj_test, bar_width, 
                       label='Refined Proj Error - Test', color=colors['refined_proj_error_test'])
        
        # Plot 2 formatieren
        ax2.set_title(f'Error Type Comparison - {metric} (Coarse Proj vs Refined Proj)', fontsize=14, fontweight='bold')
        ax2.set_ylabel(metric, fontsize=12)
        ax2.set_xlabel('Experiments', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(experiments_display, rotation=45, ha='right')
        ax2.set_ylim(y_min, y_max)
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(axis='y', alpha=0.3)
        
        # Werte auf Balken anzeigen (Plot 2)
        for bars in [bars5, bars6, bars7, bars8]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plots speichern
        fig1.tight_layout()
        fig2.tight_layout()
        
        output_path1 = os.path.join(output_dir, f'error_comparison_coarse_vs_refined_{metric.lower().replace(" ", "_")}.pdf')
        output_path2 = os.path.join(output_dir, f'error_comparison_proj_{metric.lower().replace(" ", "_")}.pdf')
        
        fig1.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
        fig2.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
        
        print(f"Plot gespeichert: {output_path1}")
        print(f"Plot gespeichert: {output_path2}")
        
        plt.show()

def create_single_metric_plot(metrics_dir, metric_name, error_type, output_dir=None, 
                            val_color='#1F3E74', test_color='#4469AD', figsize=(12, 6), y_range=None):
    """
    Erstellt ein einzelnes Säulendiagramm für eine spezifische Metrik und Fehlertyp.
    
    Args:
        metrics_dir (str): Pfad zum Directory mit den CSV-Dateien
        metric_name (str): Name der Metrik (z.B. 'Accuracy')
        error_type (str): Fehlertyp (z.B. 'refined_error')
        output_dir (str): Pfad zum Output Directory (optional)
        val_color (str): Hex-Farbe für Validation Balken
        test_color (str): Hex-Farbe für Test Balken
        figsize (tuple): Größe der Figur
        y_range (tuple): Y-Achsen Bereich, z.B. (0, 1) für Accuracy
    """
    
    if output_dir is None:
        output_dir = metrics_dir
    
    # CSV-Dateien finden und laden
    csv_pattern = os.path.join(metrics_dir, "overall_metrics_*.csv")
    csv_files = glob(csv_pattern)
    
    data = {}
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        match = re.match(r'overall_metrics_(.+)_(val|test)\.csv', filename)
        if not match:
            continue
            
        experiment_name = match.group(1)
        dataset_type = match.group(2)
        
        df = pd.read_csv(csv_file, index_col=0)
        
        if experiment_name not in data:
            data[experiment_name] = {}
        data[experiment_name][dataset_type] = df
    
    # Daten für Plot sammeln
    experiments = []
    val_values = []
    test_values = []
    
    for exp_name, exp_data in data.items():
        if 'val' in exp_data and 'test' in exp_data:
            experiments.append(exp_name.replace('_', ' '))
            val_values.append(exp_data['val'].loc[error_type, metric_name])
            test_values.append(exp_data['test'].loc[error_type, metric_name])
    
    # Plot erstellen
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(experiments))
    width = 0.4  # Breitere Balken für weniger Lücke zwischen val/test
    
    bars1 = ax.bar(x - width/2, val_values, width, label='Validation', color=val_color)
    bars2 = ax.bar(x + width/2, test_values, width, label='Test', color=test_color)
    
    # Plot formatieren
    ax.set_title(f'Ablation Study - {metric_name} ({error_type.replace("_", " ").title()})', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_xlabel('Experiments', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    
    # Feste Y-Achsen Skalierung
    if y_range:
        ax.set_ylim(y_range[0], y_range[1])
    
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Werte auf Balken anzeigen
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Plot speichern
    output_path = os.path.join(output_dir, f'ablation_{metric_name.lower()}_{error_type}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot gespeichert: {output_path}")
    
    plt.show()

# Beispiel für die Verwendung:
if __name__ == "__main__":
    # Feste Skalen für konsistente Darstellung definieren
    fixed_scales = {
        'Accuracy': (0, 1.0)
    }
    
    # Experiment-Reihenfolge definieren (optional)
    #experiment_order = ['all-pois', 'excel-exclude', 'excel-outliers-exclude', 'include-com']
    #experiment_order = ['subreg-0.5-zoom', 'subreg-1.0-zoom', 'subreg-2.0-zoom', 'vertseg', 'surface-mask', 'ct-scan', 'neighbors']
    experiment_order = ['standard-architecture', 'only-coarse-module', 'no-coarse-preds', 'no-global-features', 'no-poi-id', 'no-vert-id', 'no-poi-vert-id', 'no-patch-features', 'no-poi-vert-id-global-features', 'no-projection']
    # Alle Plots mit festen Skalen erstellen
    create_ablation_plots('architecture/run_1', #'dataloader/include_pois/run_1',
                         fixed_scales=fixed_scales,
                         experiment_order=experiment_order,
                         val_color='#1F3E74', 
                         test_color='#4469AD')
    
    # Neue Error-Type Vergleichsplots erstellen
    create_error_type_comparison_plots('architecture/run_1', #'dataloader/include_pois/run_1', #
                                      experiment_order=experiment_order,
                                      bar_width=0.2,  # Säulenbreite kontrollieren
                                      fixed_scales=fixed_scales)
    
    # Oder nur einen spezifischen Plot mit fester Skala
    #create_single_metric_plot('ablation_study_metrics', 
    #                         'Accuracy', 
    #                         'refined_error',
    #                         y_range=(0, 1.0),
    #                         val_color='#1F3E74', 
    #                         test_color='#4469AD')