import os
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import numpy as np
from glob import glob
import re

def create_ablation_plots(metrics_dir, output_dir=None, val_color='#1F3E74', test_color='#4469AD', 
                         fixed_scales=None, experiment_order=None):
    """
    Erstellt Säulendiagramme für Ablation Study Ergebnisse mit Plotly.
    
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
        # Subplots erstellen (2x2 Grid)
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[error_type.replace("_", " ").title() for error_type in error_types],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
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
        
        for idx, error_type in enumerate(error_types):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            # Daten für diesen Fehlertyp und Metrik sammeln
            experiments = []
            val_values = []
            test_values = []
            
            # Experiment-Reihenfolge bestimmen
            if experiment_order:
                exp_names = [exp for exp in experiment_order if exp in data and 'val' in data[exp] and 'test' in data[exp]]
                exp_names.extend([exp for exp in data.keys() if exp not in exp_names and 'val' in data[exp] and 'test' in data[exp]])
            else:
                exp_names = sorted([exp for exp in data.keys() if 'val' in data[exp] and 'test' in data[exp]])
            
            for exp_name in exp_names:
                exp_data = data[exp_name]
                experiments.append(exp_name.replace('_', '<br>'))  # HTML Zeilenumbruch
                val_values.append(exp_data['val'].loc[error_type, metric])
                test_values.append(exp_data['test'].loc[error_type, metric])
            
            # Validation Bars hinzufügen
            fig.add_trace(
                go.Bar(
                    name='Validation' if idx == 0 else None,
                    x=experiments,
                    y=val_values,
                    marker_color=val_color,
                    text=[f'{val:.2f}' for val in val_values],
                    textposition='outside',
                    textfont=dict(size=10),
                    offsetgroup=1,
                    legendgroup='validation',
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )
            
            # Test Bars hinzufügen
            fig.add_trace(
                go.Bar(
                    name='Test' if idx == 0 else None,
                    x=experiments,
                    y=test_values,
                    marker_color=test_color,
                    text=[f'{val:.2f}' for val in test_values],
                    textposition='outside',
                    textfont=dict(size=10),
                    offsetgroup=2,
                    legendgroup='test',
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )
            
            # Y-Achsen Bereich setzen
            fig.update_yaxes(range=[y_min, y_max], row=row, col=col)
            fig.update_yaxes(title_text=metric, row=row, col=col)
            fig.update_xaxes(tickangle=45, row=row, col=col)
        
        # Layout aktualisieren
        fig.update_layout(
            title=dict(
                text=f'Ablation Study Results - {metric}',
                font=dict(size=18, family="Arial Black"),
                x=0.5
            ),
            barmode='group',
            height=800,
            width=1200,
            font=dict(size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white'
        )
        
        # Plot speichern (HTML und PDF)
        output_path_html = os.path.join(output_dir, f'ablation_study_{metric.lower().replace(" ", "_")}.html')
        output_path_pdf = os.path.join(output_dir, f'ablation_study_{metric.lower().replace(" ", "_")}.pdf')
        
        fig.write_html(output_path_html)
        fig.write_image(output_path_pdf, width=1200, height=800, scale=2)
        
        print(f"Plot gespeichert: {output_path_html}")
        print(f"Plot gespeichert: {output_path_pdf}")
        
        fig.show()

def create_error_type_comparison_plots(metrics_dir, output_dir=None, experiment_order=None, 
                                      bar_width=0.18, fixed_scales=None):
    """
    Erstellt Säulendiagramme zum Vergleich verschiedener Error Types mit Plotly.
    
    Args:
        metrics_dir (str): Pfad zum Directory mit den CSV-Dateien
        output_dir (str): Pfad zum Output Directory (optional)
        experiment_order (list): Gewünschte Reihenfolge der Experimente
        bar_width (float): Wird bei Plotly nicht verwendet (automatische Spacing)
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
    
    experiments_display = [exp.replace('_', '<br>') for exp in exp_names]
    
    # Für jede Metrik zwei Plots erstellen
    for metric in metrics:
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
        
        # Plot 1: coarse_error vs refined_error
        fig1 = go.Figure()
        
        # Daten sammeln
        coarse_val = [data[exp]['val'].loc['coarse_error', metric] for exp in exp_names]
        coarse_test = [data[exp]['test'].loc['coarse_error', metric] for exp in exp_names]
        refined_val = [data[exp]['val'].loc['refined_error', metric] for exp in exp_names]
        refined_test = [data[exp]['test'].loc['refined_error', metric] for exp in exp_names]
        
        # Bars hinzufügen
        fig1.add_trace(go.Bar(
            name='Coarse Error - Val',
            x=experiments_display,
            y=coarse_val,
            marker_color=colors['coarse_error_val'],
            text=[f'{val:.2f}' for val in coarse_val],
            textposition='outside'
        ))
        
        fig1.add_trace(go.Bar(
            name='Coarse Error - Test',
            x=experiments_display,
            y=coarse_test,
            marker_color=colors['coarse_error_test'],
            text=[f'{val:.2f}' for val in coarse_test],
            textposition='outside'
        ))
        
        fig1.add_trace(go.Bar(
            name='Refined Error - Val',
            x=experiments_display,
            y=refined_val,
            marker_color=colors['refined_error_val'],
            text=[f'{val:.2f}' for val in refined_val],
            textposition='outside'
        ))
        
        fig1.add_trace(go.Bar(
            name='Refined Error - Test',
            x=experiments_display,
            y=refined_test,
            marker_color=colors['refined_error_test'],
            text=[f'{val:.2f}' for val in refined_test],
            textposition='outside'
        ))
        
        # Layout für Plot 1
        fig1.update_layout(
            title=f'Error Type Comparison - {metric} (Coarse vs Refined)',
            xaxis_title='Experiments',
            yaxis_title=metric,
            barmode='group',
            height=600,
            width=1000,
            template='plotly_white',
            yaxis=dict(range=[y_min, y_max]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Plot 2: coarse_proj_error vs refined_proj_error
        fig2 = go.Figure()
        
        coarse_proj_val = [data[exp]['val'].loc['coarse_proj_error', metric] for exp in exp_names]
        coarse_proj_test = [data[exp]['test'].loc['coarse_proj_error', metric] for exp in exp_names]
        refined_proj_val = [data[exp]['val'].loc['refined_proj_error', metric] for exp in exp_names]
        refined_proj_test = [data[exp]['test'].loc['refined_proj_error', metric] for exp in exp_names]
        
        fig2.add_trace(go.Bar(
            name='Coarse Proj Error - Val',
            x=experiments_display,
            y=coarse_proj_val,
            marker_color=colors['coarse_proj_error_val'],
            text=[f'{val:.2f}' for val in coarse_proj_val],
            textposition='outside'
        ))
        
        fig2.add_trace(go.Bar(
            name='Coarse Proj Error - Test',
            x=experiments_display,
            y=coarse_proj_test,
            marker_color=colors['coarse_proj_error_test'],
            text=[f'{val:.2f}' for val in coarse_proj_test],
            textposition='outside'
        ))
        
        fig2.add_trace(go.Bar(
            name='Refined Proj Error - Val',
            x=experiments_display,
            y=refined_proj_val,
            marker_color=colors['refined_proj_error_val'],
            text=[f'{val:.2f}' for val in refined_proj_val],
            textposition='outside'
        ))
        
        fig2.add_trace(go.Bar(
            name='Refined Proj Error - Test',
            x=experiments_display,
            y=refined_proj_test,
            marker_color=colors['refined_proj_error_test'],
            text=[f'{val:.2f}' for val in refined_proj_test],
            textposition='outside'
        ))
        
        # Layout für Plot 2
        fig2.update_layout(
            title=f'Error Type Comparison - {metric} (Coarse Proj vs Refined Proj)',
            xaxis_title='Experiments',
            yaxis_title=metric,
            barmode='group',
            height=600,
            width=1000,
            template='plotly_white',
            yaxis=dict(range=[y_min, y_max]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Plots speichern
        output_path1_html = os.path.join(output_dir, f'error_comparison_coarse_vs_refined_{metric.lower().replace(" ", "_")}.html')
        output_path2_html = os.path.join(output_dir, f'error_comparison_proj_{metric.lower().replace(" ", "_")}.html')
        output_path1_pdf = os.path.join(output_dir, f'error_comparison_coarse_vs_refined_{metric.lower().replace(" ", "_")}.pdf')
        output_path2_pdf = os.path.join(output_dir, f'error_comparison_proj_{metric.lower().replace(" ", "_")}.pdf')
        
        fig1.write_html(output_path1_html)
        fig2.write_html(output_path2_html)
        fig1.write_image(output_path1_pdf, width=1000, height=600, scale=2)
        fig2.write_image(output_path2_pdf, width=1000, height=600, scale=2)
        
        print(f"Plot gespeichert: {output_path1_html}")
        print(f"Plot gespeichert: {output_path2_html}")
        print(f"Plot gespeichert: {output_path1_pdf}")
        print(f"Plot gespeichert: {output_path2_pdf}")
        
        fig1.show()
        fig2.show()

def create_single_metric_plot(metrics_dir, metric_name, error_type, output_dir=None, 
                            val_color='#1F3E74', test_color='#4469AD', figsize=(12, 6), y_range=None):
    """
    Erstellt ein einzelnes Säulendiagramm für eine spezifische Metrik und Fehlertyp mit Plotly.
    
    Args:
        metrics_dir (str): Pfad zum Directory mit den CSV-Dateien
        metric_name (str): Name der Metrik (z.B. 'Accuracy')
        error_type (str): Fehlertyp (z.B. 'refined_error')
        output_dir (str): Pfad zum Output Directory (optional)
        val_color (str): Hex-Farbe für Validation Balken
        test_color (str): Hex-Farbe für Test Balken
        figsize (tuple): Größe der Figur (width, height) - wird in Plotly anders verwendet
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
            experiments.append(exp_name.replace('_', '<br>'))
            val_values.append(exp_data['val'].loc[error_type, metric_name])
            test_values.append(exp_data['test'].loc[error_type, metric_name])
    
    # Plot erstellen
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Validation',
        x=experiments,
        y=val_values,
        marker_color=val_color,
        text=[f'{val:.2f}' for val in val_values],
        textposition='outside',
        textfont=dict(size=12)
    ))
    
    fig.add_trace(go.Bar(
        name='Test',
        x=experiments,
        y=test_values,
        marker_color=test_color,
        text=[f'{val:.2f}' for val in test_values],
        textposition='outside',
        textfont=dict(size=12)
    ))
    
    # Layout aktualisieren
    fig.update_layout(
        title=dict(
            text=f'Ablation Study - {metric_name} ({error_type.replace("_", " ").title()})',
            font=dict(size=18, family="Arial Black", color="black")
        ),
        xaxis_title=dict(
            text='Experiments',
            font=dict(size=14, family="Arial", color="black")
        ),
        yaxis_title=dict(
            text=metric_name,
            font=dict(size=14, family="Arial", color="black")
        ),
        barmode='group',
        height=figsize[1] * 80,  # Umrechnung von figsize
        width=figsize[0] * 80,
        template='plotly_white',
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            font=dict(size=12, family="Arial", color="black")
        ),
        font=dict(family="Arial", color="black"),  # Globale Schriftart
        xaxis=dict(
            tickfont=dict(size=11, family="Arial", color="black")
        ),
        yaxis=dict(
            tickfont=dict(size=12, family="Arial", color="black")
        )
    )
    
    # Feste Y-Achsen Skalierung
    if y_range:
        fig.update_yaxes(range=y_range)
    
    # Plot speichern
    output_path_html = os.path.join(output_dir, f'ablation_{metric_name.lower()}_{error_type}.html')
    output_path_pdf = os.path.join(output_dir, f'ablation_{metric_name.lower()}_{error_type}.pdf')
    
    fig.write_html(output_path_html)
    print(f"Plot gespeichert: {output_path_html}")
    
    # PDF-Export mit Fehlerbehandlung
    try:
        fig.write_image(output_path_pdf, width=figsize[0] * 80, height=figsize[1] * 80, scale=2)
        print(f"Plot gespeichert: {output_path_pdf}")
    except Exception as e:
        print(f"PDF-Export fehlgeschlagen (Chrome/Kaleido Problem): {e}")
        print("Nur HTML-Version wurde gespeichert.")
    
    fig.show()

# Beispiel für die Verwendung:
if __name__ == "__main__":
    try:
        # Feste Skalen für konsistente Darstellung definieren
        fixed_scales = {
            'Accuracy': (0, 1.0)
        }
        
        # Experiment-Reihenfolge definieren (optional)
        #experiment_order = ['standard-architecture', 'only-coarse-module', 'no-coarse-preds', 
        #                   'no-global-features', 'no-poi-vert', 'no-patch-features', 'no-projection']

        #experiment_order = ['standard-architecture1', 'standard-architecture2', 'standard-architecture3', 'standard-architecture4',
        #                    'standard-architecture5', 'standard-architecture6', 'standard-architecture7', 'standard-architecture8', 
        #                     'standard-architecture9', 'no-global-feature1', 'no-global-feature2', 'no-global-feature3', 'no-global-feature4', 
        #                     'no-global-feature5', 'no-global-feature6', 'no-global-feature7', 'no-global-feature8', 'no-global-feature9']
        
        #experiment_order = ['standard-architecture', 'only-coarse-module', 'no-coarse-preds', 'no-global-features', 'no-poi-vert', 'no-patch-features', 'no-projection']

        experiment_order = ['include-com', 'all-pois', 'excel-exclude', 'excel-outliers-exclude', 'subreg-0.5-zoom', 'subreg-2.0-zoom', 'vertseg', 'surface-mask', 'ct-scan', 'neighbors']
        # Test ob das Verzeichnis existiert
        metrics_dir = 'dataloader/combined'
        if not os.path.exists(metrics_dir):
            print(f"Verzeichnis {metrics_dir} existiert nicht!")
            print("Bitte passen Sie den Pfad an.")
        else:
            print(f"Verwende Verzeichnis: {metrics_dir}")
            
            # Alle Plots mit festen Skalen erstellen
            create_ablation_plots(metrics_dir,
                                 fixed_scales=fixed_scales,
                                 experiment_order=experiment_order,
                                 val_color='#1F3E74', 
                                 test_color='#4469AD')
            
            # Neue Error-Type Vergleichsplots erstellen
            create_error_type_comparison_plots(metrics_dir,
                                              experiment_order=experiment_order,
                                              fixed_scales=fixed_scales)
            
            # Oder nur einen spezifischen Plot mit fester Skala
            # create_single_metric_plot('ablation_study_metrics', 
            #                          'Accuracy', 
            #                          'refined_error',
            #                          y_range=(0, 1.0),
            #                          val_color='#1F3E74', 
            #                          test_color='#4469AD')
            
    except Exception as e:
        print(f"Fehler beim Erstellen der Plots: {e}")
        import traceback
        traceback.print_exc()