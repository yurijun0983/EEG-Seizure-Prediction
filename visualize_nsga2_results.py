"""
Visualize NSGA-II Patient Selection Optimization Results

This script loads and visualizes the results from NSGA-II multi-objective optimization:
1. Pareto Front visualization (3D scatter plot)
2. Evolution history (F1 score over generations)
3. Patient selection frequency analysis
4. Trade-off analysis between objectives
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse


def load_results(results_path):
    """Load NSGA-II results from JSON file"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def plot_evolution_history(results, save_dir):
    """Plot evolution history"""
    history = results['history']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Best F1 per generation
    axes[0, 0].plot(history['best_f1_per_gen'], marker='o', linewidth=2, markersize=6)
    axes[0, 0].set_title('Best F1 Score per Generation', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Best F1 Score')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Average F1 per generation
    axes[0, 1].plot(history['avg_f1_per_gen'], marker='s', color='orange', linewidth=2, markersize=6)
    axes[0, 1].set_title('Average F1 Score per Generation', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Average F1 Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Population diversity
    axes[1, 0].plot(history['diversity_per_gen'], marker='^', color='green', linewidth=2, markersize=6)
    axes[1, 0].set_title('Population Diversity per Generation', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Diversity (Std of Genes)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Pareto front size
    pareto_sizes = [len(pf) for pf in history['pareto_front_per_gen']]
    axes[1, 1].plot(pareto_sizes, marker='d', color='red', linewidth=2, markersize=6)
    axes[1, 1].set_title('Pareto Front Size per Generation', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Number of Solutions')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evolution_history.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(save_dir, 'evolution_history.png')}")
    plt.close()


def plot_pareto_front_3d(results, save_dir):
    """Plot 3D Pareto front"""
    pareto_front = results['pareto_front']
    
    # Extract objectives
    f1_scores = [sol['objectives']['f1_score'] for sol in pareto_front]
    n_patients = [sol['objectives']['n_patients'] for sol in pareto_front]
    imbalances = [sol['objectives']['imbalance'] for sol in pareto_front]
    
    # 3D scatter plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(n_patients, imbalances, f1_scores, 
                        c=f1_scores, cmap='viridis', s=200, alpha=0.8, edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel('Number of Patients →', fontsize=12, fontweight='bold')
    ax.set_ylabel('Imbalance Penalty →', fontsize=12, fontweight='bold')
    ax.set_zlabel('F1 Score →', fontsize=12, fontweight='bold')
    ax.set_title('NSGA-II Pareto Front (3D)', fontsize=16, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('F1 Score', fontsize=12, fontweight='bold')
    
    # Mark recommended solution
    recommended = results['recommended_solution']
    ax.scatter([recommended['objectives']['n_patients']], 
              [recommended['objectives']['imbalance']], 
              [recommended['objectives']['f1_score']], 
              c='red', s=400, marker='*', edgecolors='black', linewidth=2, label='Recommended')
    
    ax.legend(fontsize=12)
    plt.savefig(os.path.join(save_dir, 'pareto_front_3d.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(save_dir, 'pareto_front_3d.png')}")
    plt.close()


def plot_pareto_front_2d(results, save_dir):
    """Plot 2D projections of Pareto front"""
    pareto_front = results['pareto_front']
    
    # Extract objectives
    f1_scores = [sol['objectives']['f1_score'] for sol in pareto_front]
    n_patients = [sol['objectives']['n_patients'] for sol in pareto_front]
    imbalances = [sol['objectives']['imbalance'] for sol in pareto_front]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # F1 vs N_patients
    axes[0].scatter(n_patients, f1_scores, c=f1_scores, cmap='viridis', s=150, alpha=0.7, edgecolors='black')
    recommended = results['recommended_solution']
    axes[0].scatter([recommended['objectives']['n_patients']], 
                   [recommended['objectives']['f1_score']], 
                   c='red', s=300, marker='*', edgecolors='black', linewidth=2, label='Recommended', zorder=5)
    axes[0].set_xlabel('Number of Patients', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[0].set_title('F1 vs Number of Patients', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # F1 vs Imbalance
    axes[1].scatter(imbalances, f1_scores, c=f1_scores, cmap='plasma', s=150, alpha=0.7, edgecolors='black')
    axes[1].scatter([recommended['objectives']['imbalance']], 
                   [recommended['objectives']['f1_score']], 
                   c='red', s=300, marker='*', edgecolors='black', linewidth=2, label='Recommended', zorder=5)
    axes[1].set_xlabel('Imbalance Penalty', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[1].set_title('F1 vs Imbalance', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # N_patients vs Imbalance
    axes[2].scatter(n_patients, imbalances, c=f1_scores, cmap='coolwarm', s=150, alpha=0.7, edgecolors='black')
    axes[2].scatter([recommended['objectives']['n_patients']], 
                   [recommended['objectives']['imbalance']], 
                   c='red', s=300, marker='*', edgecolors='black', linewidth=2, label='Recommended', zorder=5)
    axes[2].set_xlabel('Number of Patients', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Imbalance Penalty', fontsize=12, fontweight='bold')
    axes[2].set_title('Patients vs Imbalance', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pareto_front_2d.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(save_dir, 'pareto_front_2d.png')}")
    plt.close()


def plot_patient_frequency(results, save_dir):
    """Analyze patient selection frequency in Pareto front"""
    pareto_front = results['pareto_front']
    source_patients = results['source_patients']
    
    # Count frequency of each patient
    patient_counts = {p: 0 for p in source_patients}
    for sol in pareto_front:
        for patient in sol['selected_patients']:
            patient_counts[patient] += 1
    
    # Sort by frequency
    sorted_patients = sorted(patient_counts.items(), key=lambda x: x[1], reverse=True)
    patients = [p[0] for p in sorted_patients]
    counts = [p[1] / len(pareto_front) * 100 for p in sorted_patients]  # Convert to percentage
    
    # Bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(patients, counts, color='steelblue', edgecolor='black', linewidth=1.5)
    
    # Highlight recommended patients
    recommended_patients = results['recommended_solution']['selected_patients']
    for i, patient in enumerate(patients):
        if patient in recommended_patients:
            bars[i].set_color('orange')
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(2.5)
    
    ax.set_xlabel('Source Patients', fontsize=12, fontweight='bold')
    ax.set_ylabel('Selection Frequency (%)', fontsize=12, fontweight='bold')
    ax.set_title('Patient Selection Frequency in Pareto Front', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    
    # Add value labels
    for i, (p, c) in enumerate(zip(patients, counts)):
        ax.text(i, c + 2, f'{c:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', edgecolor='black', label='Non-recommended'),
        Patch(facecolor='orange', edgecolor='red', label='Recommended')
    ]
    ax.legend(handles=legend_elements, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'patient_frequency.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(save_dir, 'patient_frequency.png')}")
    plt.close()


def print_summary(results):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("NSGA-II Optimization Summary")
    print("="*70)
    
    print(f"\nTarget Patient: {results['target_patient']}")
    print(f"Source Patients: {', '.join(results['source_patients'])}")
    
    print(f"\nAlgorithm Parameters:")
    params = results['parameters']
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    print(f"\nPareto Front: {len(results['pareto_front'])} solutions")
    
    print(f"\nRecommended Solution:")
    rec = results['recommended_solution']
    print(f"  Selected Patients: {', '.join(rec['selected_patients'])}")
    print(f"  Number of Patients: {rec['objectives']['n_patients']}")
    print(f"  F1 Score: {rec['objectives']['f1_score']:.4f}")
    print(f"  Imbalance Penalty: {rec['objectives']['imbalance']:.4f}")
    
    print(f"\nTop 5 Pareto Solutions (sorted by F1):")
    sorted_pf = sorted(results['pareto_front'], key=lambda x: x['objectives']['f1_score'], reverse=True)
    for i, sol in enumerate(sorted_pf[:5]):
        print(f"  #{i+1}: F1={sol['objectives']['f1_score']:.4f}, "
              f"N={sol['objectives']['n_patients']}, "
              f"Imb={sol['objectives']['imbalance']:.4f}, "
              f"Patients={sol['selected_patients']}")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize NSGA-II results')
    parser.add_argument('--results_path', type=str, 
                       default='outputs_nsga2_pn14/nsga2_patient_selection_results.json',
                       help='Path to NSGA-II results JSON file')
    parser.add_argument('--output_dir', type=str, 
                       default='outputs_nsga2_pn14/visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Load results
    print(f"\nLoading results from: {args.results_path}")
    results = load_results(args.results_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print summary
    print_summary(results)
    
    # Generate visualizations
    print("Generating visualizations...")
    print("-" * 70)
    
    plot_evolution_history(results, args.output_dir)
    plot_pareto_front_3d(results, args.output_dir)
    plot_pareto_front_2d(results, args.output_dir)
    plot_patient_frequency(results, args.output_dir)
    
    print("-" * 70)
    print(f"\nAll visualizations saved to: {args.output_dir}")
    print("Done!\n")


if __name__ == "__main__":
    main()
