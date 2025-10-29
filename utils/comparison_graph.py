import matplotlib.pyplot as plt
import numpy as np

def create_comparison_radar_chart():
    # Labels for the radar chart axes
    labels = [
        'Retenção de Conhecimento',
        'Tempo de Adaptação',
        'Acurácia em Ambientes Dinâmicos',
        'Capacidade de Reorganização',
        'Aprendizado e Escalabilidade',
        'Custo Computacional'
    ]

    # Number of variables
    num_vars = len(labels)

    # Scores for SEONN (theoretical advantages)
    # Scale: 0-10 (higher is better, except for Cost where lower is better)
    # SEONN scores are based on its design principles and potential.
    seonn_scores = [
        9,  # Retenção de Conhecimento (designed for continual learning)
        8,  # Tempo de Adaptação (dynamic structure allows faster adaptation)
        9,  # Acurácia em Ambientes Dinâmicos (adapts to changing data)
        10, # Capacidade de Reorganização (core feature)
        7,  # Aprendizado e Escalabilidade (potential for efficient learning, but early stage)
        4   # Custo Computacional (currently higher due to dynamic nature, but potential for optimization)
    ]

    # Scores for Traditional ANNs
    # ANNs are highly optimized but struggle with dynamic aspects.
    ann_scores = [
        4,  # Retenção de Conhecimento (catastrophic forgetting)
        5,  # Tempo de Adaptação (requires retraining/fine-tuning)
        6,  # Acurácia em Ambientes Dinâmicos (can degrade without retraining)
        2,  # Capacidade de Reorganização (fixed architecture)
        9,  # Aprendizado e Escalabilidade (highly optimized for static tasks)
        8   # Custo Computacional (highly optimized, especially on GPU, for static tasks)
    ]

    # Adjust Cost score so higher value means better (lower cost)
    # For visualization, we invert the cost score for ANNs and SEONN
    # Max possible cost score is 10, so 10 - actual_score
    seonn_scores_adjusted = list(seonn_scores)
    ann_scores_adjusted = list(ann_scores)

    seonn_scores_adjusted[5] = 10 - seonn_scores[5] # Invert SEONN Cost
    ann_scores_adjusted[5] = 10 - ann_scores[5]     # Invert ANN Cost

    # Add the first value to the end to close the circle
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    seonn_values = seonn_scores_adjusted + seonn_scores_adjusted[:1]
    ann_values = ann_scores_adjusted + ann_scores_adjusted[:1]

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})

    # Plot SEONN
    ax.plot(angles, seonn_values, linewidth=2, linestyle='solid', label='SEONN', color='blue')
    ax.fill(angles, seonn_values, color='blue', alpha=0.25)

    # Plot Traditional ANNs
    ax.plot(angles, ann_values, linewidth=2, linestyle='solid', label='Redes Neurais Artificiais (RNAs)', color='red')
    ax.fill(angles, ann_values, color='red', alpha=0.25)

    # Set labels for each axis
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rticks([2, 4, 6, 8, 10]) # Radial ticks
    ax.set_rlabel_position(0) # Position of the r-label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 10) # Set y-axis limits

    # Add a legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.title('Comparativo: SEONN vs. Redes Neurais Artificiais (RNAs)', size=16, color='black', y=1.1)
    plt.show()

if __name__ == "__main__":
    create_comparison_radar_chart()
