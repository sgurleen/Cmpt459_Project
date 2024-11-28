import numpy as np
import matplotlib.pyplot as plt

# Cross-validation scores
cv_scores = [0.82963315, 0.84672919, 0.88483457, 0.88388476, 0.88800063]

# Compute the mean score
mean_score = np.mean(cv_scores)

# Plot the cross-validation scores
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='-', label='Cross-Validation Score', color='b')
plt.axhline(y=mean_score, color='r', linestyle='--', label=f'Mean Score ({mean_score:.4f})')

# Add labels, title, and legend
plt.title('Cross-Validation Scores', fontsize=14)
plt.xlabel('Fold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xticks(range(1, len(cv_scores) + 1))
plt.legend(fontsize=12)
plt.grid(alpha=0.5)

# Show the plot
plt.tight_layout()
plt.show()
