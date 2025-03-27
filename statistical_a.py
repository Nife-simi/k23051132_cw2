from scipy.stats import ttest_rel

accuracies = [38.70, 26.0, 48.4]  # SSE, Fully Supervised, Semi-Supervised

accuracies_random = [19.4, 14.0, 23.3]  # SSE, Fully Supervised, Semi-Supervised

# Paired t-test
t_stat, p_value = ttest_rel(accuracies, accuracies_random)

print(f"T-Statistic: {t_stat:.3f}, P-value: {p_value:.3f}")

# Interpretation
if p_value < 0.05:
    print("Significant difference")
else:
    print("No significant difference")
