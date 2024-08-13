import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Define capability benchmarks
cap_names = ["logiqa", "piqa", "hellaswag", "winogrande", 'superglue_copa', "medqa_4options", "arc_challenge", "mmlu", "minerva_math", "lambada_openai", "gsm8k", "bbh"]

def load_data():
    """
    Load data from CSV files and return as pandas DataFrames.
    """
    base_model_df = pd.read_csv('data/E_base_model.csv', index_col='Unnamed: 0')
    chat_model_df = pd.read_csv('data/E_chat_model.csv', index_col='Unnamed: 0')
    evals_df = pd.read_csv('data/evals_info.csv', index_col='Unnamed: 0')
    return base_model_df, chat_model_df, evals_df

def find_nans(model_df):
    """
    Identifies and prints the locations of NaNs in the dataframe.
    """
    df_filtered = model_df.drop(columns=["model_size", "FLOP", "name", "type"])

    missing_fraction = df_filtered.isnull().mean().mean()
    print(f"Fraction missing data: {missing_fraction}")

    missing_data = df_filtered.isnull()
    missing_indices = missing_data.stack()[missing_data.stack()].index

    for row, col in missing_indices:
        print(f"Missing data at row {row}, column {col}")

# Load data and print missing values
base_model_df, chat_model_df, evals_df = load_data()
find_nans(base_model_df)
find_nans(chat_model_df)

##### RUN ANALYSIS ON BASE AND CHAT MODELS #####

def run_analysis(model_df, evals_df, cap_names, label, correlation_type = "spearman"):
    """
    Runs normalization, computes and plots correlation matrix, performs PCA, and prints analysis results. Returns modified evals_df and model_df, and eigenvalues and correlation matrix from PCA.

    We compute the correlation of each safety benchmark individually, dropping the rows that contains NaNs for each safety benchmark (if any).
    """

    # Normalize data
    normalized_df = model_df.drop(columns=["model_size", "FLOP", "name", "type"])
    normalized_df = normalized_df.dropna(axis=1, how='all')
    normalized_df = (normalized_df - normalized_df.mean()) / normalized_df.std(ddof=0)
    normalized_df = normalized_df.dropna(axis=1, how='all') # In case one column has a std dev of zero and became NaN during normalization

    df_cap = normalized_df[cap_names].copy().dropna() # capabilities evals dataframe
    safety_names = [col for col in normalized_df.columns if col not in cap_names]
    df_safety = normalized_df[safety_names].copy() # "safety" evals dataframe

    # Compute correlation matrix
    cap_matrix = df_cap.corr(method=correlation_type).to_numpy()

    # Plot correlation matrix
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(cap_matrix, cmap="viridis", vmin=-0.5, vmax=1.1)
    for i in range(len(cap_names)):
        for j in range(len(cap_names)):
            ax.text(j, i, f"{(cap_matrix[i, j]*100):.0f}", ha="center", va="center", color="w", fontsize=20)
    ax.set_xticks(np.arange(len(cap_names)))
    ax.set_yticks(np.arange(len(cap_names)))
    ax.set_xticklabels([evals_df.loc[name]["name"] for name in cap_names], fontsize = 18)
    ax.set_yticklabels([evals_df.loc[name]["name"] for name in cap_names], fontsize = 18)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(label, fontsize=35, pad=15)
    fig.tight_layout()
    plt.show()

    # Compute statistics of correlations between two capabilities benchmarks
    triu_indices = np.triu_indices_from(cap_matrix, k=1)
    upper_tri_values = cap_matrix[triu_indices]
    mean_correlation = np.mean(upper_tri_values)
    std_dev_correlation = np.std(upper_tri_values)

    # Compute capabilities scores
    eigenvals, eigenvecs = np.linalg.eigh(cap_matrix)
    pc = eigenvecs[:, -1]
    pc = pc if np.abs(pc).max() == pc.max() else -pc
    model_cap_score = df_cap.to_numpy() @ pc

    # Update dataframes with capabilities scores
    df_safety["cap_score"] = model_cap_score
    df_cap["cap_score"] = model_cap_score
    evals_df_copy = evals_df.copy()

    # Compute correlations for each evaluation and update dataframes
    for safety_name in safety_names:
        safety_task_df = df_safety[[safety_name]].dropna() # Drop NaNs
        score_df = df_safety[["cap_score"]].loc[safety_task_df.index]
        reduced_df = pd.concat([safety_task_df, score_df], axis=1) # Concatenate with capabilities scores
        corr_value = reduced_df.corr(method=correlation_type).to_numpy()[0, 1] # 2x2 matrix where we take entry (0,1)
        evals_df_copy.loc[safety_name, f"{label}_{correlation_type}_correlations"] = corr_value # populate correlation column

    for cap_name in cap_names:
        cap_task_df = df_cap[[cap_name]].dropna()
        score_df = df_cap[["cap_score"]].loc[cap_task_df.index]
        reduced_df = pd.concat([cap_task_df, score_df], axis=1)
        corr_value = reduced_df.corr(method=correlation_type).to_numpy()[0, 1]
        evals_df_copy.loc[cap_name, f"{label}_{correlation_type}_correlations"] = corr_value
    
    # Calculate total variance explained
    variance_explained_pc1 = eigenvals[-1] / np.sum(eigenvals)

    # Print analysis results
    print(f"Mean correlation between two capabilities benchmarks: {mean_correlation}")
    print(f"Standard deviation of correlation between two capabilities benchmarks: {std_dev_correlation}")
    print(f"Total capabilities variance explained from PC1: {variance_explained_pc1*100:.1f}")

    print("\nCAPABILITIES CORRELATIONS:")
    for cap_name in cap_names:
        print(cap_name, evals_df_copy.loc[cap_name, f"{label}_{correlation_type}_correlations"].round(3) * 100)

    print("\nSAFETY CORRELATIONS:")
    for safety_name in safety_names:
        print(safety_name, evals_df_copy.loc[safety_name, f"{label}_{correlation_type}_correlations"].round(3) * 100)

    print("\nMODEL G-SCORES")
    model_cap_dict = dict(sorted(list(zip(model_df.index, model_cap_score)), key=lambda item: item[1]))
    for k, v in model_cap_dict.items():
        print(f"{k}, {v:.2f}")

    # Update model dataframe with scores
    model_df_copy = model_df.copy()
    model_df_copy["score"] = model_cap_score

    return evals_df_copy, model_df_copy, eigenvals, cap_matrix

evals_df, base_model_df, base_eigenvals, base_cap_matrix = run_analysis(base_model_df, evals_df, cap_names, "Base", "spearman")
evals_df, chat_model_df, chat_eigenvals, chat_cap_matrix = run_analysis(chat_model_df, evals_df, cap_names, "Chat", "spearman")

##### PLOTTING DEMOS #####

def plot_capabilities_score(model_df):
    plt.figure(figsize=(10, 7))
    ax = sns.scatterplot(
        data=model_df.sort_values('score'),
        y='score',
        x=range(len(model_df)),
        color='royalblue',
        s=80
    )
    ax.set_ylabel('Capabilities Score', fontsize=20)
    ax.tick_params(axis='x', labelsize=14, rotation=90)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(range(len(model_df)))
    ax.set_xticklabels(model_df.sort_values('score').index)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_eigenvalues(base_eigenvals, chat_eigenvals):
    plt.plot(base_eigenvals[::-1], label='Base Eigenvalues', color='blue')
    plt.plot(chat_eigenvals[::-1], label='Chat Eigenvalues', color='orange')
    plt.title('Eigenvalues of the Capability Correlation Matrix')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_safety_vs_capabilities(model_df, x_axis, benchmark_name, class_type, y_label, x_label, color, title, xlim):
    gscores = model_df[x_axis].to_numpy()
    fig, ax = plt.subplots(1, 1)
    benchmark_scores = model_df[benchmark_name].to_numpy()
    ax.scatter(gscores, benchmark_scores, color=color, s=10)
    sns.regplot(x=x_axis, y=benchmark_name, data=model_df, ax=ax, color=color, scatter_kws={'s': 10}, label=class_type)
    ax.grid(linestyle='--')
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_xlabel(x_label, fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    tick_values = ax.get_xticks()

    if any(tick % 1 != 0 for tick in tick_values):
        tick_labels = [int(tick * 100) for tick in tick_values]
        ax.set_xticks(tick_values)
        ax.set_xticklabels(tick_labels)
    
    ax.set_xlim([min(gscores)-xlim, max(gscores)+xlim])
    ax.set_title(title, fontsize=20)
    fig.tight_layout()
    plt.show()

    pearson_corr, _ = stats.pearsonr(gscores, benchmark_scores)
    spearman_corr, _ = stats.spearmanr(gscores, benchmark_scores)

    print(f"Pearson Correlation: {pearson_corr}")
    print(f"Spearman Correlation: {spearman_corr}")

plot_capabilities_score(chat_model_df)
plot_eigenvalues(base_eigenvals, chat_eigenvals)
plot_safety_vs_capabilities(
    chat_model_df, 
    x_axis="mmlu", 
    benchmark_name="rmsce_mmlu", 
    class_type="chat", 
    y_label="1 - RMS Calibration Error (↑)", 
    x_label="MMLU", 
    color="green", 
    title="RMSCE MMLU", 
    xlim=0.05
)
plot_safety_vs_capabilities(
    chat_model_df, 
    x_axis="score", 
    benchmark_name="truthfulqa_mc1", 
    class_type="chat", 
    y_label="Accuracy (↑)", 
    x_label="Capabilities Score", 
    color="red", 
    title="TruthfulQA MC1", 
    xlim=1
)
