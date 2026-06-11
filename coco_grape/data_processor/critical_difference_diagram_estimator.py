import scikit_posthocs as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

#pip install git+https://github.com/maximtrp/scikit-posthocs.git

class CriticalDifferenceDiagramEstimator:
    def __init__(self, data_estimators, score_func=accuracy_score, n_repetitions=5, use_predict_proba=False, verbose=False):
        """
        Initializes the estimator.

        Parameters:
        - data_estimators: List of estimator objects to evaluate.
        - n_repetitions: Number of repetitions for cross-validation.
        - verbose: If True, displays progress bars.
        """
        self.data_estimators = data_estimators
        self.score_func = score_func
        self.n_repetitions = n_repetitions
        self.use_predict_proba = use_predict_proba
        self.verbose = verbose

    def fit(self, graphs, targets):
        """
        Fits each estimator and collects evaluation scores.

        Parameters:
        - graphs: Input data (e.g., feature matrices or graph representations).
        - targets: Target labels.

        Returns:
        - self: The instance itself.
        """
        # Initialize an empty DataFrame to store results
        self.empirical_results = pd.DataFrame()

        # Optionally wrap the outer loop with tqdm for a progress bar
        estimators = enumerate(self.data_estimators)
        if self.verbose:
            estimators = tqdm(
                estimators,
                total=len(self.data_estimators),
                desc="Estimators"
            )

        # Iterate over each estimator
        for i, data_estimator in estimators:
            # Optionally wrap the inner loop with tqdm for a progress bar
            repetitions = range(self.n_repetitions)
            if self.verbose:
                repetitions = tqdm(
                    repetitions,
                    desc=f"Repetitions (Estimator {i})",
                    leave=False
                )

            # Perform multiple repetitions for score estimate
            scores = [self.predict_score(data_estimator, graphs, targets, score_func=self.score_func, train_size=0.7, random_state=it) for it in repetitions]
                
            # Store the scores for the current estimator
            self.empirical_results[f'est_{i}'] = scores

        return self
    
    def predict_score(self, data_estimator, graphs, targets, score_func, train_size, random_state):
        # Split the data into training and testing sets
        train_graphs, test_graphs, train_targets, test_targets = train_test_split(
            graphs, targets, train_size=train_size, random_state=random_state
        )

        # Train the estimator on the training data
        data_estimator.fit(train_graphs, train_targets)

        # Predict on the test data
        if self.use_predict_proba:
            preds = data_estimator.predict_proba(test_graphs)[:,-1]
        else:
            preds = data_estimator.predict(test_graphs)

        # Calculate the score
        score = score_func(test_targets, preds)
        return score

    def get_dataframe(self):
        """
        Returns the DataFrame containing the empirical results.

        Returns:
        - DataFrame with scores for each estimator.
        """
        return self.empirical_results
    
    def get_plot(self, filename=None):
        """
        Generates and displays the critical difference diagram.
        If a filename is provided, saves the plot to the specified file.

        Parameters:
        - filename: Optional. If provided and is a string, saves the plot to this filename.
        """
        # Copy the empirical results to avoid modifying the original DataFrame
        data = self.empirical_results.copy()

        # Add an experiment ID column for identification
        data['experiment_id'] = range(len(data))

        # Melt the DataFrame to long format suitable for statistical analysis
        data_melted = pd.melt(
            data,
            id_vars=['experiment_id'],
            var_name='variable',
            value_name='value'
        )

        # Calculate average ranks for each estimator across experiments
        avg_rank = data_melted.groupby('experiment_id')['value'].rank(pct=True)
        avg_rank = avg_rank.groupby(data_melted['variable']).mean()

        # Perform the Conover post-hoc test after the Friedman test
        test_results = sp.posthoc_conover_friedman(
            data_melted,
            melted=True,
            block_col='experiment_id',
            group_col='variable',
            y_col='value'
        )

        # Plot the critical difference diagram
        plt.figure(figsize=(5, 1.5), dpi=70)
        plt.title('Critical Difference Diagram of Average Score Ranks')
        sp.critical_difference_diagram(avg_rank, test_results, elbow_props={'color': 'k', 'linewidth': 1}, marker_props={'marker': 'o', 's': 12,}, crossbar_props={'linewidth': 4})
        

        # If a filename is provided, save the plot to the file
        if isinstance(filename, str):
            plt.savefig(filename, bbox_inches='tight')
            plt.close()  # Close the figure to free up memory
        else:
            plt.show()
            
