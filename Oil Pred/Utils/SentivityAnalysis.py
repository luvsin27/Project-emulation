import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from distython import HEOM
#from SALib.sample import morris as ms
#from SALib.analyze import morris as ma
#from SALib.plotting import morris as mp
#from SALib.sample import saltelli as ss
#from SALib.analyze import sobol as sa
#from SALib.plotting import bar as sp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SensitivityAnalysis:
    def __init__(self, model):
        self.model = model

    def global_sensitivity_analysis(self, X, num_samples=100, plot=True):
        sample_data = pd.DataFrame({col: np.random.uniform(low=X[col].min(), high=X[col].max(), size=num_samples)
                                    for col in X.columns})
        predictions = self.model.predict(sample_data)
        feature_importances = sample_data.apply(lambda x: np.var(self.model.predict(sample_data.assign(**{x.name: x}))), axis=0)

        if plot:
            feature_importances_sorted = feature_importances.sort_values(ascending=False)
            plt.figure(figsize=(10, 6))
            feature_importances_sorted.plot(kind='bar')
            plt.title('Global Sensitivity Analysis - Feature Importances')
            plt.ylabel('Variance in Model Output')
            plt.xlabel('Features')
            plt.tight_layout()
            plt.show()

        return feature_importances

    def local_sensitivity_analysis(self, X, feature, feature_range, plot=True):
        predictions = []

        for value in feature_range:
            X_temp = X.copy()
            X_temp[feature] = value
            predictions.append(self.model.predict(X_temp).mean())

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(feature_range, predictions, marker='o')
            plt.title(f"Sensitivity Analysis of {feature}")
            plt.xlabel(feature)
            plt.ylabel('Average Predicted Output')
            plt.grid(True)
            plt.show()

        return feature_range, predictions
