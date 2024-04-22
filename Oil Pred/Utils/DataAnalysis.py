import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import ccf

class WellDataProcessor:
    def __init__(self, filepath):
        self.data = self.prepare_data(filepath)
        self.oil_data, self.inj_data = self.split_data_by_well_type(self.data)

    @staticmethod
    def prepare_data(filepath):
        data = pd.read_csv(filepath, parse_dates=['DATE'])
        data['DATE'] = pd.to_datetime(data['DATE'])
        return data

    @staticmethod
    def split_data_by_well_type(data):
        oil_data = data[data['WELL_TYPE'] == 'OP'].sort_values('DATE').set_index('DATE')
        inj_data = data[data['WELL_TYPE'] == 'WI'].sort_values('DATE').set_index('DATE')
        return oil_data, inj_data

    @staticmethod
    def preprocess_data(data, well_type, well_name):
        well_data = data[(data['WELLNAME'] == well_name) & (data['WELL_TYPE'] == well_type)]
        all_dates = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
        return well_data.reindex(all_dates).ffill().bfill()

    def visualize_well_data(self):
        self.plot_well_data(self.data)

    def analyze_wells(self, production_wells, injection_wells):
        for well in production_wells:
            for inj_well in injection_wells:
                well_data = self.preprocess_data(self.oil_data, 'OP', well)
                inj_well_data = self.preprocess_data(self.inj_data, 'WI', inj_well)
                self.plot_cross_correlation(well_data, inj_well_data, well, inj_well)

    def remove_outliers(self, column):
        return self._remove_outliers(self.oil_data, column)

    @staticmethod
    def _remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    def plot_well_data(self, data):
        # Create a copy to avoid modifying the original dataframe
        data = self.data.copy()
        data['DATE'] = pd.to_datetime(data['DATE'])
        data.set_index('DATE', inplace=True)

        # Determine the number of unique wells and initialize subplots
        unique_wells = data['WELLNAME'].nunique()
        fig, axs = plt.subplots(unique_wells, figsize=(15, 20), sharex=True)
        axs = axs.ravel() if unique_wells > 1 else [axs]

        for ax, (well_name, well_data) in zip(axs, data.groupby('WELLNAME')):
            numeric_cols = well_data.select_dtypes(include=[np.number])
            monthly_data = numeric_cols.resample('M').mean()

            ax.plot(monthly_data.index, monthly_data['OIL_PROD'], label='Oil Production', color='green')
            ax.plot(monthly_data.index, monthly_data['WAT_PROD'], label='Water Production', color='blue')

            ax2 = ax.twinx()
            ax2.plot(monthly_data.index, monthly_data['WAT_INJ'], label='Water Injection', linestyle='--', color='red')
            ax2.set_ylabel('Water Injection Volume', color='red')
            ax2.tick_params(axis='y', colors='red')

            ax.set_title(f'Well {well_name} - Production and Injection')
            ax.set_ylabel('Production Volume')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_cross_correlation(well_data, inj_data, well, inj_well):
        series = well_data['OIL_PROD']
        inj_series = inj_data['WAT_INJ']
        ccf_values = ccf(series, inj_series, adjusted=True)
        max_corr_lag = np.argmax(np.abs(ccf_values)) - len(ccf_values) // 2

        plt.figure(figsize=(10, 5))
        plt.stem(range(len(ccf_values)), ccf_values)
        plt.title(f'Cross-Correlation between Prod Well {well} and Inj Well {inj_well}')
        plt.xlabel('Lag')
        plt.ylabel('Cross-correlation')
        plt.axvline(x=max_corr_lag, color='red', linestyle='--', label=f'Max Corr at Lag = {max_corr_lag}')
        plt.legend()
        plt.show()
        print(f"Between Prod Well {well} and Inj Well {inj_well}, the lag with highest cross-correlation is: {max_corr_lag} days\n")
    
    def plot_implot_vs_production(self, filtered_data, x_col, y_col, title):
        sns.lmplot(x=x_col, y=y_col, data=filtered_data, height=7, aspect=2)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(title)
        plt.show()

    def plot_choke_management(self, oil_data):
        fig, ax1 = plt.subplots(figsize=(15, 7))
        ax2 = ax1.twinx()
        ax1.plot(oil_data.index, oil_data['CHOKE_1_PERC'], 'g-', label='Choke 1 Openness (%)')
        ax2.plot(oil_data.index, oil_data['CHOKE_2_PERC'], 'b-', label='Choke 2 Openness (%)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Choke 1 Openness (%)', color='g')
        ax2.set_ylabel('Choke 2 Openness (%)', color='b')
        plt.title('Choke Openness Over Time for Oil-Producing Wells')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.show()

    def plot_pressure_management(self, oil_data):
        fig, ax1 = plt.subplots(figsize=(15, 7))
        ax2 = ax1.twinx()
        ax1.plot(oil_data.index, oil_data['WHP'], 'g-', label='WHP')
        ax2.plot(oil_data.index, oil_data['BHP'], 'b-', label='BHP')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('WHP', color='g')
        ax2.set_ylabel('BHP', color='b')
        plt.title('Pressure Management Over Time for Oil-Producing Wells')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.show()    
        
    def calculate_and_plot_cumulative(self):
        oil_data = self.data[self.data['WELL_TYPE'] == 'OP'].copy()
        inj_data = self.data[self.data['WELL_TYPE'] == 'WI'].copy()

        if 'DATE' not in oil_data.columns:
            oil_data = oil_data.reset_index()
        if 'DATE' not in inj_data.columns:
            inj_data = inj_data.reset_index()

        oil_data.sort_values('DATE', inplace=True)
        inj_data.sort_values('DATE', inplace=True)

        oil_data['Cumulative_OIL_PROD'] = oil_data.groupby('WELLNAME')['OIL_PROD'].cumsum()
        oil_data['Cumulative_WAT_PROD'] = oil_data.groupby('WELLNAME')['WAT_PROD'].cumsum()
        inj_data['Cumulative_WAT_INJ'] = inj_data.groupby('WELLNAME')['WAT_INJ'].cumsum()
        plt.figure(figsize=(15, 7))
        for well_name in oil_data['WELLNAME'].unique():
            well_data = oil_data[oil_data['WELLNAME'] == well_name]
            plt.plot(well_data['DATE'], well_data['Cumulative_OIL_PROD'], label=f'Cumulative Oil Prod {well_name}')
            plt.plot(well_data['DATE'], well_data['Cumulative_WAT_PROD'], label=f'Cumulative Water Prod {well_name}')

        for well_name in inj_data['WELLNAME'].unique():
            well_data = inj_data[inj_data['WELLNAME'] == well_name]
            plt.plot(well_data['DATE'], well_data['Cumulative_WAT_INJ'], label=f'Cumulative Water Injection {well_name}')

        plt.xlabel('Date')
        plt.ylabel('Cumulative Volume')
        plt.title('Cumulative Oil Production vs. Cumulative Water Injection')
        plt.legend()
        plt.show()
        
    def plot_correlation_heatmap(self):
        correlation_data = self.data[self.data['WELLNAME'].isin([5351, 5599])][['BHP', 'WHP', 'CHOKE_1_PERC', 'CHOKE_2_PERC', 'OIL_PROD', 'GAS_PROD', 'WAT_PROD']]
        correlation_matrix = correlation_data.corr('spearman')
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix Heatmap')
        plt.show()    
        
    def decompose_and_plot_seasonal(self, column_name, well_type):
        
        data_resampled = self.data[self.data['WELL_TYPE']== well_type].set_index('DATE')[column_name].resample('M').mean().bfill()
        decomposition = seasonal_decompose(data_resampled, model='additive')
        decomposition.seasonal.plot(title='Seasonal Component of ' + column_name + ' Volumes')