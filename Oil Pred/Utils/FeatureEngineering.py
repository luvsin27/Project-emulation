import numpy as np
import pandas as pd

class FeatureTransformer:
    def __init__(self):
        pass

    def compute_pressure_features(self, df):
        if 'BHP_CHANGE_RATE' not in df.columns:
            df['BHP_CHANGE_RATE'] = 0.0
        if 'WHP_Change_Rate' not in df.columns:
            df['WHP_Change_Rate'] = 0.0

        if not np.issubdtype(df['DATE'].dtype, np.datetime64):
            df['DATE'] = pd.to_datetime(df['DATE'])

        for well_name in df['WELLNAME'].unique():
            well_filter = (df['WELLNAME'] == well_name) & (df['WELL_TYPE'] == 'OP')
            df.loc[well_filter, 'BHP_CHANGE_RATE'] = df.loc[well_filter, 'BHP'].diff(30) / 30
            df.loc[well_filter, 'WHP_Change_Rate'] = df.loc[well_filter, 'WHP'].diff(30) / 30

        df['BHP_CHANGE_RATE'].fillna(0, inplace=True)
        df['WHP_Change_Rate'].fillna(0, inplace=True)
        return df

    def compute_choke_features(self, df):
        for col in ['NORM_CHOKE_1', 'NORM_CHOKE_2', 'CHOKE_INTERACTION']:
            if col not in df.columns:
                df[col] = 0.0

        max_choke_1 = df['CHOKE_1_PERC'].max()
        max_choke_2 = df['CHOKE_2_PERC'].max()
        df['NORM_CHOKE_1'] = df['CHOKE_1_PERC'] / max_choke_1 if max_choke_1 else 0
        df['NORM_CHOKE_2'] = df['CHOKE_2_PERC'] / max_choke_2 if max_choke_2 else 0
        df['CHOKE_INTERACTION'] = df['NORM_CHOKE_1'] * df['NORM_CHOKE_2']
        return df

    def compute_cumulative_changes(self, df):
        if 'CUMULATIVE_BHP_CHANGE' not in df.columns:
            df['CUMULATIVE_BHP_CHANGE'] = 0.0
        if 'CUMULATIVE_WHP_CHANGE' not in df.columns:
            df['CUMULATIVE_WHP_CHANGE'] = 0.0

        for well_name in df['WELLNAME'].unique():
            well_filter = (df['WELLNAME'] == well_name) & (df['WELL_TYPE'] == 'OP')
            df.loc[well_filter, 'CUMULATIVE_BHP_CHANGE'] = df.loc[well_filter, 'BHP'].diff(30).cumsum()
            df.loc[well_filter, 'CUMULATIVE_WHP_CHANGE'] = df.loc[well_filter, 'WHP'].diff(30).cumsum()

        df['CUMULATIVE_BHP_CHANGE'].fillna(0, inplace=True)
        df['CUMULATIVE_WHP_CHANGE'].fillna(0, inplace=True)
        return df

    def compute_injection_features(self, df):
        injection_cols = ['INJ_VOLUME_LAG_542D', 'INJ_VOLUME_LAG_522D', 'INJ_RATE_CHANGE', 
                          'INJ_RATE_CHANGE_LAG_542D', 'INJ_RATE_CHANGE_LAG_522D', 'CUMULATIVE_INJ_VOL']
        for col in injection_cols:
            if col not in df.columns:
                df[col] = 0.0

        for well_name in df['WELLNAME'].unique():
            iw_filter = (df['WELLNAME'] == well_name) & (df['WELL_TYPE'] == 'WI')
            df.loc[iw_filter, 'INJ_VOLUME_LAG_542D'] = df.loc[iw_filter, 'WAT_INJ'].shift(542).fillna(0)
            df.loc[iw_filter, 'INJ_VOLUME_LAG_522D'] = df.loc[iw_filter, 'WAT_INJ'].shift(522).fillna(0)
            df.loc[iw_filter, 'INJ_RATE_CHANGE'] = df.loc[iw_filter, 'WAT_INJ'].diff(30).bfill() / 30
            df.loc[iw_filter, 'INJ_RATE_CHANGE_LAG_542D'] = df.loc[iw_filter, 'INJ_RATE_CHANGE'].shift(542).fillna(0)
            df.loc[iw_filter, 'INJ_RATE_CHANGE_LAG_522D'] = df.loc[iw_filter, 'INJ_RATE_CHANGE'].shift(522).fillna(0)
            df.loc[iw_filter, 'CUMULATIVE_INJ_VOL'] = df.loc[iw_filter, 'WAT_INJ'].cumsum()

        return df

    def compute_water_cut(self, df):
        if 'WATER_CUT' not in df.columns:
            df['WATER_CUT'] = 0.0

        for well_name in df['WELLNAME'].unique():
            well_filter_op = (df['WELLNAME'] == well_name) & (df['WELL_TYPE'] == 'OP')
            water_cut_calculation = df.loc[well_filter_op, 'WAT_PROD'] / (df.loc[well_filter_op, 'OIL_PROD'] + df.loc[well_filter_op, 'WAT_PROD'])
            df.loc[well_filter_op, 'WATER_CUT'] = water_cut_calculation.replace([np.inf, -np.inf, np.nan], 0)

        return df

    def compute_seasonal_features(self, df):
        if not np.issubdtype(df['DATE'].dtype, np.datetime64):
            df['DATE'] = pd.to_datetime(df['DATE'])

        if 'INJ_MONTH' not in df.columns:
            df['INJ_MONTH'] = 0

        peak_injection_months = [3, 4, 5]
        medium_injection_months = [8, 9, 10, 11, 12, 1]
        low_injection_months = [2, 6, 7]

        df['INJ_MONTH'] = np.select(
            [df['DATE'].dt.month.isin(peak_injection_months),
             df['DATE'].dt.month.isin(medium_injection_months),
             df['DATE'].dt.month.isin(low_injection_months)],
            [3, 2, 1], 
            default=0
        )
        return dfß