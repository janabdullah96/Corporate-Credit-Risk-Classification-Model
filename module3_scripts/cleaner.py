import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set_style('darkgrid')


class DatasetCleaner:

    """Diagnoses and cleans dataframe"""

    def __init__(self, df, dep_var, continuous_cols, categorical_cols, exceptions=[]):

        """
        Args:
            df:
                Pandas dataframe containing all variables
            dep_var:
                String denoting the dependent/target variable
            continuous_cols:
                List of continuous variables
            categorical_cols:
                List of categorical variables
            exceptions:
                List of variables to not read and report cleaning diagnostics
        """
        self.df = df
        self.dep_var = dep_var
        self.cont_cols = continuous_cols
        self.cat_cols = categorical_cols
        self.exceptions = exceptions
        self.df_clean = None
        self.df_cont_clean = None
        self.df_cat_clean = None
        return

    def clean(self, config_dict):
        
        """
        Removes data from dataframe per the specified configuration

        Args:
            config_dict:
                Nested dictionary where parent key is column name.
                    - Values are dictionaries where keys are 'min' and 'max' 
                      and values are numerical values denoting the range of 
                      values to keep. 
        """

        self.df_clean = self.df
        for k, v in config_dict.items():
            n = len(self.df_clean[k])
            self.df_clean[k] = self.df_clean[k].map(lambda x: None if x < v['min'] or x > v['max'] else x)
            n_removed = self.df_clean[k].isna().sum()
            n_removed_pct = round(n_removed/n*100,2)
            print(f"{n_removed} ({n_removed_pct}% of column) data points in column '{k}' outside of specified range")
        n = len(self.df_clean)
        self.df_clean.dropna(axis=0, inplace=True)
        self.df_cont_clean = self.df[self.cont_cols]
        self.df_cat_clean = self.df[self.cat_cols]
        n_clean = len(self.df_clean)
        n_removed_pct = round((1-n_clean/n)*100,2)
        print(f'\n{n-n_clean} ({n_removed_pct}% of dataframe) rows removed from overall dataframe!')
    
    def filter_cols(self, remove_cols=[], cleaned=True):
        
        """removes columns from relevant attributes"""

        if cleaned:
            self.df_clean.drop(remove_cols, axis=1, inplace=True)
            self.df_cat_clean.drop([col for col in remove_cols if col in self.cat_cols], axis=1, inplace=True)
            self.df_cont_clean.drop([col for col in remove_cols if col in self.cont_cols], axis=1, inplace=True)
        else:
            self.df.drop(remove_cols, axis=1, inplace=True)
        print(f'Removed columns {remove_cols} from all relevant dataframe attributes in instance object!')
                
    def display_summary_report(self):

        """Displays useful information to user to help them setting cleaning configurations"""

        df = self.df[[elem for elem in self.df.columns.values if elem not in self.exceptions]]
        print('----------Displaying frequency distributions with KDE overlay and column summary reports----------')
        print('='*100)
        for col in df.columns:

            values = df[col].values
            summary_dict = {
                'column': col,
                'type': 'continuous' if col in self.cont_cols 
                        else 'categorical' if col in self.cat_cols
                        else 'dep_var' if col == self.dep_var
                        else 'ERROR!',
                'n': len(df[col]),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'stdev': df[col].std(),
                'skew': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'max': df[col].max(),
                'min': df[col].min(),
                '0.005% quantile': df[col].quantile(0.005),
                '0.025% quantile': df[col].quantile(0.025),
                '97.5% quantile': df[col].quantile(0.975),
                '99.5% quantile': df[col].quantile(0.995)
            }

            plt.figure(figsize=(10,8))
            sns.distplot(df[col], hist=True, kde=True)
            plt.xlabel(col)
            plt.ylabel('Normalized Frequency Distribution')
            plt.title(f'Distribution for Column: {col}')
            plt.axvline(summary_dict['mean'], color='k', linestyle='dashed', linewidth=1)
            plt.show()

            summary_df = pd.DataFrame(summary_dict, index=['summary']).T
            value_counts_df = (pd.DataFrame(
                df[col].value_counts(sort=True, ascending=True)
                )
                .reset_index()
                .sort_values(by='index', ascending=False)
                .sort_values(by=col, ascending=True)
                .rename(
                    columns={'index': 'value', col: 'count'}
                    ))
            value_counts_df.index = range(1, len(value_counts_df)+1)

            n_zeros = sum(df[col]==0)
            pct_zeros = n_zeros/len(df[col])
            zeros_df = pd.DataFrame({'zeros': n_zeros, 'pct_zeros': f'{round(pct_zeros*100,2)}%'}, index=['summary']).T
            df_dict = {
                'Summary Report': summary_df, 
                'Bottom 10 Value Counts': value_counts_df.head(10),
                'Top 10 Value Counts': value_counts_df.tail(10), 
                'Zeros Report': zeros_df
                }
            for k, v in df_dict.items():
                print(f'== {k} ==')
                display(v)
                print('\n')

        