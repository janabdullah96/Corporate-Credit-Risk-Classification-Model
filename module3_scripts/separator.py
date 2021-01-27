
class Separator:
    
    """
    Separates the input dataframe into categorical and continuous variables
    Includes methods for user to manually classify variables as well
    """

    def __init__(
        self, df, dep_var, n_unique_threshold, drop_cols=[]
        ):

        """
        Args:
            df:
                Pandas dataframe of raw input data
            dep_var:
                String denoting the dependent/target variable in the dataset
            n_unique_threshold:
                Integer cutoff value for continuous/categorical variable determination.
                    -Any columns with number of unique values above this parameter will be classified 
                     as continuous, otherwise, categorical
            drop_cols:
                List of columns to drop from input dataframe
        """

        self.df = df.drop(drop_cols, axis=1)
        self.dep_var = dep_var
        self.n_unique_threshold = n_unique_threshold
        self.df_cont = None
        self.df_cat = None
        return

    def split_categorical_continuous(self):

        print('=== Beginning process of separating continuous and categorical variable columns ===')
        df = self.df.drop(self.dep_var, axis=1)
        self.df_cont = df[[col for col in df if df[col].nunique() > self.n_unique_threshold]]
        self.df_cat = df[[col for col in df if df[col].nunique() <= self.n_unique_threshold]]
        print(f'\nContinuous variable columns: \n{self.df_cont.columns.values}')
        print(f'Categorical variable columns: \n{self.df_cat.columns.values}\n')
        print('=== Completed separating continuous and categorical variables ===')

    def manual_separation_override(self, _from, _to, cols=[]):
        
        """
        Method where user can call on the object to manually re-arrage/re-classify variables

        Args:
            _from:
                String of classification to move variable FROM (has to be either 'cont' or 'cat')
            _to:
                String of classification to move variable TO (has to be either 'cont' or 'cat')
            cols:
                List of cols to move in specified direction
        """

        print('=== Beginning manual amendments/overrides of continuous and categorical variable column separation ===')
        if _from == 'cont' and _to == 'cat':
            self.df_cont_cols_ls = [elem for elem in self.df_cont.columns if elem not in cols]
            self.df_cat_cols_ls = list(self.df_cat.columns.values) + cols
        elif _from == 'cat' and _to == 'cont':
            self.df_cat_cols_ls = [elem for elem in self.df_cat.columns if elem not in cols]
            self.df_cont_cols_ls = list(self.df_cont.columns.values) + cols
        else:
            raise ValueError("Check _from and _to inputs! Should be either 'cont' or 'cat'!")
        self.df_cont = self.df[self.df_cont_cols_ls]
        self.df_cat = self.df[self.df_cat_cols_ls]
        print(f'\nAmended Continuous variable columns: \n{self.df_cont_cols_ls}')
        print(f'\nAmended Categorical variable columns: \n{self.df_cat_cols_ls}\n')
        print('=== Completed manual override ===')

    def manual_continuous_to_categorical_transform_binary(self, cols=[]):

        """
        Method to convert categorical column values to binary values. 
        All false-y values will be set to 0 and truth-y values set to 1

        Args:
            cols:
                List of cols to convert
        """

        print('=== Beginning process of manually converting continious variables into binary categorical variables ===')
        for col in cols:
            print(f'\n Converting column: {col}')
            for df in [self.df, self.df_cat]:
                df[col] = df[col].map(lambda x: 0 if x == False else 1).copy()
        print('\n=== Completed manual continuous to binary categorical override conversion ===')
        


