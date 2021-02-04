import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sns.set_style("darkgrid")


class Preprocessor:

    """Pre-processes dataset for modeling"""

    def __init__(
        self,
        df, 
        dep_var, 
        categorical_cols, 
        continuous_cols, 
        smote=False, 
        random_state=123, 
        test_size=0.25,
        transformation_exceptions=[]
    ):

        """
        Args:
            df: 
                Pandas input dataframe containing all variables
            dep_var:
                String denoting the dependent/target variable
            categorical_cols:
                List of categorical variables
            continuous_cols:
                List of continuous variables
            smote:
                Boolean to toggle SMOTE resampling (default=False)
            random_state:
                Integer setting random datate (default=123)
            test_size:
                Integer setting test size for train-test split (default=0.25)
            transformation_exceptions:
                List of predictor variables to exempt from scaling/transformation
        """
        self.input_df = df
        self.dep_var = dep_var
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.smote = smote
        self.random_state = random_state
        self.test_size = test_size
        self.transformation_exceptions = transformation_exceptions
        self.y = df[dep_var]
        self.X = df.drop(dep_var, axis=1)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def preprocess(self):
        
        """
        Preprocesses the input data in below steps:
            1. Split dataset into training and test sets
            2. One-hot encode categorical variables
            3. Apply SMOTE resampling if applicable
            4. Use StandardScaler to scale predictor training and test sets
            
        Assigns all transformations directly to instance attributes
        self.X_train, self.X_test, self.y_train and self.y_test
        """

        print('=== Initiating pre-processing of dataset ===')
        
        print('Applying train-test split on dataset...')
        print(f'\t Test set size: {self.test_size}')
        print(f'\t Random state: {self.random_state}')
        (self.X_train, 
         self.X_test, 
         self.y_train, 
         self.y_test) = train_test_split(
            self.X, 
            self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        print('Applying one-hot encoding on all categorical variables...')
        ohe = OneHotEncoder()
        #fit using training set
        ohe.fit(self.X_train[self.categorical_cols])
        ohe_exceptions_for_transformation = []
        #transform training and test sets
        ohe_dfs = []
        for df in [self.X_train, self.X_test]:
            ohe_transformed = ohe.transform(df[self.categorical_cols]).toarray()
            ohe_columns = ['Sector_'+col for col in ohe.categories_][0]
            ohe_exceptions_for_transformation.extend(ohe_columns)
            df_ohe_transformed = pd.DataFrame(ohe_transformed, columns=ohe_columns)
            df = df.drop(self.categorical_cols, axis=1)
            df = pd.concat([df.reset_index(), df_ohe_transformed], axis=1).drop('index', axis=1)
            ohe_dfs.append(df)
        
        self.X_train, self.X_test = ohe_dfs
        
        if self.smote:
            print('Applying SMOTE resampling on training set...')
            self.X_train, self.y_train = SMOTE().fit_sample(self.X_train, self.y_train)
        
        self.X_train.reset_index(inplace=True)
        self.X_test.reset_index(inplace=True)
        
        print('Transforming predictor training and test sets using StandardScaler...')
        print(f'\tColumns exempted from transformation: {self.transformation_exceptions}')
        
        exceptions = ohe_exceptions_for_transformation + self.transformation_exceptions
        
        train_exceptions = [col for col in self.X_train.columns if col in exceptions]
        transform_training_df = self.X_train.drop(train_exceptions, axis=1)
        transform_training_df_exception = self.X_train[train_exceptions]
        
        test_exceptions = [col for col in self.X_test.columns if col in exceptions]
        transform_test_df = self.X_test.drop(test_exceptions, axis=1)
        transform_test_df_exception = self.X_test[test_exceptions]
            
        scaler = StandardScaler()
        scaler.fit(transform_training_df)
        transform_training_df = pd.DataFrame(
            scaler.transform(transform_training_df), 
            columns=transform_training_df.columns
        )
        transform_test_df = pd.DataFrame(
            scaler.transform(transform_test_df), 
            columns=transform_test_df.columns
        ) 
        self.X_train = pd.concat(
            [transform_training_df_exception, 
            transform_training_df],
            axis=1
        ).drop('index', axis=1)
        self.X_test = pd.concat(
            [transform_test_df_exception, 
            transform_test_df], 
            axis=1
        ).drop('index', axis=1)

        print('=== Completed pre-processing! ===')

    def display_transformed_distributions(self):
        
        """
        Displays distributions of transformed variables in 
        predictor training set
        """
        try:
            if not self.X_train.empty:
                print('=== Displaying distributions of transformed variables in predictor training set ===')
                print('='*83)
                plt.figure(figsize=(10,35))
                n = len(self.X_train.columns)
                ncols = 3
                nrows = n//3 + 1
                cols = [col for col in self.X_train if col not in self.transformation_exceptions]
                for i, col in enumerate(cols):
                    plt.subplot(nrows, ncols, i+1)
                    plt.tight_layout()
                    sns.distplot(self.X_train[col], hist=True, kde=True)
        except AttributeError:
            raise TypeError(
                'Predictor training set not assigned! Call the .preprocess() method on object first!'
            )