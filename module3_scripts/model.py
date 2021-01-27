
import pandas as pd
import numpy as np
import json
from statistics import mode
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, StackingClassifier)
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (precision_score, recall_score, accuracy_score, f1_score, 
                             roc_curve, auc, confusion_matrix)
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


class Model:
    
    """
    Class for instantiating and fitting models, and running evaluation metrics
    """
    
    __BASE_MODEL_CLASSES = {
        'DummyClassifier': DummyClassifier,
        'LogisticRegression': LogisticRegression,
        'KNN': KNeighborsClassifier,
        'DecisionTree': DecisionTreeClassifier,
        'Bagging': BaggingClassifier,                      
        'RandomForest': RandomForestClassifier,
        'AdaBoost': AdaBoostClassifier,                  
        'GradientBoosting': GradientBoostingClassifier,
        'XGBoost': XGBClassifier,
        'GaussianNB': GaussianNB,
        'SVM': SVC
    }
    
    __BASE_PARAM_GRID = {
        'SVM': {'probability': [True]},
        'XGBoost': {'verbosity': [0]}
    }
    
    def __init__(
        self, 
        X_train, 
        X_test,
        y_train, 
        y_test, 
        random_state=123, 
        gs_param_grid = {}, 
        cv=3
    ):
        
        """
        Args:
        X_train:
            pandas DataFrame of predictor variables training set
        X_test:
            pandas DataFrame of predictor variables test set
        y_train:
            pandas DataFrame of target variable training set
        y_test:
            pandas DataFrame of target variable test_set
        random_state:
            integer denoting random state (default = 123)
        gs_param_dict:
            dictionary of grid search parameter dictionaries
            where keys are the names of the classificatons models to
            apply the grid search to and the values are the parameter grids
        cv:
            integer denoting number of folds for cross-validation (for Grid Search)
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_state = random_state
        self.cv = cv
        self.gs_param_grid = gs_param_grid
        self._class_labels = np.sort(self.y_train.unique())
        self._instantiated_model_dict = {}
        self._fitted_model_dict = {}
        self._sets_dict = {
            "Training": [
                self.X_train, 
                self.y_train
            ], 
            "Test": [
                self.X_test, 
                self.y_test
            ]
        }
        self.y_pred_dict = {}
        return
    
    def instantiate_models(self, models=__BASE_MODEL_CLASSES.keys(), display_best_params=False):
        
        """
        Args:
            models:
                List of model classifier names
                -refer to '__BASE_MODEL_CLASSES' class attribute for 
                 model naming conventions
            display_best_params:
                Boolean value to toggle display of best GridSearchCV parameters
                (default=False). Note* would require more time as GS classifier
                would have to be fitted first to get best parameters. 
        
        This method iterates through all the elements in 'models' 
        and instantiates each classfier using a OneVsRestClassifier
        wrapper. If a GridSearchCV parameter grid is passed for a
        model, the GS will be performed during instantiation. 
        After each instantiation, the '_instantiated_model_dict'
        instance attribute will be updated where:
            key: name of classifier
            value: instantiated classifier object
        """
        
        print('='*50)
        classifiers_dict = {
            model: self.__BASE_MODEL_CLASSES.get(model) 
            for model in models
        }
        params_dict = {
            **{
                clf: {**gs_param_grid, **self.__BASE_PARAM_GRID.get(clf, {})} 
                for clf, gs_param_grid in self.gs_param_grid.items()
                },
            **{
                clf: base_params 
                for clf, base_params in self.__BASE_PARAM_GRID.items()
                if clf not in self.gs_param_grid.keys()
                }
        }
        for clf_name, classifier in classifiers_dict.items():
            print(f'Instantiating {clf_name}...\n')
            try:
                clf = GridSearchCV(
                    classifier(), 
                    param_grid=params_dict.get(clf_name), 
                    scoring='accuracy',
                    cv=self.cv
                )
                if clf_name in self.gs_param_grid.keys():
                    param_type = 'Grid Search Params:'
                elif clf_name in self.__BASE_PARAM_GRID.keys():
                    param_type = 'Static Params:'
                else:
                    pass
                print(f'{param_type} {json.dumps(params_dict.get(clf_name), indent=4)}\n')
                if display_best_params:
                    print(
                        f'Optimal {clf_name} params: {json.dumps(clf.fit(self.X_train, self.y_train).best_params_, indent=4)}\n'
                    )
            except TypeError:
                clf = classifier()
            self._instantiated_model_dict[clf_name] = OneVsRestClassifier(clf)
            print('='*50)
            
    def fit_all(self):
        
        """
        Note* This method requires that the .instantiate_models() method
              be called on the instance object as a pre-requisite.
        
        This method uses sklearn's .fit() and .predict() methods for 
        each classifier and populates:
        1. the 'fitted_model_dict' attribute where:
            key: name of classifier
            value: fitted classifier object
        2. the 'y_pred_dict' attribute where:
            key: name of classifier
            value: dictionary where:
                key: set ('Training' or 'Test')
                value: predictions for the set
        """
        
        for clf_name, model in self._instantiated_model_dict.items():
            print(f'Fitting and predicting target class with {clf_name}...')
            self._fitted_model_dict[clf_name] = model.fit(self.X_train, self.y_train)
            #generate training and test set predictions for classifier
            #data[0] is the X dataset for the set
            self.y_pred_dict[clf_name] = {
                set_: self._fitted_model_dict[clf_name].predict(data[0]) 
                for set_, data in self._sets_dict.items()
            }
            
    def display_metrics(self):
        
        """
        This method displays the following:
            1. Confusion matrices for the training and test sets of each classifier
            2. Multi-class micro-averaged ROC curbes for the training and test sets
               of each classifier
            3. Aggregated metrics dataframe for each classifier, for each set. 
            4. A bar graph visualizing #3. 
        """
        for clf_name, model in self._instantiated_model_dict.items():
            print('='*120)
            self.__display_confusion_matrices(clf_name, model)
            self.__display_roc_curves(clf_name, model)
            print('='*120)
        self.__generate_metrics_df()
        print('='*120)
        
    def __generate_metrics_df(self):
        
        """
        Generates an aggregated evaluation metrics report for each 
        classifier. Displays each classifiers score for precision, recall,
        accuracy, and f1.
        """
        metrics_dict = {}
        for set_, data in self._sets_dict.items():
            for clf_name, model in self._instantiated_model_dict.items():
                y = data[1]
                y_pred = self.y_pred_dict[clf_name][set_]
                metrics_dict[clf_name] = [
                    precision_score(y, y_pred, average='weighted'),
                    recall_score(y, y_pred, average='weighted'),
                    accuracy_score(y, y_pred),
                    f1_score(y, y_pred, average='weighted')
                ]
            df_metrics = pd.DataFrame(
                metrics_dict, 
                index=['precision', 'recall', 'accuracy', 'f1']
            ).T
            self.__plot_metrics_df(set_, df_metrics)

    def __plot_metrics_df(self, set_, df):
        
        """
        Plots the metrics dataframe as a bar graph
        """
        print(f'\n=== Displaying Evaluation Metrics Across All Classifiers for {set_} Data ===')
        display(df)
        df = df.sort_values(
            by=['accuracy', 'f1', 'precision', 'recall'], 
            ascending=False
        ).copy()
        plt.figure(figsize=(15,7))
        bar_width = 0.2
        x = np.arange(df.shape[0])
        for col in df.columns:
            plt.bar(x, df[col], width=bar_width, edgecolor='white', label=col)
            x = [x + bar_width for x in x]
        plt.xlabel('Model', fontsize=15)
        plt.ylabel('Score', fontsize=15)
        plt.xticks([r + bar_width + 0.02 for r in range(len(x))], df.index, fontsize=13, rotation=-35)
        plt.yticks(fontsize=13)
        plt.ylim([0, 1])
        plt.title(f'{set_} Set Metric Scores by Model', fontsize=15)
        plt.legend(loc='best', fontsize=14)
        plt.show()
        
    def __display_confusion_matrices(self, clf_name, model):
        
        """
        Displays confusion matrices for the train and tests sets of each classifier
        """
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(15, 7.5)
        for i, (set_, data) in enumerate(self._sets_dict.items()):
            y = data[1]
            y_pred = self.y_pred_dict[clf_name][set_]
            conf_matrix = confusion_matrix(y, y_pred)
            sns.heatmap(
                conf_matrix, 
                cmap='Blues', 
                annot=True, 
                fmt='.6g', 
                xticklabels=self._class_labels, 
                yticklabels=self._class_labels, 
                cbar=False,
                ax=ax[i]
            )
            ax[i].set_title(f'{clf_name} {set_} Set Confusion Matrix', fontsize=17)
            ax[i].tick_params(axis='both', which='major', labelsize=15)
            ax[i].tick_params(axis='both', which='minor', labelsize=15)
            plt.tight_layout()
        plt.show()

    def __display_roc_curves(self, clf_name, model):
        
        """
        Generates and displays a micro-averaged ROC curve and AUC score
        for the training and test sets of each classifier.
        """
        #use sklearn's label_binarize function to convert the target class
        #into binary arrays to be able to calculate micro-averaged ROC and AUC.
        #refit the models with the binarized y_train array
        binarized_fit = model.fit(
            self.X_train,
            label_binarize(self.y_train, classes=self._class_labels)
        )
        roc_curve_dict = {}
        for i, (set_, data) in enumerate(self._sets_dict.items()):
            X = data[0]
            y = data[1]
            y_score = binarized_fit.predict_proba(X)
            #binarize the actual target arrays for ROC and AUC calculation
            y = label_binarize(y, classes=self._class_labels)
            fpr, tpr, _ = roc_curve(y.ravel(), y_score.ravel())
            roc_auc = auc(fpr, tpr)
            roc_curve_dict[set_] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc} 
                
        plt.figure(figsize=(7, 7))
        for set_, line_color in {"Test": "orange", "Training": "green"}.items():
            plt.plot(
                roc_curve_dict[set_]["fpr"], 
                roc_curve_dict[set_]["tpr"], 
                linestyle='-',
                color=line_color, 
                label=f"{set_} set (AUC: {round(roc_curve_dict[set_]['auc'], 5)})"
            )
        plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
        plt.title(f'{clf_name} Multiclass Micro-averaged ROC Curve', fontsize=15)
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.legend(loc='best', fontsize=12)
        plt.show();


class Ensemble:
    
    @classmethod
    def stack_models(cls, y_pred_dict, y_test, base=[], method='weighted'):
        
        """
        class method that computes the score of model-stacked predictions
        Stacked predictions are weighted by individual model accuracy scores
        
        Args:
            y_pred_dict:
                Dictionary where:
                    keys: name of classifier 
                    values: predictions of classifier. Must be predictions of the 
                            predictor variable test set. 
            y_test:
                Array of target variable test set values
            base:
                Classifier to use as a base. The predictions of these classifiers 
                will have a higher weight in the stacked models predictions array
                (Weighted as 100% accurate) (Default=[])
            method:
                String denoting method of stacking predictions
                Method descriptions can be found in docstrics of static methods
        """
        
        print(
            '=== Initiating model predictions stacking process ===',
            f'\nBase: {base if base else None}',
            f'\nMethod: {method}\n'
        )
        
        scores_dict = {
            clf_name: accuracy_score(y_test, preds) 
            for clf_name, preds in y_pred_dict.items()
        }
        display(
            '=== Displaying Invididual Classifier Scores ===',
            pd.DataFrame(scores_dict, index=['score']).T
        )
        
        best_clf = sorted(scores_dict, key=lambda s: scores_dict[s], reverse=True)[0]
        best_clf_score = scores_dict.get(best_clf)
        
        if method == 'weighted':
            final_preds = cls._weighted_method(y_pred_dict, y_test, scores_dict, base)
        elif method == 'mode':
            final_preds = cls._mode_method(y_pred_dict, y_test, scores_dict)
        else:
            raise ValueError("Enter either 'weighted' or 'mode' for method parameter!")
        stacked_score = accuracy_score(y_test, final_preds)
    
        if best_clf_score > stacked_score:
            print(
                '\n' + '='*40 + ' ** NOTE ** ' + '='*40,
                f'\nThe {best_clf} classifier predictions alone are better than the stacked predictions',
                f'\nRecommend taking the predictions of solely the {best_clf} classifier',
                f'\n\nBest Individual Classifier Score: {best_clf_score}\n',
                '='*90
            )
        
        #calculate baseline accuracy:
        #the accuracy a model would achieve if it predicted the class with the most 
        #observations every time
        baseline_acc = sum(y_test == mode(y_test))/len(y_test)

        print(f'\nStacked Score: {stacked_score}')
        print(f'\nBaseline Accuracy: {baseline_acc}')
        print(f'\nRandom Guessing Accuracy: {1/y_test.nunique()}')

        #plot confusion matrix for y-actual against model-stacked y predictions
        conf_matrix = confusion_matrix(y_test, final_preds)
        plt.figure(figsize=(10,10))
        sns.heatmap(
                conf_matrix, 
                cmap='Blues', 
                annot=True, 
                fmt='.6g', 
                xticklabels=np.unique(np.array(final_preds)), 
                yticklabels=np.unique(np.array(y_test)), 
                cbar=False
            )
        plt.title(f'Model Stacked Predictions Confusion Matrix', fontsize=17)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.tick_params(axis='both', which='minor', labelsize=15)
        plt.show()
        return final_preds
    
    @staticmethod
    def _mode_method(y_pred_dict, y_test, scores_dict):
        
        """
        Retrieves stacked predictions by taking the class value
        predicted by the most classifiers, regardless of classifier
        scores. Note* When there is a tie in predictions amongst
        classifiers, weighting is applied.
        
        Args:
            y_pred_dict:
                Dictionary where:
                    keys: name of classifier 
                    values: predictions of classifier. Must be predictions of the 
                            predictor variable test set. 
            y_test:
                Array of target variable test set values
            scores_dict:
                Dictionary where:
                    keys: name of classifier
                    values: accuracy score of classifier model
        Returns:
            list of final predictions
        """
        def get_mode(row, df, y_test, y_pred_dict, scores_dict):

            try:
                return mode([row[col] for col in df.columns])
            except:
                weighted_preds = []
                for clf_name in df.columns:
                    pred = row[clf_name]
                    clf_score = scores_dict.get(clf_name)
                    votes = int(round(clf_score*10000, 0))
                    weighted_preds.extend([pred]*votes)
                return mode(weighted_preds)
            
        df = pd.DataFrame(y_pred_dict, columns=y_pred_dict.keys())
        df['mode'] = df.apply(lambda row: get_mode(row, df, y_test, y_pred_dict, scores_dict), axis=1)
        return df['mode'].values
    
    @staticmethod
    def _weighted_method(y_pred_dict, y_test, scores_dict, base):
        
        """
        Retrieves stacked predictions by weighing each 
        classifier's prediction by the classifier's score
        Args:
            y_pred_dict:
                Dictionary where:
                    keys: name of classifier 
                    values: predictions of classifier. Must be predictions of the 
                            predictor variable test set. 
            y_test:
                Array of target variable test set values
            scores_dict:
                Dictionary where:
                    keys: name of classifier
                    values: accuracy score of classifier model
            base:
                Classifier to use as a base. The predictions of these classifiers 
                will have a higher weight in the stacked models predictions array
                (Weighted as 100% accurate) (Default=[])
        Returns:
            list of final predictions 
        """
        weighted_preds = [[] for x in range(len(y_test))]
        for clf_name, preds in y_pred_dict.items():
            score = scores_dict.get(clf_name)
            if clf_name in base:
                votes = 10000
            else:
                votes = int(round(score*10000, 0))
            for i, j in enumerate(preds):
                weighted_preds[i].extend([j]*votes)
        return [mode(elem) for elem in weighted_preds]