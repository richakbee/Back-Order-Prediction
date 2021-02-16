from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


class tuner:
    """
               This class is used while model training . to get the models with best hyper parameters .
               all hyper parameter tuning is done here.

               Written By: richabudhraja8@gmail.com
               Version: 1.0
               Revisions: None
    """

    def __init__(self, file_object, logger_object):
        self.file_obj = file_object
        self.log_writer = logger_object
        pass

    def get_params_for_catboost(self, x_train, y_train):
        """
                                   Method Name: get_params_for_catboost
                                   Description: This method defines as catboost model for classification . It also performs grid search
                                                 to find the best hyper parameters for the classifier.
                                   Output: Returns a catboost model with best hyper parameters
                                   On Failure: Raise Exception

                                   Written By: richabudhraja8@gmail.com
                                   Version: 1.0
                                   Revisions: None

                                """
        #refer EDA for hyper parameter selection
        estimator = CatBoostClassifier()
        params_grid = {'learning_rate': [0.021846000105142593], 'subsample': [0.800000011920929],
             'boosting_type': ['Plain'], 'bootstrap_type': ['MVS'], 'nan_mode': ['Min'],
             'eval_metric': ['Logloss'], 'model_shrink_rate': [0], 'iterations': [1000],
             'model_shrink_mode': ['Constant'], 'sampling_frequency': ['PerTree'],
             'min_data_in_leaf': [1], 'leaf_estimation_iterations': [10],
             'use_best_model': [False], 'penalties_coefficient': [1],
             'l2_leaf_reg': [3], 'leaf_estimation_method': ['Newton'],
             'task_type': ['CPU'], 'boost_from_average': [False],
             'leaf_estimation_backtracking': ['AnyImprovement'],
              'grow_policy': ['SymmetricTree'],
             'feature_border_type': ['GreedyLogSum'], 'border_count': [254],
             'max_leaves': [64], 'score_function': ['Cosine'],
             'class_names': [[0, 1]], 'best_model_min_trees': [1],
             'sparse_features_conflict_fraction': [0], 'rsm': [1],
             'posterior_sampling': [False], 'random_seed': [786],
             'depth': [6], 'random_strength': [1],
             'loss_function': ['Logloss'], 'model_size_reg': [0.5],
              'classes_count': [0]}

        self.log_writer.log(self.file_obj, "Entered get_params_for_catboost of tuner class!!")
        try:
            grid_cv = GridSearchCV(estimator, param_grid=params_grid, cv=5, return_train_score=False)
            grid_cv.fit(x_train, y_train)

            # fetch the best estimator
            best_estimator = grid_cv.best_estimator_
            best_estimator.fit(x_train, y_train)
            self.log_writer.log(self.file_obj,
                                "get best params for xgboost successful"+ str(grid_cv.best_params_) +".Exited get_params_for_xgboost of tuner class!!")

            return best_estimator

        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in get_params_for_xgboost of tuner class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'get best params for xgboost unsuccessful .Exited get_params_for_xgboost of tuner class!!')
            raise e


    def get_params_for_random_forest(self, x_train, y_train):
        """
                                           Method Name: get_params_for_random_forest
                                           Description: This method defines as random_forest model for classification . It also performs grid search
                                                         to find the best hyper parameters for the classifier.
                                           Output: Returns a random_forest model with best hyper parameters
                                           On Failure: Raise Exception

                                           Written By: richabudhraja8@gmail.com
                                           Version: 1.0
                                           Revisions: None

                                        """
        estimator = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight='balanced',
                       criterion='gini', max_depth=10, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0, min_impurity_split=None,
                       min_samples_leaf=6, min_samples_split=7,
                       min_weight_fraction_leaf=0.0, n_estimators=190,
                       n_jobs=-1, oob_score=False, random_state=786, verbose=0,
                       warm_start=False)



        self.log_writer.log(self.file_obj, "Entered get_params_for_random_forest of tuner class!!")
        try:



            # fetch the best estimator
            estimator.fit(x_train, y_train)
            self.log_writer.log(self.file_obj,
                                "get best params for random_forest successful .Exited get_params_for_random_forest of tuner class!!")

            return estimator

        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in get_params_for_random_forest of tuner class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'get best params for random_forest unsuccessful .Exited get_params_for_random_forest of tuner class!!')
            raise e


    def get_params_for_logistic_reg(self, x_train, y_train):
        """
                           Method Name: get_params_for_logistic_reg
                           Description: This method defines as logistic_reg model for classification . It also performs grid search
                                         to find the best hyper parameters for the classifier.
                           Output: Returns a logistic_reg model with best hyper parameters
                           On Failure: Raise Exception

                           Written By: richabudhraja8@gmail.com
                           Version: 1.0
                           Revisions: None

                        """
        estimator = LogisticRegression()
        params_grid = {'tol': [0.0001], 'class_weight': [None],
                   'random_state': [786], 'solver': ['lbfgs'],
                   'verbose': [0], 'fit_intercept': [True], 'warm_start': [False],
                   'n_jobs': [None], 'l1_ratio': [None], 'penalty': ['l2'],
                   'max_iter': [1000], 'intercept_scaling': [1], 'multi_class': ['auto'],
                   'dual': [False], 'C': [1.0]}


        self.log_writer.log(self.file_obj, "Entered get_params_for_logistic_reg of tuner class!!")
        try:

            grid_cv = GridSearchCV(estimator, param_grid=params_grid, cv=5, return_train_score=False)
            grid_cv.fit(x_train, y_train)

            # fetch the best estimator
            best_estimator = grid_cv.best_estimator_
            best_estimator.fit(x_train, y_train)
            self.log_writer.log(self.file_obj,
                                "get best params for logistic_reg successful ."+str(grid_cv.best_params_) +".Exited get_params_for_logistic_reg of tuner class!!")

            return best_estimator

        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in get_params_for_logistic_reg of tuner class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'get best params for logistic_reg unsuccessful .Exited get_params_for_logistic_reg of tuner class!!')
            raise e


    def get_params_for_lightgbm(self, x_train, y_train):
        """
                   Method Name: get_params_for_lightgbm
                   Description: This method defines as lightgbm model for classification . It also performs grid search
                                 to find the best hyper parameters for the classifier.
                   Output: Returns a lightgbm model with best hyper parameters
                   On Failure: Raise Exception

                   Written By: richabudhraja8@gmail.com
                   Version: 1.0
                   Revisions: None

                """
        estimator = LGBMClassifier()
        params_grid={'class_weight': [None], 'max_depth': [-1], 'random_state': [786],
             'learning_rate': [0.1], 'n_jobs': [-1], 'n_estimators': [100],
             'subsample': [1.0], 'boosting_type': ['gbdt'], 'min_split_gain': [0.0],
             'subsample_freq': [0], 'min_child_weight': [0.001], 'min_child_samples': [20],
             'importance_type': ['split'], 'reg_lambda': [0.0], 'colsample_bytree': [1.0],
             'reg_alpha': [0.0], 'objective': [None], 'num_leaves': [31], 'silent': [True],
             'subsample_for_bin': [200000]}

        self.log_writer.log(self.file_obj, "Entered get_params_for_lightgbm of tuner class!!")
        try:

            grid_cv = GridSearchCV(estimator, param_grid=params_grid, cv=5 ,return_train_score=False)
            grid_cv.fit(x_train, y_train)

            #fetch the best estimator
            best_estimator=grid_cv.best_estimator_
            best_estimator.fit(x_train, y_train)
            self.log_writer.log(self.file_obj,
                                "get best params for lightgbm successful ."+ str(grid_cv.best_params_) +".Exited get_params_for_lightgbm of tuner class!!")

            return best_estimator

        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in get_params_for_lightgbm of tuner class!! Exception message:' + str(e))
            self.log_writer.log(self.file_obj,
                                'get best params for lightgbm unsuccessful .Exited get_params_for_lightgbm of tuner class!!')
            raise e
