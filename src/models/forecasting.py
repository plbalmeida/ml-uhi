import numpy as np
from sklearn.metrics import mean_squared_error
from abc import ABC, abstractmethod
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.multioutput import RegressorChain
from hyperopt import hp, fmin, tpe, rand, STATUS_OK, Trials, space_eval
import pickle


class Trainer(ABC):
    '''Basic training class, defines basic structures of training'''
    
    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class RidgeHyperoptTS(Trainer):
    '''
    Training Ridge Regression model for time series forecasting with 
    Bayesian hyperparameter optimization.
    '''

    def __init__(self):
        pass

    def fit(self, X, y, n_splits=3, max_train_size=None, test_size=None, save_path='rd.pkl'):
        '''
        Define basic training structure.

        Parameters
        ----------
        X : pandas data frame
            Data frame with predictor features.
        y : pandas series
            Pandas series with target data.
        n_splits : int
            Number of cross-validation folds. 
        test_size : int
            Size of forecasting horizon.
        '''

        def rd_hyperopt(X, y, verbose=False, persistIterations=True):
            '''
            Bayesian hyperparameter optimization with Hyperopt.
            
            Parameters
            ----------
            X : pandas data frame
                Data frame with predictor features.
            y : pandas series
                Pandas series with target data.
            
            Returns
            -------
            space_eval(parameters, best) : dict
                Best hyperparameters. 
            '''

            def objective(params):
                '''
                Function to minimize RMSE score.
                
                Paramenters
                -----------
                params : python dict
                    Hyperparamenter space to check MSE score.

                Returns
                -------
                score : float   
                    MSE score.
                '''

                model = RegressorChain(Ridge(**params, random_state=123), random_state=123)

                score = np.sqrt(-cross_val_score(
                    model, 
                    X, 
                    y, 
                    cv=TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size, test_size=test_size), 
                    scoring='neg_mean_squared_error', 
                    verbose=False, 
                    n_jobs=-1
                    ).mean())

                return {'loss': score, 'status': STATUS_OK}

            parameters = {
                'alpha' : hp.uniform('alpha', 1.0, 10.0),
                }

            best = fmin(
                objective, 
                space=parameters, 
                algo=tpe.suggest, 
                max_evals=3, 
                trials=Trials(),
                rstate=np.random.default_rng(123)
                )

            return space_eval(parameters, best)

        best_hp = rd_hyperopt(X, y)

        rd = RegressorChain(Ridge(**best_hp, random_state=123), random_state=123)
        rd.fit(X, y)

        with open(save_path, 'wb') as f:
            pickle.dump(rd, f)

    def predict(self, future, save_path='rd.pkl'):
        '''
        Define basic prediction structure.
        
        Parameters
        ----------
        future : pandas data frame 
            Data frame with predictor features.
        
        Returns
        -------
        list(prediction) : list
            List with predictions.
        '''

        with open(save_path, 'rb') as f:
            model = pickle.load(f)

        prediction = model.predict(future)

        return list(prediction)


class GBHyperoptTS(Trainer):
    '''
    Training Gradient Boosting Regressor model for time series forecasting with 
    Bayesian hyperparameter optimization.
    '''

    def __init__(self):
        pass

    def fit(self, X, y, n_splits=3, max_train_size=None, test_size=None, save_path='gb.pkl'):
        '''
        Define basic training structure.

        Parameters
        ----------
        X : pandas data frame
            Data frame with predictor features.
        y : pandas series
            Pandas series with target data.
        n_splits : int
            Number of cross-validation folds. 
        test_size : int
            Size of forecasting horizon.
        '''

        def gb_hyperopt(X, y, verbose=False, persistIterations=True):
            '''
            Bayesian hyperparameter optimization with Hyperopt.
            
            Parameters
            ----------
            X : pandas data frame
                Data frame with predictor features.
            y : pandas series
                Pandas series with target data.
            
            Returns
            -------
            space_eval(parameters, best) : dict
                Best hyperparameters. 
            '''

            def objective(params):
                '''
                Function to minimize RMSE score.
                
                Paramenters
                -----------
                params : python dict
                    Hyperparamenter space to check MSE score.

                Returns
                -------
                score : float   
                    MSE score.
                '''

                params = {
                    'max_depth' : int(params['max_depth']),
                    'max_iter': int(params['max_iter']),
                    'learning_rate': params['learning_rate'],
                    'min_samples_leaf': int(params['min_samples_leaf'])
                    }

                model = RegressorChain(
                    HistGradientBoostingRegressor(
                        **params,
                        warm_start=True,
                        early_stopping=True,
                        random_state=123
                        ), 
                    random_state=123
                )   

                score = np.sqrt(-cross_val_score(
                    model, 
                    X, 
                    y, 
                    cv=TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size, test_size=test_size), 
                    scoring='neg_mean_squared_error', 
                    verbose=False, 
                    n_jobs=-1
                    ).mean())

                return {'loss': score, 'status': STATUS_OK}

            parameters = {
                'max_depth' : hp.uniform('max_depth', 3, 50),
                'max_iter': hp.quniform('max_iter', 100, 300, 5),
                'learning_rate': hp.loguniform('learning_rate', -3, 1),
                'min_samples_leaf': hp.uniform('min_samples_leaf', 20, 60)
                }

            best = fmin(
                objective, 
                space=parameters, 
                algo=tpe.suggest, 
                max_evals=3, 
                trials=Trials(),
                rstate=np.random.default_rng(123)
                )

            return space_eval(parameters, best)

        best_hp = gb_hyperopt(X, y)

        best_hp = {
            'max_depth' : int(best_hp['max_depth']),
            'max_iter': int(best_hp['max_iter']),
            'learning_rate': best_hp['learning_rate'],
            'min_samples_leaf': int(best_hp['min_samples_leaf'])
            }

        gb = RegressorChain(
            HistGradientBoostingRegressor(
                **best_hp,
                warm_start=True,
                early_stopping=True,
                random_state=123
                ), 
            random_state=123
        )
            
        gb.fit(X, y)

        with open(save_path, 'wb') as f:
            pickle.dump(gb, f)

    def predict(self, future, save_path='gb.pkl'):
        '''
        Define basic prediction structure.
        
        Parameters
        ----------
        future : pandas data frame 
            Data frame with predictor features.
        
        Returns
        -------
        list(prediction) : list
            List with predictions.
        '''

        with open(save_path, 'rb') as f:
            model = pickle.load(f)

        prediction = model.predict(future)

        return list(prediction)


class MLPHyperoptTS(Trainer):
    '''
    Training MLP Regressor model for time series forecasting with 
    Bayesian hyperparameter optimization.
    '''

    def __init__(self):
        pass

    def fit(self, X, y, n_splits=3, max_train_size=None, test_size=None, save_path='mlp.pkl'):
        '''
        Define basic training structure.

        Parameters
        ----------
        X : pandas data frame
            Data frame with predictor features.
        y : pandas series
            Pandas series with target data.
        n_splits : int
            Number of cross-validation folds. 
        test_size : int
            Size of forecasting horizon.
        '''

        def mlp_hyperopt(X, y, verbose=False, persistIterations=True):
            '''
            Bayesian hyperparameter optimization with Hyperopt.
            
            Parameters
            ----------
            X : pandas data frame
                Data frame with predictor features.
            y : pandas series
                Pandas series with target data.
            
            Returns
            -------
            space_eval(parameters, best) : dict
                Best hyperparameters. 
            '''

            def objective(params):
                '''
                Function to minimize RMSE score.
                
                Paramenters
                -----------
                params : python dict
                    Hyperparamenter space to check MSE score.

                Returns
                -------
                score : float   
                    MSE score.
                '''

                model = MLPRegressor(
                    **params, 
                    warm_start=True, 
                    early_stopping=True, 
                    random_state=123, 
                    max_iter=300
                    )   

                score = np.sqrt(-cross_val_score(
                    model, 
                    X, 
                    y, 
                    cv=TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size, test_size=test_size), 
                    scoring='neg_mean_squared_error', 
                    verbose=False, 
                    n_jobs=-1
                    ).mean())

                return {'loss': score, 'status': STATUS_OK}

            parameters = {
                'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(100, )]), 
                'activation' : hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu']),
                'solver' : hp.choice('solver', ['sgd', 'adam']),
                'alpha': hp.loguniform('alpha', -4, 1),
                'learning_rate' : hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive'])
                }

            best = fmin(
                objective, 
                space=parameters, 
                algo=tpe.suggest, 
                max_evals=3, 
                trials=Trials(),
                rstate=np.random.RandomState(0)
                )

            return space_eval(parameters, best)

        best_hp = mlp_hyperopt(X, y)
        
        mlp = MLPRegressor(
            **best_hp, 
            warm_start=True, 
            early_stopping=True, 
            random_state=123,
            max_iter=300
            )  
        
        mlp.fit(X, y)

        with open(save_path, 'wb') as f:
            pickle.dump(mlp, f)

    def predict(self, future, save_path='mlp.pkl'):
        '''
        Define basic prediction structure.
        
        Parameters
        ----------
        future : pandas data frame 
            Data frame with predictor features.
        
        Returns
        -------
        list(prediction) : list
            List with predictions.
        '''

        with open(save_path, 'rb') as f:
            model = pickle.load(f)

        prediction = model.predict(future)

        return list(prediction)


class KNNHyperoptTS(Trainer):
    '''
    Training k-nearest neighbors model for time series forecasting with 
    Bayesian hyperparameter optimization.
    '''

    def __init__(self):
        pass

    def fit(self, X, y, n_splits=3, max_train_size=None, test_size=None, save_path='knn.pkl'):
        '''
        Define basic training structure.

        Parameters
        ----------
        X : pandas data frame
            Data frame with predictor features.
        y : pandas series
            Pandas series with target data.
        n_splits : int
            Number of cross-validation folds. 
        test_size : int
            Size of forecasting horizon.
        '''

        def knn_hyperopt(X, y, verbose=False, persistIterations=True):
            '''
            Bayesian hyperparameter optimization with Hyperopt.
            
            Parameters
            ----------
            X : pandas data frame
                Data frame with predictor features.
            y : pandas series
                Pandas series with target data.
            
            Returns
            -------
            space_eval(parameters, best) : dict
                Best hyperparameters. 
            '''

            def objective(params):
                '''
                Function to minimize RMSE score.
                
                Paramenters
                -----------
                params : python dict
                    Hyperparamenter space to check MSE score.

                Returns
                -------
                score : float   
                    MSE score.
                '''

                params = {
                    'weights': params['weights'],
                    'p': params['p']
                }

                model = KNeighborsRegressor(**params, n_jobs=-1)   

                score = np.sqrt(-cross_val_score(
                    model, 
                    X, 
                    y, 
                    cv=TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size, test_size=test_size), 
                    scoring='neg_mean_squared_error', 
                    verbose=False, 
                    n_jobs=-1
                    ).mean())

                return {'loss': score, 'status': STATUS_OK}

            parameters = {
                'weights': hp.choice('weights', ['uniform', 'distance', 'callable']),
                'p': hp.choice('p', [1, 2])
                }

            best = fmin(
                objective, 
                space=parameters, 
                algo=tpe.suggest, 
                max_evals=3, 
                trials=Trials(),
                rstate=np.random.RandomState(0)
                )

            return space_eval(parameters, best)

        best_hp = knn_hyperopt(X, y)

        best_hp = {
            'weights': best_hp['weights'],
            'p': best_hp['p']
            }

        knn = KNeighborsRegressor(**best_hp, n_jobs=-1)  
        knn.fit(X, y)

        with open(save_path, 'wb') as f:
            pickle.dump(knn, f)

    def predict(self, future, save_path='knn.pkl'):
        '''
        Define basic prediction structure.
        
        Parameters
        ----------
        future : pandas data frame 
            Data frame with predictor features.
        
        Returns
        -------
        list(prediction) : list
            List with predictions.
        '''

        with open(save_path, 'rb') as f:
            model = pickle.load(f)

        prediction = model.predict(future)

        return list(prediction)
