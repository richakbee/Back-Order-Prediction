import numpy
import sklearn
from Best_model_finder.tuner import tuner


class model_finder:
    def __init__(self,  logger_object,file_object):

        self.file_obj = file_object
        self.log_writer = logger_object
        self.tuner = tuner(file_object, logger_object)

    def get_best_model(self,x_train, y_train, x_test, y_test):
        """
           Method Name: get_best_model
           Description: Find out the Model which has the best AUC score.
           Output: The best model name and the model object
           On Failure: Raise Exception

           Written By: richabudhraja8@gmail.com
           Version: 1.0
           Revisions: None

         """
        self.log_writer.log(self.file_obj, "entered get_best_model method in class model_finder!! ")
        try:

            #step 1 define all the names of model to be tested
            model_names = ["cat_boost", "random_forest", "logistic_reg", "lightgbm"]
            models=[]

            self.log_writer.log(self.file_obj, "making calls to functions for get_best_params for each model in tuner class !! ")
            print(len(x_train),len(y_train))
            models.append(self.tuner.get_params_for_catboost( x_train, y_train))
            models.append(self.tuner.get_params_for_random_forest(x_train, y_train))
            models.append(self.tuner.get_params_for_logistic_reg(x_train, y_train))
            models.append(self.tuner.get_params_for_lightgbm( x_train, y_train))

            # models = [eval(x) for x in model_functions_cal]
            self.log_writer.log(self.file_obj, "Calls to functions for get_best_params for each model in tuner class ended successfully!! ")

            #step4 use each model to predict on x_test to get y_pred
            y_pred=[]
            for model in models:
                y_pred.append(model.predict(x_test))

            #step 5 for each model calculate the auc score /accuracy
              #step 5.1 if theres is just one label in y_test then use accuracy else use roc_auc_score
            scores=[]
            if (len(y_test.unique())==1):
                for y_pred_ in y_pred:
                    scores.append(sklearn.metrics.accuracy_score(y_test, y_pred_))
            else:
                for y_pred_ in y_pred:
                    scores.append(sklearn.metrics.roc_auc_score(y_test, y_pred_))

            #step 6 compare the scores for each model in scores list and get the index for best score and return the model at that index

            index = numpy.argmax(scores)
            best_model_name = model_names[index]
            best_model = models[index]

            self.log_writer.log(self.file_obj, "Exited the get_best_model method of the Clustering class. ")

            return best_model_name, best_model

        except Exception as e:
            self.log_writer.log(self.file_obj, "Error in  the get_best_model method of the Clustering class. %s" % e)
            raise e