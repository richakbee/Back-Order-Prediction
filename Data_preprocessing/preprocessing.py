import csv
import pickle
from datetime import datetime
import os
from sklearn.decomposition import PCA
import numpy
import pandas
import sklearn
import matplotlib.pyplot as plt

class preprocessing:
    """
            This class shall  be used to clean and transform the data before training.

            Written By: richabudhraja8@gmail.com
            Version: 1.0
            Revisions: None

            """

    def __init__(self, logger_object, file_object):

        self.file_obj = file_object
        self.log_writer = logger_object

    def remove_columns(self, data, column_list):
        """
                Method Name: remove_columns
                Description: TThis method removed the  the columns in the column list from the data.
                Output: Returns A pandas data frame.
                On Failure: Raise Exception

                Written By: richabudhraja8@gmail.com
                Version: 1.0
                Revisions: None

                """
        self.log_writer.log(self.file_obj, "Entered remove_columns of preprocessing class!!")
        try:
            new_data = data.drop(columns=column_list, axis=1)
            self.log_writer.log(self.file_obj,
                                "Column removal Successful.Exited the remove_columns method of the Preprocessor class!!")
            return new_data
        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in remove_columns method of the Preprocessor class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'removing columns from data  failed.Error in  remove_columns method of the Preprocessor class!!')
            raise e

    def separate_features_and_label(self, data, label_column_name):
        """
            Method Name: separate_features_and_label
            Description: This method separates the features and a Label Coulmns.
            Output: Returns two separate Dataframes, one containing features and the other containing Labels .
            On Failure: Raise Exception

            Written By: richabudhraja8@gmail.com
            Version: 1.0
            Revisions: None

        """
        self.log_writer.log(self.file_obj, "Entered separate_features_and_label of class preprocessing!!")
        try:
            Y = data[label_column_name]
            X = data.drop(columns=[label_column_name], axis=1)
            self.log_writer.log(self.file_obj,
                                "Label separation successful .Exited separate_features_and_label of class preprocessing!!")
            return X, Y
        except Exception as e:
            self.log_writer.log(self.file_obj,
                                "Label separation unsuccessful !!")
            self.log_writer.log(self.file_obj, "Error in  separate_features_and_label of class preprocessing %s" % e)
            raise e

    def is_null_present(self, X):
        """
                    Method Name: is_null_present
                    Description: This method takes input as dataframe . and tells if there are nulls in any column
                    Output: Returns boolean yes or no .if yes then a csv will be stored will count of null for each column
                            at location "preprocessing_data/null_values.csv". Also returns a list of column names with null values
                    On Failure: Raise Exception

                    Written By: richabudhraja8@gmail.com
                    Version: 1.0
                    Revisions: None

                """
        null_df_loc = "preprocessing_data/"
        self.log_writer.log(self.file_obj,
                            "Entered is_null_present in class preprocessing. Checking for null values in training data")
        bool_value = False
        columns_with_null = []
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        try:
            count_null = X.isnull().sum()
            for count_ in count_null:
                if count_ > 0:
                    bool_value = True
                    break
            if bool_value:
                null_df = pandas.DataFrame(count_null).reset_index().rename(
                    columns={'index': 'col_name', 0: 'no_of_nulls'})
                file_name = 'null_values_' + str(date) + "_" + str(time) + '.csv'
                dest = os.path.join(null_df_loc, file_name)
                if not os.path.isdir(dest):
                    null_df.to_csv(dest, index=False, header=True)

                # list of columns that has null values
                columns_with_null = list(null_df[null_df['no_of_nulls'] > 0]['col_name'].values)

                self.log_writer.log(self.file_obj,
                                    "Finding missing values is a success.Data written to the null values file at {}. Exited the is_null_present method of the Preprocessor class".format(
                                        null_df_loc))
            return bool_value, columns_with_null
        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in is_null_present method of the Preprocessor class. Exception message:  ' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise e

    def get_col_with_zero_std_deviation(self, X):
        """
        Method Name: get_col_with_zero_std_deviation
        Description: TThis method finds out the columns which have a standard deviation of zero.
        Output: Returns A list of all coulmns which have a standard deviation of zero.
        On Failure: Raise Exception

        Written By: richabudhraja8@gmail.com
        Version: 1.0
        Revisions: None

        """
        self.log_writer.log(self.file_obj, "Entered get_col_with_zero_std_deviation of preprocessing class!!")
        try:
            # get the standard deviation of all columns as data frame , where index is column name .
            std_columns = pandas.DataFrame(X.describe().loc['std'])
            col_list = std_columns[std_columns['std'] == 0].index.to_list()
            self.log_writer.log(self.file_obj,
                                "Column search for Standard Deviation of Zero Successful. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class!!")
            return col_list
        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in get_col_with_zero_std_deviation method of the Preprocessor class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'fetching column list with standard deviation of zero failed. get_col_with_zero_std_deviation method of the Preprocessor class!!')
            raise e

    def encode_categorical_columns(self, data, categorical_columns, yes_no_col):
        """
                        Method Name: encode_categorical_columns
                        Description: TThis method get list of  the columns which have categorical value & the data and
                                     changes the categorical columns by encoding them. also take input the list
                                     of names of columns that has yes or no in them so as to make sure
                                     yes is encoded as 1 and no as 0
                        Output: Returns data with categorical columns encoded . Save the dictionary used for mapping
                        On Failure: Raise Exception

                        Written By: richabudhraja8@gmail.com
                        Version: 1.0
                        Revisions: None

                        """
        dict_mapping_file_loc = 'preprocessing_data/'
        file_name = 'encoding_mapping_csv_file.csv'
        self.log_writer.log(self.file_obj, "Entered encode_categorical_columns of preprocessing class!!")
        yes_no_col = yes_no_col

        try:
            # create an auto generated mapping dictionary. this will be used lated in prediction too
            dict_col_encode = {}
            temp = {}
            for col in categorical_columns:
                x = data[col].unique()
                temp = dict(list(enumerate(x)))
                dict_col_encode[col] = {v: k for k, v in temp.items()}

            # ensuring yes:1 and no :0 encoded properly
            for key in yes_no_col:
                inner_dict = dict_col_encode[key]
                for inner_key in inner_dict.keys():
                    if inner_key == 'YES' or inner_key == 'Y' or inner_key == 'yes' or inner_key == 'Yes':
                        inner_dict[inner_key] = 1
                    elif inner_key == 'NO' or inner_key == 'N' or inner_key == 'no' or inner_key == 'No':
                        inner_dict[inner_key] = 0

            # saving the dictionary

            dest = os.path.join(dict_mapping_file_loc, file_name)
            if not os.path.isdir(dest):
                with open(dest, 'w') as f:  # You will need 'wb' mode in Python 2.x
                    w = csv.DictWriter(f, dict_col_encode.keys())
                    w.writeheader()
                    w.writerow(dict_col_encode)

            # using the dictionary to encode the data in data frame
            for col in categorical_columns:
                val_dic = dict_col_encode[col]
                data[col] = data[col].apply(lambda x: int(val_dic[
                                                              x]))  # by default it takes as float. will create an issue in Ml algorithms if target columns/label  is float not integer

            self.log_writer.log(self.file_obj,
                                "Encoding successful. Exited encode_categorical_columns of preprocessing class!!")
            return data

        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in encode_categorical_columns method of the Preprocessor class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'Encoding categorical features failed. Exited encode_categorical_columns method of the Preprocessor class!!')
            raise e

    def scale_numerical_columns(self, data, categorical_columns):
        """
           Method Name: scale_numerical_columns
           Description: This method get a data frame,& list of categorical columns in the daat & finds the numerical columns and scale them
                        using standard scaler() of preprocessing from sklearn.
                        the scaler is saved at location  'preprocessing_data/standardScaler.pkl' for use in prediction.
           Output: Returns data with numerical columns scaled .
           On Failure: Raise Exception

           Written By: richabudhraja8@gmail.com
           Version: 1.0
           Revisions: None

        """
        scaler_path = 'preprocessing_data/StandardScaler.pkl'
        self.log_writer.log(self.file_obj, "Entered scale_numerical_columns of preprocessing class!!")
        try:
            # find the numerical columns
            numerical_col_list = list(numpy.setdiff1d(list(data.columns), categorical_columns))
            # get data frames for tarin & test
            df_num = data[numerical_col_list]

            # define standard scaler object
            scaler = sklearn.preprocessing.StandardScaler()

            # fitting the scaler on data set
            scaler.fit(df_num)
            # saving the scaler obj
            pickle.dump(scaler, open(scaler_path, 'wb'))

            # transform data set
            df_scaled = scaler.transform(df_num)

            # scaled data is a array convert back to data frame
            Scaled_df = pandas.DataFrame(df_scaled, columns=df_num.columns.tolist(), index=df_num.index)

            self.log_writer.log(self.file_obj,
                                "Scaling numerical columns successful.Exited scale_numerical_columns of preprocessing class!!")
            return Scaled_df

        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in scale_numerical_columns method of the Preprocessor class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'Scaling numerical columns failed. Exited scale_numerical_columns method of the Preprocessor class!!')
            raise e

    def impute_Categorical_values(self, data, columns_with_null):
        """
                        Method Name: impute_Categorical_values
                        Description: TThis method get list of  the columns which have categorical value & has nulls
                                     cin the columns. it imputes the null with mode of the column
                        Output: Returns data with imputed value inplace of nulls
                        On Failure: Raise Exception

                        Written By: richabudhraja8@gmail.com
                        Version: 1.0
                        Revisions: None

                        """
        self.log_writer.log(self.file_obj, "Entered impute_Categorical_values of preprocessing class!!")
        try:
            for col in columns_with_null:
                # only for category columns in list of columns that has null values , fill them with mode
                if ((data[col].dtypes) == 'object') or ((data[col].dtypes) == 'category'):
                    data[col].fillna(data[col].mode().values[0], inplace=True)

            self.log_writer.log(self.file_obj,
                                "imputing null value in categorical features successful. Exited the impute_Categorical_values method of the Preprocessor class!!")
            return data

        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in impute_Categorical_values method of the Preprocessor class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'imputing null value in categorical features  failed. impute_Categorical_values method of the Preprocessor class!!')
            raise e

    def one_hot_encode_cagtegorical_col(self, data, categorical_features):

        df_cat = data[categorical_features].copy()
        for col in categorical_features:
            df_cat = pandas.get_dummies(df_cat, columns=[col], prefix=[col], drop_first=True)

        return df_cat

    def pcaTransformation(self, X_num_scaled,pca_num_components):
        """
               Method Name: pcaTransformation
               Description: TThis method get dataframe of numerical columns & performs linear
                            PCA transformation with given value  for number of components
               Output: Returns data with original number of rows, no of columns= no of pca components
                        saves the explainer_variance graph,scree plot & importance of features graph
                        @ 'preprocessing_data/pca/'
               On Failure: Raise Exception

               Written By: richabudhraja8@gmail.com
               Version: 1.0
               Revisions: None

        """
        explained_variance_plot_loc= 'preprocessing_data/pca_explained_variance.png'
        scree_plot_loc = 'preprocessing_data/pca_scree_plot.png'
        feature_importance_plot_loc = 'preprocessing_data/feature_importance_plot.png'
        pca_obj_loc = 'preprocessing_data/pca.pkl'
        self.log_writer.log(self.file_obj, "Entered pcaTransformation of preprocessing class!!")
        try:
            # explained variance plot
            pca = PCA()
            principalComponents = pca.fit_transform(X_num_scaled)
            plt.figure()
            plt.plot(numpy.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('Number of Components')
            plt.ylabel('Variance (%)')  # for each component
            plt.title('Explained Variance')
            plt.savefig(explained_variance_plot_loc)
            self.log_writer.log(self.file_obj,"saved explained variance plot")

            #scree plot
            plt.figure()
            PC_values = numpy.arange(pca.n_components_) + 1
            plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
            plt.title('Scree Plot')
            plt.xlabel('Principal Component')
            plt.ylabel('Proportion of Variance Explained')
            plt.savefig(scree_plot_loc)
            self.log_writer.log(self.file_obj, "saved scree plot")

            #feature importance plot
            plt.figure(figsize=(20, 5))
            plt.bar(X_num_scaled.columns, pca.explained_variance_)
            p = plt.xticks(rotation=45)
            plt.savefig(feature_importance_plot_loc)
            self.log_writer.log(self.file_obj, "saved feature importance plot")

            #save the pca object for use in later
            pca = PCA(n_components=pca_num_components)
            new_data = pca.fit_transform(X_num_scaled)

            pickle.dump(pca, open(pca_obj_loc, 'wb'))

            column_names = ['PC-'+str(i+1) for i in range(pca_num_components)]
            X_num_pca = pandas.DataFrame(new_data,columns=column_names)

            self.log_writer.log(self.file_obj,
                                "PCA transformation successful. Exited the pcaTransformation method of the Preprocessor class!!")
            return X_num_pca

        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in pcaTransformation method of the Preprocessor class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'PCA transformation  failed. pcaTransformations method of the Preprocessor class!!')
            raise e
        pass

    def transform_log(self,data, log_df_col):
        """
            Method Name: transform_log
            Description: TThis method get dataframe & list of columns of data frame that needs
                        to have log 1P tranformation done
            Output: Returns data frame with log 1p transformations in the required columns
            On Failure: Raise Exception

            Written By: richabudhraja8@gmail.com
            Version: 1.0
            Revisions: None

                """

        self.log_writer.log(self.file_obj, "Entered transform_log of preprocessing class!!")
        try:
            #creates log transformed columns with name as the nome of column with a '_log' suffix
            #eg: column name 'a' ,a new columns with name 'a_log' is created
            for col in log_df_col:
                data[col] = numpy.log(1 + data[col])

            data.replace([numpy.inf, -numpy.inf], numpy.nan, inplace=True)
            data.dropna(inplace=True)
            self.log_writer.log(self.file_obj,
                                "log transformation on selected features successful. Exited the transform_log method of the Preprocessor class!!")
            return data

        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in transform_log method of the Preprocessor class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'log transformation on selected features  failed. transform_log method of the Preprocessor class!!')
            raise e

