from datetime import datetime
from sklearn.model_selection import train_test_split
from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator

import numpy as np
import pandas as pd


def load_data(filename: str, have_true_val: bool = True):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename).drop_duplicates()

    # TODO's
    # create num from order date

    features = full_data[[
        "no_of_children",
        "no_of_extra_bed",
        "no_of_room",
        "guest_is_not_the_customer",
        "original_selling_amount",
        "no_of_adults",
    ]]

    features = pd.concat(
        [
            pd.get_dummies(full_data["charge_option"],
                           columns=["charge_option"]),

            pd.get_dummies(full_data["customer_nationality"],
                           columns=["customer_nationality"]),

            pd.get_dummies(full_data["accommadation_type_name"],
                           columns=["accommadation_type_name"]),

            pd.get_dummies(full_data["hotel_country_code"],
                           columns=["hotel_country_code"]),

            pd.get_dummies(full_data["original_payment_method"],
                           columns=["original_payment_method"]),

            pd.get_dummies(full_data["cancellation_policy_code"],
                           columns=["cancellation_policy_code"]),

            pd.get_dummies(full_data["hotel_star_rating"],
                           columns=["hotel_star_rating"]),

            # FIXME work, but chek later how to convert bool to 1/0
            pd.get_dummies(full_data["is_first_booking"],
                           columns=["is_first_booking"], drop_first=True),

            pd.get_dummies(full_data["is_user_logged_in"],
                           columns=["is_user_logged_in"], drop_first=True),

            features], axis=1)

    features["no_order_days"] = full_data.apply(
        extract_days_diff_between_str_date,
        axis=1, args=("checkout_date", "checkin_date"))

    features["no_before"] = full_data.apply(
        extract_days_diff_between_str_date,
        axis=1, args=("checkin_date", "booking_datetime"))

    labels = []
    if have_true_val:
        labels = full_data["cancellation_datetime"].apply(
            lambda x: 0 if pd.isnull(x) else 1)

    return features, labels


# first - second
def extract_days_diff_between_str_date(data_row, first_data_name,
                                       second_date_name):
    d1 = datetime.strptime(data_row[first_data_name][:10], '%Y-%m-%d')
    d2 = datetime.strptime(data_row[second_date_name][:10], '%Y-%m-%d')

    return (d1 - d2).days


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray,
                        filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X),
                 columns=["predicted_values"]).to_csv(
        filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data(
        "../datasets/agoda_cancellation_train.csv")

    # Store model predictions over test set
    test_set_week = load_data("test_set_week_1.csv", False)[0]

    for i in range(1):
        # Get missing columns in the training test
        df, test_set_week = df.align(test_set_week, join='outer', axis=1,
                                     fill_value=0)

        train_X, test_X, train_y, test_y = \
            train_test_split(df, cancellation_labels, test_size=0.001)

        # Fit model over data
        estimator = AgodaCancellationEstimator().fit(train_X.to_numpy(),
                                                     train_y.to_numpy())

        # FIXME print
        print(estimator.loss(test_X.to_numpy(), test_y.to_numpy()))
        print((train_X.size, train_y.size))
        print((test_X.size, test_y.size))

        evaluate_and_export(estimator, test_set_week.to_numpy(),
                            "209381284_211997275_318164886.csv")
