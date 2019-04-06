# -*- coding: utf-8 -*-
"""
Moody's Analytics Proxy Generator Prototype

This script shows the implementation of a Multi Layered Perceptron regressor
with the aid of SciKit Learn (sklearn) and Pandas.

"""

# Pandas will be used for data storage and manipulation through dataframes
import pandas as pd

# SciKit Learn provides the model we shall use as well as the train_test_split
# tool which enables us to randomly partition data for testing and training 
# purposes
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# The plotting tools we use to display outputs. Matplotlib, a staple of data
# visualisation in Python and Seaborn a package used to beautify our plots.
# The 3D plotting tools are also imported from matplotlib to enable more 
# complex plotting.
# The 3D tools are used implicitly in the background so some IDEs may note that
# it is not used since there is no explicit reference. Note that removal of 
# this import will cause the 3D plotting to fail.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn

# Numpy (np) is used for numerical manipulation and the vectorisation of 
# calculations. It is imported itself as part of Pandas and hence the 
# naming is reassigned for ease of use.
np = pd.np

# Applying Seaborn's styling to our plots.
seaborn.set()

# GLOBAL CONSTANTS
#==============================================================================
# Path to the original data set to use for training. Must be a csv file.
ORIGINAL_DATA_FILEPATH = r'C:\Users\DaviesI\Documents\Misc\data.csv'

# The column title to use as an index. Set to none if no column can act as an
# index. In this case pandas will simply set up a numeric 0 based index.
INDEX_COLUMN = 'Scenario'

# The variable we wish to predict
TARGET_VARIABLE = 'BEL_CE'

#==============================================================================

def get_data(random_state=None, sample_size=0.9, normalise=False):
    '''
    Function for getting and returning the data set in a Pandas DataFrame.
    The data source is hard coded as this script is designed to be single use.
    
    get_data(random_state=None, sample_size=0.5, normalise=False)
    returns data, training_data, test_data, unscaled_data
    
    Arguments
    random_state: The random state used for subsampling the data into testing 
    and training data sets. Defaults to None so that no specific sampling 
    initialisation is imposed.
    
    sample_size: The proportion of data used in forming the training set. The 
    complement to the proportion of data used in forming the test set. Defaults
    to 0.9. Must be between 0 and 1.
    
    normalise: Boolean paramter that if true means that the data will be
    normalised to standard z values. If true data will be normalised with the
    raw data available through unscaled_data. If false the outputs scaled_data
    and unscaled_data will be the same.
    
    Returns
    data: Pandas data frame containing the data set read from the csv file. 
    Normalised if the input normalise is set to True.
    
    training_data: Pandas dataframe containing the subsample of data to be used
    for training the model.
    
    test_data: Pandas data frame containing the subsample of data to be used
    for testing the model on unseen examples.
    
    unscaled_data: Pandas data frame containing the raw data (before any 
    normalisation).
    '''
    # Read in the data set setting the index to the scenario column
    data = pd.read_csv(ORIGINAL_DATA_FILEPATH, index_col=INDEX_COLUMN)
    # Taking a deep copy of the original data so that when normalisation is 
    # applied the original data is retained in unscaled_data.
    unscaled_data = data.copy()
    
    # Aplpying the normalisation if required
    if normalise:
        data = data.apply(lambda x: (x - np.mean(x)) / np.std(x))
    
    # Splitting the data after normalisation is applied (if needed) for 
    # testing and training purposes using the random state and sample size 
    # passed to get_data.
    training_data, test_data = train_test_split(data, 
                                                test_size = 1-sample_size,
                                                random_state=random_state) 
    
    return data, training_data, test_data, unscaled_data


def generate_and_test_NN(test=False):
    '''
    Builds and trains the neural network regressor using the 
    paramterisation determined though our research and testing.
    
    If the boolean input test is true a test function, test_nn, is called to 
    calculate the errors of the prediction on the test set, plots their 
    distribution and provides some metrics concerning goodness-of-fit including 
    r-squared, average absolute error and maximum absolute error. 
    
    The call to the test function also generates plots of each 
    feature and its relationship with the target variable (BEL_CE).
    
    Arguments
    test: Boolean flag to determine wether or not to test the neural network 
    before returning it to the user.
    
    Returns
    nn: The trained multi-layered perceptron regressor as designed with four
    layers. The input layer (13 inputs), hidden layer #1 (30 neurons), hidden
    layer #2 (100 neurons), output layer (1 output).
    
    '''
    
    # First the data is attained. Normalised data is used for training the 
    # in order to improve the computational and fitting performance of the 
    # model. If one tries to use the raw data training will likely fail.
    # The unscaled data is retained in order to undo normalisation scaling back
    # to the original data set if required.
    
    # Values in the accompanying methodology document use a random state of 76.
    data, training_data, test_data, unscaled_data = get_data(normalise=True)
    
    # Generating the model using the object imported from scikit learn
    # hidden_layer_sizes = (30, 100)
    # The network uses two hidden layers of 30 neurons and 100 neurons for 
    # the first and second hidden layers repsectively.
    
    # solver = 'lbfgs'
    # The training is performed using the limited-memory 
    # Broyden–Fletcher–Goldfarb–Shanno algorithm (lbfgs) as defined in scikit 
    # learn. [ See: https://en.wikipedia.org/wiki/Limited-memory_BFGS ]
    
    # alpha = 0.05124
    # The L2 smoothing parameter alpha is set as the result of many trials in 
    # optimising the fit to the data.
    
    # max_iter = 10000
    # The trainer is given a budget 10,000 iterations meaning that it will run
    # a maximum of 10,000 iterations terminating at the first of the optimum
    # being found or all 10000 runs being exhausted.
    
    # tol = 0
    # The learning algorithm takes an input to determine when to terminate on 
    # on the basis of continued iterations not providing sufficient benefit.
    # This tollerance is set to 0 so that the algorithm only halts when 
    # continued learning no longer yields progress. Hence termination will 
    # occur at the sooner of the reaching of the 10,000 epoch limit or the loss
    # function failing to fall for two consecutive epochs.    
    
    # Values in the document use random state 76
    nn = MLPRegressor(hidden_layer_sizes=(30,100), 
                      solver='lbfgs', 
                      alpha=0.05124, 
                      max_iter=10000, 
                      tol=0,
                      random_state=76)
    
    
    # Having initialised the model with parameters it is given data to fit.
    # This method trains the network calculating the weights in each layer.
    
    nn.fit(training_data.drop(TARGET_VARIABLE, axis=1), 
                                               training_data[TARGET_VARIABLE])

    # Run tests on predictive power and error distributions if requested
    if test: test_nn(nn, training_data, test_data, data, unscaled_data)

    # Return the trained neural network
    return nn

def test_nn(neuralnet, training_data, test_data, data, unscaled_data, normalised=True):
    '''
    Performs analysis on a neural network testing goodness-of-fit in- and out-
    of-sample as well as plotting relationships between the regressors and the 
    target variable.
    
    It is assumed that the neural network has been trained using normalised
    data as this improves performance. In the case that raw data is used the
    unscaled_data parameter is still required.
    
    Arguments
    neuralnet: Trained neural network (sklearn MLPRegrssor object) to be tested
    and evaluated.
    
    training_data: Pandas dataframe containing the data used to train the 
    neural network being tested.
    
    test_data: Pandas dataframe containing the data to be used for
    out-of-sample testing.
    
    data: Pandas datafram containing the entire data set (the union of the test
    and training sets) used for considering the errors across in- and 
    out-of-sample tests.
    
    unscaled_data: Pandas dataframe containing the entire data set before any
    normalisation transformation is applied. Used for undoing normalisation to
    attain errors on the scale of the original data set.
    
    normalised: Boolean flag reflecting whether or not the neural network was 
    trained on normalised data (and therefore whether the data and 
    unscaled_data inputs differ) or not. This is used to determine whether the
    errors need to be rescaled to be in line with the original data set. 
    Defaults to True reflecting the assumption that normalised data is used due
    it the performance improvements this yields.
    
    Returns
    None
    '''
    
    # Use the score method of the MLPRegessor object to attain an R-squared
    # value for both the training and test set of data in turn to attain the
    # in-sample R-Squared and out-of-sample R-squared respectively.
    in_sample_r2 = neuralnet.score(training_data.drop(TARGET_VARIABLE,axis=1), 
                                   training_data[TARGET_VARIABLE])
    
    out_of_sample_r2 = neuralnet.score(test_data.drop(TARGET_VARIABLE,axis=1), 
                                       test_data[TARGET_VARIABLE])
    
    # Printing out out the results of the initial goodness-of-fit tests.
    print('In-Sample Neural Net R-Squared:\t\t', in_sample_r2)
    print('Out-of-Sample Neural Net R-Squared:\t', out_of_sample_r2)

    # Using the predict method of the neural network (MLPRegressor object) to 
    # attain the predicted values from which errors can be derived
    predictions = neuralnet.predict(data.drop(TARGET_VARIABLE, axis=1))
    
    # Undoing the normalisation process using the unscaled data in the case 
    # that normalised data was used in training the neural network therefore
    # resulting in normalised output.
    if normalised:
        predictions = (predictions * unscaled_data[TARGET_VARIABLE].std()) \
                                    + unscaled_data[TARGET_VARIABLE].mean()
    
    # Calculating the errors as the difference between the predictions and the 
    # true values.
    errors = predictions - unscaled_data[TARGET_VARIABLE]
    
    # Getting error quantiles
    upper_quartile_abs, median_error_abs, lower_quartile_abs = \
                    (pd.np.percentile(errors.abs(), q) for q in (75, 50, 25))
    
    upper_quartile, median_error, lower_quartile = \
                    (pd.np.percentile(errors, q) for q in (75, 50, 25))
    
    
    # Plot the distribution of the errors using the function defined below
    plot_histogram(errors)
    
    # Finding the maximal absolute error by first taking the absolute values of
    # the errors and then finding the maximum amongst them
    max_error_abs = errors.abs().max()
    min_error_abs = errors.abs().min()
    
    max_error = errors.max()
    min_error = errors.min()
    
    # Round the error up to the cloest integer and print it with comma 
    # thousands separators used to aid readability.
    print('\nMax Absolute Error: {:,}'.format(int(np.ceil(max_error_abs))))
    print('Upper Quartile of Absolute Errors: {:,}'\
                                  .format(int(np.ceil(upper_quartile_abs))))
    print('Median Absolute Error: {:,}'.format(int(round(median_error_abs))))
    print('Lower Quartile of Absolute Errors: {:,}'\
                                  .format(int(np.ceil(lower_quartile_abs))))
    print('Minimum Absolute Error: {}\n'.format(int(np.floor(min_error_abs))))
    

    print('\nMax Error: {:,}'.format(int(np.ceil(max_error))))    
    print('Upper Quartile Error: {:,}'.format(int(np.ceil(upper_quartile))))
    print('Median Error: {:,}'.format(int(round(median_error))))
    print('Lower Quartile of Errors: {:,}'\
                                      .format(int(np.ceil(lower_quartile))))
    print('Minimum Error: {:,}\n'.format(int(np.floor(min_error))))
    
    # Taking the average (mean) absolute error and printing it using the same
    # formatting as for the maximum absolute error.
    avg_abs_err = errors.abs().mean()
    root_mean_squared_error = np.sqrt(np.square(errors).mean())
    print('Average Absolute Error: {:,}'.format(int(np.ceil(avg_abs_err))))
    print('Root Mean Squared Error: {:,}'\
                              .format(int(np.ceil(root_mean_squared_error))))
    data_with_errors = unscaled_data.copy()
    data_with_errors['Errors'] = errors
    plot_error_charts(data_with_errors)

def plot_histogram(series):
    '''
    Plots a histogram to visualise the distribution of a series of data.
    No return value since this is a utility to plot distributions rather than
    performing any data manipulation.
    
    Arguments
    series: Pandas series, array, list, pandas dataframe with one feature - the
    data for which the distribution of the data points is to be plotted.
    
    Returns
    None
    '''
    
    # Using matplotlib's histogram plotting function splitting the data into
    # 50 equally sized bins to provide a sufficiently detailed plot of the data
    plt.hist(series, bins=50, figure=plt.figure(figsize=(14,8)))
    
    # Labelling the axes and providing the figure with a title
    plt.xlabel('Error')
    plt.ylabel('Frequency Density')
    plt.title('Distribution of Errors')
    
    # Displaying the plot in the console to end this plotting series.
    # This frees up plotting tools for a new plot to be generated without 
    # plotting on top of the existing data.
    plt.show()

def interaction_plots(nn):
    '''
    Plots the interaction of each regressor with the output variable for the 
    range of the inputs observed in the original data set. This provides one 
    plot per input variable looking at the impact when all other variables are
    held at their average (mean) value.
    
    This uses the get_data function above to pull in data so that the user need
    only set up data locations once.
    
    There is no return value as this is designed as a data visualisation 
    utility and does not materially manipulate the data for future alternate 
    uses.
    
    Arguments
    nn: Sci-Kit learn MLPRegressor object - The trained neural network 
    regression model. This must have been trained on normalised data.
    
    Returns
    None
    '''
    
    # Pull in the data. As we do not need the training or test set for plotting
    # these vales are ignored (hence the _ in their place).
    # The normalised data is taken as this is what the network uses for 
    # prediction however the original raw data is also kept to enable mapping 
    # back to the original scales.
    data, _, _, raw_data = get_data(None, normalise=True)
    
    # Calculating some meausures of the distributions of the inputs to aid us
    # in generating plots true to the observed data ranges.
    # The average values are used to hold variables not being plotted at their
    # average to distill the impact of the single variable of interest.
    means = data.mean()
    minima = data.min()
    maxima = data.max()

    # Loop through the input data (i.e. the data less the final column which 
    # constitutes the desired output) and construct a new dataframe one
    # variable at a time.
    # A plot is generated for each input variable.
    for var in data.columns[:-1]:
        # For each input variable...
        
        # Initialise a new dataframe
        dataset = pd.DataFrame()
        
        # For each input variable...
        for col in data.columns[:-1]:
            # If the variable is not the one we wish to plot the impact of then
            # set the variable equal to its mean in the master data set for all
            # observations.
            if col != var:
                dataset[col] = np.full(1000, means[col])

            # If however the variable is the one which we are plotting then set
            # the variable to a 1,000 step range from its observed minimum to
            # its observed maximum. This then enables us to see output 
            # interactions across the entire observed range.
            # This range is the domain for our plot and is labeled as such for
            # use in plotting.
            else:
                domain = np.linspace(minima[var], maxima[var], 1000)
                dataset[col] = domain
                
        # Having constructed our data set holding all variables at the average 
        # of observations we have of them except the variable for which we aim 
        # to plot the impact of (letting this variable vary across its observed
        # range) we may use our model to predict outcomes in all cases we have 
        # constructed. This therefore gives us the predicted outcomes only 
        # influenced by the variable of interest (var).
        output = nn.predict(dataset)
        
        # Undoing normalisation of the variable of interest and the output in 
        # order to get back on terms with the original data set for a more 
        # meaningful plot to be generated.
        domain = raw_data[var].mean() + (domain * raw_data[var].std())
        output = raw_data[TARGET_VARIABLE].mean() \
                                + (output * raw_data[TARGET_VARIABLE].std())
        
        # Plot the relationship between the variable of interest (var) and the
        # target variable as predicted in our modelling.
        plt.plot(domain, output)
        
        # Entitle the plot by the variable under consideration for 
        # identification of each plot.
        plt.title(var)
        
        # Display the plot to the user and therefore free up plotting tools for
        # the next plot to begin.
        plt.show()

def threeDPlot(nn, var1, var2):
    '''
    Plots the interaction of two variables (var1, var2) with the output 
    variable of the model generating a 3D surface plot of the outcomes.
    
    Initially designed for use in Jupyter notebooks where the plot generated is
    interactive enabling rotation and zooming in. When run outside of a Jypter
    notebook a static (png) image is displayed.
    
    Arguments
    nn: Sci-Kit learn MLPRegressor object - The trained neural network 
    regression model. This must have been trained on normalised data.
    
    var1: String - The first input variable to be allowed to vary creating the 
    X axis of the resulting plot.
    
    var2: String - The second input variable to be allowed to vary creating the 
    Y axis of the resulting plot.
    
    Returns 
    None
    '''
    
    # Pull in the data. As we do not need the training or test set for plotting
    # these vales are ignored (hence the _ in their place).
    # The normalised data is taken as this is what the network uses for 
    # prediction however the original raw data is also kept to enable mapping 
    # back to the original scales.
    data, _, _, raw_data = get_data(None, normalise=True)
    
    # Calculating some meausures of the distributions of the inputs to aid us
    # in generating plots true to the observed data ranges.
    # The average values are used to hold variables not being plotted at their
    # average to distill the impact of the single variable of interest.
    means = data.mean()
    minima = data.min()
    maxima = data.max()
    
    # Setting up the x and y variable domains by taking the maximimum and 
    # and minimum of each variable and generating a series of 1,000 equally 
    # spaced points between them for var1 and var2.
    x_domain = np.linspace(minima[var1], maxima[var1], 1000)
    y_domain = np.linspace(minima[var2], maxima[var2], 1000)
    
    # In order to generate a three dimensional plot a coordinate system needs 
    # to be set up so that for a pair of value (x, y) the vertical z value can
    # be determined and plotted in 3D space at point (x, y, z). 
    # Generating a meshgrid does this for us. If our x values were [1,2,3] and 
    # our y values [0.1, 0.2, 0.3] meshgrid(x, y) would result in two 2D arrays
    # X = [[1,2,3], [1,2,3], [1,2,3]] 
    # Y = [[0.1,0.1,0.1], [0.2,0.2,0.2], [0.3,0.3,0.3]] 
    # This then enables us to go through each point in 3D space and pick out
    # an x value from X and the corresponging y value from Y to then get the z
    # value and plot it. This is enabling iteration for each y value over all
    # x values in an exhaustive fashion without creating nested loops.
    X, Y = np.meshgrid(x_domain, y_domain)

    # To convert our data from two dimensional arrays to the one dimensional 
    # dataframe column (a Pandas Series to be exact) we must chain the rows one
    # after another to attain single lists such that we can read down the 
    # columns to iterate thorough each pair of x and y.
    # e.g. ravel([[1,2,3], [0.1,0.2,0.3]]) => [1, 2, 3, 0.1, 0.2, 0.3]
    x_values = np.ravel(X)
    y_values = np.ravel(Y)
    
    # Undoing the normalisation of the x and y values in order to be able to 
    # use them for plotting where the values are plotted on the scale of the
    # original data. The outcome of this transformation is then shaped back to
    # be suitable for 3D plotting.
    X = (raw_data[var1].mean() + (x_values * raw_data[var1].std())).reshape(X.shape)
    Y = (raw_data[var2].mean() + (y_values * raw_data[var2].std())).reshape(Y.shape)
    
    # Initialising a new dataframe in order to store the data set constructed 
    # by setting values outside of those of interest (var1 and var2) to their 
    # averages.
    dataset = pd.DataFrame()
    
    # For each variable...
    for col in data.columns[:-1]:
        # If the variable is not one of those we wish to vary. Set it equal to 
        # the mean from the normalised data for all overservations.
        if col not in (var1, var2):
            dataset[col] = np.full(len(x_values), means[col])
        
        # Otherwise if the variable is the first of the two variables allowed
        # to vary then set it to the previously made list of x values to use in
        # plotting.
        elif col == var1:
            dataset[col] = x_values
        # Otherwise it must be the second variable we are allowing to vary and 
        # hence set this column equal to the previously made list of y values
        # to use in plotting.
        else:
            dataset[col] = y_values
    
    # The z values for our plot are the output values (z = f(x, y)) and hence 
    # we may find them using the predict method of our neural network 
    # regression model.
    Z = nn.predict(dataset)
    
    # The predcicted values must be rescaled in order to place them back into
    # terms of the scale of the original data set.
    Z = raw_data[TARGET_VARIABLE].mean() + (Z * raw_data[TARGET_VARIABLE].std())
    
    # The Z values array must be reshaped for plotting. We do this by reshaping
    # it to match the shape of the X values.
    Z = Z.reshape(X.shape)
    
    # Initialise a figure in which to build the 3D plot
    fig = plt.figure(figsize=(15,6))
    
    # Add a subplot (the only plot) in position 111 and make it a 3D plot. 
    # Note: This is where the Axes3D object imported is implicitly called.
    ax = fig.add_subplot('111', projection='3d')
    
    # Plot the surface of the predicted values 
    ax.plot_surface(X, Y, Z, color='blue', shade=True, alpha=0.95, linewidth=0)
    
    # Some set up to avoid illegible labels when rotating the axes in the 
    # Jupyter notebook interactive charts.
    ax.zaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)

    # Label the axes.
    ax.set_xlabel(var1, rotation=0, labelpad=20)
    ax.set_ylabel(var2, rotation=0, labelpad=20)
    ax.set_zlabel(TARGET_VARIABLE, rotation=0, labelpad=10)
    
    # Displaying the plot and releasing the plot object for any new charts to 
    # start afresh.
    plt.show()
    
def apply_to_new_data(neuralnet, file_path, save_path):
    '''
    Use the previously calibrated neural network regressor (neuralnet) on a new
    data set in order to generate predictions.
    
    Arguments
    neuralnet: SciKit-Learn trained MLPRegressor object - The trained neural
    network model to be applied to the new data set.
    
    file_path: String - The file path of the new dataset (in csv format).
    
    save_path: String - The file path to save the completed data set (i.e. 
    with predictions). The data will be saved in csv format. This file path
    need not already exist the program will generate files and folders as 
    necessary.
    
    Returns
    new_data: Pandas dataframe containing all of the original data with 
    predictions appended in a final column.
    '''
    
    # Pulling in the original data set used to train the neural network so that
    # the new data may be normalised and transformed back using the same 
    # values.
    # The transformed data and the training and test sets are ignored (hence 
    # the _ assignments) since we only need the raw data for transformations
    # and the column titles.
    _, _, _, raw_data = get_data()
    
    # Reading in the new data set on which to apply our regression model.
    # This data set should have the input varibles with the same names (column
    # titles) as the original data used to train the model as the model is 
    # expected to work in the same way.
    new_data = pd.read_csv(file_path, index_col=INDEX_COLUMN)
    
    # Attaining a list of column titles from the original data set and then 
    # using this list to rearrange the new data to make it suitable for use in
    # our regression model. Through pandas's column indexing system.
    columns = list(raw_data.columns[:-1])
    new_data = new_data[columns]
    
    # Saving the means and standard deviations for each variable in the 
    # raw_data to enable transformation and reverse transformation of the new
    # data set. This results in two pandas Series means and stds.
    means = raw_data.mean()
    stds = raw_data.std()
    
    # Deep copying the new data set so that it may be normalised without 
    # fluencing the original data. The scaled data will be normalised and used
    # in preidction and the original dataset retained to append the scaled 
    # results by way of completing the dataset.
    scaled_data = new_data.copy()
    
    # For each variable apply the normalisation using the mean and standard
    # deviation of the raw_data used to train the model. It is important to
    # scale using the same process as was used on the training data in order 
    # for our neural network to be applied evenly. This is why the means and
    # standard deviations of the original dataset were calculated earlier.
    for col in columns:
        scaled_data[col] = (new_data[col] - means[col])/stds[col]
    
    # Apply the neural network regression to the scaled data to attain our
    # predictions.
    predictions = neuralnet.predict(scaled_data)
    
    # Rescale the predictions to the original scale of the output variable 
    # observed in the raw (unscaled) training data and then append the scaled
    # output values to the new dataset.
    new_data[TARGET_VARIABLE] = (predictions * stds[TARGET_VARIABLE]) \
                                                    + means[TARGET_VARIABLE]
    
    # Save the completed output data to a csv file for reuse
    new_data.to_csv(save_path)
    
    # Return the complete dataset with rearranged columns and prediction 
    # values.
    return new_data    

def plot_error_charts(data_we, threeD_x='IR_PC1_EUR', threeD_y='Idx_Bond'):
    for col in data_we.columns[:-2]:
        plt.scatter(data_we[col], data_we['Errors'])
        plt.xlabel(col)
        plt.ylabel('Estimation Error')
        plt.title(col + ' vs. Error')
        plt.show()

#    x = data_we[threeD_x]
#    y = data_we[threeD_y]
#    X, Y = np.meshgrid(x, y)
#    Z = np.zeros_like(X)
#    for i in X.shape[0]:
#        for j in X.shape[1]:
#            Z[i,j] = data_we['Errors']
#    
#    fig = plt.figure(figsize=(15,6))
#    
#    # Add a subplot (the only plot) in position 111 and make it a 3D plot. 
#    # Note: This is where the Axes3D object imported is implicitly called.
#    ax = fig.add_subplot('111', projection='3d')
#    
#    # Plot the surface of the predicted values 
#    ax.plot_surface(X, Y, Z, color='blue', shade=True, alpha=0.95, linewidth=0)
#    
#    # Some set up to avoid illegible labels when rotating the axes in the 
#    # Jupyter notebook interactive charts.
#    ax.zaxis.set_rotate_label(False)
#    ax.yaxis.set_rotate_label(False)
#
#    # Label the axes.
#    ax.set_xlabel(threeD_x, rotation=0, labelpad=20)
#    ax.set_ylabel(threeD_y, rotation=0, labelpad=20)
#    ax.set_zlabel('Estimation Error', rotation=0, labelpad=10)
#    
#    # Displaying the plot and releasing the plot object for any new charts to 
#    # start afresh.
#    plt.show()
    

# If this file is being used as a script rather than as a modular set of 
# functions to be imported then generate and test the neural network, plot a 
# series of charts analysing the output and generate a three dimension plot of 
# the first principal component and the bond index (in this case update string
# inputs to generate new plots).
if __name__ == '__main__':
    neuralnet = generate_and_test_NN(test=True)
    interaction_plots(neuralnet)
    threeDPlot(neuralnet, 'IR_PC1_EUR', 'Idx_Bond')
    
    

    