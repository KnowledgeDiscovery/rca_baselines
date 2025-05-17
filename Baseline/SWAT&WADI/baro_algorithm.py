import warnings

warnings.filterwarnings("ignore")
import pandas
import numpy as np
from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as ss
from itertools import islice
from numpy.linalg import inv
from functools import partial
import pandas as pd


def find_cps(maxes):
    """Find change points given a `maxes` array."""
    cps = []
    for i in range(1, len(maxes)):
        if abs(maxes[i] - maxes[i-1]) > 1:
            cps.append((i, abs(maxes[i] - maxes[i-1])))
    return cps

def drop_constant(df: pd.DataFrame):
    """Drop constant columns from the DataFrame."""
    return df.loc[:, (df != df.iloc[0]).any()]

def nsigma(data, k=3, startsfrom=100):
    """For each time series (column) in the data,
    detect anomalies using the n-sigma rule.

    Parameters:
    - data : pandas DataFrame
        The input data containing time series columns.
    - k : int, optional
        The number of standard deviations from the mean to consider as an anomaly. Default is 3.
    - startsfrom : int, optional
        The index from which to start calculating mean and standard deviation. Default is 100.

    Returns:
    - anomalies : list
        List of timestamps where anomalies were detected.
    """
    anomalies = []
    for col in data.columns:
        if col == "time":
            continue
        # for each timestep starts from `startsfrom`,
        # calculate the mean and standard deviation
        # of the all past timesteps
        for i in range(startsfrom, len(data)):
            mean = data[col].iloc[:i].mean()
            std = data[col].iloc[:i].std()
            if abs(data[col].iloc[i] - mean) > k * std:
                anomalies.append(data['time'].iloc[i])
    return anomalies


def find_anomalies(data, time_col=None, threshold=0.01):
    """Find anomalies in the data based on a given threshold.

    Parameters:
    - data : list or numpy array
        The input data to search for anomalies.
    - time_col : pandas Series, optional
        The timestamps corresponding to the data. Default is None.
    - threshold : float, optional
        The threshold value above which a data point is considered an anomaly. Default is 0.01.

    Returns:
    - merged_anomalies : list
        List of merged timestamps where anomalies were detected.
    - anomalies : list
        List of timestamps where anomalies were detected.
    """
    anomalies = []
    for i in range(1, len(data)):
        if data[i] > threshold:
            # anomalies.append(i)
            anomalies.append(time_col.iloc[i])

    # re-try if threshold doesn't work
    if len(anomalies) == 0:
        head = 5
        data = data[head:]
        # anomalies = [np.argmax(data) + head]
        anomalies = [time_col.iloc[np.argmax(data) + head]]

    # merge continuous anomalies if the distance are shorter than 5 steps
    merged_anomalies = [] if len(anomalies) == 0 else [anomalies[0]]
    for i in range(1, len(anomalies)):
        if anomalies[i] - anomalies[i - 1] > 5:
            merged_anomalies.append(anomalies[i])

    return merged_anomalies, anomalies


def constant_hazard(lam, r):
    """
    Hazard function for bayesian online learning
    Arguments:
        lam - inital prob
        r - R matrix
    """
    return 1 / lam * np.ones(r.shape)

hazard_function = partial(constant_hazard, 250)


def online_changepoint_detection(data, hazard_function, log_likelihood_class):
    """
    Parameters:
    data    -- the time series data

    Outputs:
        R  -- is the probability at time step t that the last sequence is already s time steps long
        maxes -- the argmax on column axis of matrix R (growth probability value) for each time step
    """
    maxes = np.zeros(len(data) + 1)

    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1

    for t, x in enumerate(data):
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        predprobs = log_likelihood_class.pdf(x)

        # Evaluate the hazard function for this interval
        H = hazard_function(np.array(range(t + 1)))

        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        R[1 : t + 2, t + 1] = R[0 : t + 1, t] * predprobs * (1 - H)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0, t + 1] = np.sum(R[0 : t + 1, t] * predprobs * H)

        # Renormalize the run length probabilities for improved numerical
        # stability.
        R[:, t + 1] = R[:, t + 1] / np.sum(R[:, t + 1])

        # Update the parameter sets for each possible run length.
        log_likelihood_class.update_theta(x, t=t)

        maxes[t] = R[:, t].argmax()

    return R, maxes









class BaseLikelihood(ABC):
    """
    This is an abstract class to serve as a template for future users to mimick
    if they want to add new models for online bayesian changepoint detection.

    Make sure to override the abstract methods to do which is desired.
    Otherwise you will get an error.

    Update theta has **kwargs to pass in the timestep iteration (t) if desired.
    To use the time step add this into your update theta function:
        timestep = kwargs['t']
    """

    @abstractmethod
    def pdf(self, data: np.array):
        raise NotImplementedError(
            "PDF is not defined. Please define in separate class to override this function."
        )

    @abstractmethod
    def update_theta(self, data: np.array, **kwargs):
        raise NotImplementedError(
            "Update theta is not defined. Please define in separate class to override this function."
        )



class MultivariateT(BaseLikelihood):
    def __init__(
        self,
        dims: int = 1,
        dof: int = 0,
        kappa: int = 1,
        mu: float = -1,
        scale: float = -1,
    ):
        """
        Create a new predictor using the multivariate student T distribution as the posterior predictive.
            This implies a multivariate Gaussian distribution on the data, a Wishart prior on the precision,
             and a Gaussian prior on the mean.
             Implementation based on Haines, T.S., Gaussian Conjugate Prior Cheat Sheet.
        :param dof: The degrees of freedom on the prior distribution of the precision (inverse covariance)
        :param kappa: The number of observations we've already seen
        :param mu: The mean of the prior distribution on the mean
        :param scale: The mean of the prior distribution on the precision
        :param dims: The number of variables
        """
        # We default to the minimum possible degrees of freedom, which is 1 greater than the dimensionality
        if dof == 0:
            dof = dims + 1
        # The default mean is all 0s
        if mu == -1:
            mu = [0] * dims
        else:
            mu = [mu] * dims

        # The default covariance is the identity matrix. The scale is the inverse of that, which is also the identity
        if scale == -1:
            scale = np.identity(dims)
        else:
            scale = np.identity(scale)

        # Track time
        self.t = 0

        # The dimensionality of the dataset (number of variables)
        self.dims = dims

        # Each parameter is a vector of size 1 x t, where t is time. Therefore each vector grows with each update.
        self.dof = np.array([dof])
        self.kappa = np.array([kappa])
        self.mu = np.array([mu])
        self.scale = np.array([scale])

    def pdf(self, data: np.array):
        """
        Returns the probability of the observed data under the current and historical parameters
        Parmeters:
            data - the datapoints to be evaualted (shape: 1 x D vector)
        """
        self.t += 1
        t_dof = self.dof - self.dims + 1
        expanded = np.expand_dims((self.kappa * t_dof) / (self.kappa + 1), (1, 2))
        ret = np.empty(self.t)
        try:
            # This can't be vectorised due to https://github.com/scipy/scipy/issues/13450
            for i, (df, loc, shape) in islice(
                enumerate(zip(t_dof, self.mu, inv(expanded * self.scale))), self.t
            ):
                ret[i] = ss.multivariate_t.pdf(x=data, df=df, loc=loc, shape=shape)
        except AttributeError:
            raise Exception(
                "You need scipy 1.6.0 or greater to use the multivariate t distribution"
            )
        return ret

    def update_theta(self, data: np.array, **kwargs):
        """
        Performs a bayesian update on the prior parameters, given data
        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
        """
        centered = data - self.mu

        # We simultaneously update each parameter in the vector, because following figure 1c of the BOCD paper, each
        # parameter for a given t, r is derived from the same parameter for t-1, r-1
        # Then, we add the prior back in as the first element
        self.scale = np.concatenate(
            [
                self.scale[:1],
                inv(
                    inv(self.scale)
                    + np.expand_dims(self.kappa / (self.kappa + 1), (1, 2))
                    * (np.expand_dims(centered, 2) @ np.expand_dims(centered, 1))
                ),
            ]
        )
        self.mu = np.concatenate(
            [
                self.mu[:1],
                (np.expand_dims(self.kappa, 1) * self.mu + data)
                / np.expand_dims(self.kappa + 1, 1),
            ]
        )
        self.dof = np.concatenate([self.dof[:1], self.dof + 1])
        self.kappa = np.concatenate([self.kappa[:1], self.kappa + 1])




def bocpd(data):
    """Perform Multivariate Bayesian Online Change Point Detection (BOCPD) on the input data.

    Parameters:
    - data : pandas DataFrame
        The input data containing metrics from microservices.

    Returns:
    - anomalies : list
        List of timestamps where anomalies were detected.
    """
    from functools import partial
    # from baro._bocpd import online_changepoint_detection, constant_hazard, MultivariateT
    data = data.copy()
    print('Before feature selection, shape of data:', data.shape)
    # select latency and error metrics from microservices
    selected_cols = []
    # names = ['Total cost per hour(USD)_memory', 'latency', 'Latency_memory', 'latency-50']
    # print(data.columns)
    for c in data.columns:
        #if 'queue-master' in c or 'rabbitmq_' in c: continue
        #if "101_cpu" in c or "301_cpu" in c or "401_cpu" in c or '402_cpu' in c or '502_cpu' in c or  '501_cpu' in c or '503_cpu' in c or '504_cpu' in c or '201_cpu' in c or '202_cpu' in c or '203_cpu' in c:
        if c == 'label':
            selected_cols.append(c)
    data = data[selected_cols]
    print('After feature selection, shape of data:', data.shape)
    # handle na
    data = drop_constant(data)
    data = data.fillna(method="ffill")
    data = data.fillna(0)
    for c in data.columns:
        data[c] = (data[c] - np.min(data[c])) / (np.max(data[c]) - np.min(data[c]))
    data = data.fillna(method="ffill")
    data = data.fillna(0)

    data = data.to_numpy()

    R, maxes = online_changepoint_detection(
        data,
        partial(constant_hazard, 50),
        MultivariateT(dims=data.shape[1])
    )
    cps = find_cps(maxes)
    anomalies = [p[0] for p in cps]
    # anomalies, merged_anomalies = find_anomalies(data=R[Nw,Nw:-1].tolist(), time_col=time_col)

    return anomalies


import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


def drop_constant(df: pd.DataFrame):
    """Drop constant columns from the DataFrame."""
    return df.loc[:, (df != df.iloc[0]).any()]


def drop_near_constant(df: pd.DataFrame, threshold: float = 0.1):
    """Drop columns with near-constant values from the DataFrame."""
    return df.loc[:, (df != df.iloc[0]).mean() > threshold]


def drop_time(df: pd.DataFrame):
    """Drop time-related columns from the DataFrame."""
    if "time" in df:
        df = df.drop(columns=["time"])
    if "Time" in df:
        df = df.drop(columns=["Time"])
    if "timestamp" in df:
        df = df.drop(columns=["timestamp"])
    return df


def select_useful_cols(data):
    """Select useful columns from the dataset based on certain criteria.

    Parameters:
    - data : pandas.DataFrame
        The dataset to select columns from.

    Returns:
    - selected_cols : list
        A list of selected column names.
    """
    selected_cols = []
    for c in data.columns:
        # keep time
        if "time" in c:
            selected_cols.append(c)

        # cpu
        if c.endswith("_cpu") and data[c].std() > 1:
            selected_cols.append(c)

        # mem
        if c.endswith("_mem") and data[c].std() > 1:
            selected_cols.append(c)

        # latency
        # if ("lat50" in c or "latency" in c) and (data[c] * 1000).std() > 10:
        if "lat50" in c and (data[c] * 1000).std() > 10:
            selected_cols.append(c)
    return selected_cols


def drop_extra(df: pd.DataFrame):
    """Drop extra columns from the DataFrame.

    Parameters:
    - df : pandas.DataFrame
        The DataFrame to remove extra columns from.

    Returns:
    - df : pandas.DataFrame
        The DataFrame after removing extra columns.
    """
    if "time.1" in df:
        df = df.drop(columns=["time.1"])

    # remove cols has "frontend-external" in name
    # remove cols start with "main_" or "PassthroughCluster_", etc.
    for col in df.columns:
        if (
                "frontend-external" in col
                or col.startswith("main_")
                or col.startswith("PassthroughCluster_")
                or col.startswith("redis_")
                or col.startswith("rabbitmq")
                or col.startswith("queue")
                or col.startswith("session")
                or col.startswith("istio-proxy")
        ):
            df = df.drop(columns=[col])

    return df


def convert_mem_mb(df: pd.DataFrame):
    """Convert memory values in the DataFrame to MBs.

    Parameters:
    - df : pandas.DataFrame
        The DataFrame containing memory values.

    Returns:
    - df : pandas.DataFrame
        The DataFrame with memory values converted to MBs.
    """

    # Convert memory to MBs
    def update_mem(x):
        if not x.name.endswith("_mem"):
            return x
        x /= 1e6
        # x = x.astype(int)
        return x

    return df.apply(update_mem)


def preprocess(data, dataset=None, dk_select_useful=False):
    """Preprocess the dataset.

    Parameters:
    - data : pandas.DataFrame
        The dataset to preprocess.
    - dataset : str, optional
        The dataset name. Default is None.
    - dk_select_useful : bool, optional
        Whether to select useful columns. Default is False.

    Returns:
    - data : pandas.DataFrame
        The preprocessed dataset.
    """
    data = drop_constant(drop_time(data))
    data = convert_mem_mb(data)

    if dk_select_useful is True:
        data = drop_extra(data)
        data = drop_near_constant(data)
        data = data[select_useful_cols(data)]
    return data


def nsigma(data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, **kwargs):
    """Perform nsigma analysis on the dataset.

    Parameters:
    - data : pandas.DataFrame
        The dataset to perform nsigma analysis on.
    - inject_time : int, optional
        The time of injection of anomalies. Default is None.
    - dataset : str, optional
        The dataset name. Default is None.
    - num_loop : int, optional
        Number of loops. Default is None.
    - sli : int, optional
        SLI (Service Level Indicator). Default is None.
    - anomalies : list, optional
        List of anomalies. Default is None.
    - kwargs : dict
        Additional keyword arguments.

    Returns:
    - dict
        A dictionary containing node names and ranks.
    """
    if anomalies is None:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(anomalies[0])
        # anomal_df is the rest
        anomal_df = data.tail(len(data) - anomalies[0])

    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    anomal_df = preprocess(
        data=anomal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    # intersect
    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    ranks = []

    for col in normal_df.columns:
        a = normal_df[col].to_numpy()
        b = anomal_df[col].to_numpy()

        scaler = StandardScaler().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = max(zscores)
        ranks.append((col, score))

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]

    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
    }


def robust_scorer(
        data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, **kwargs
):
    """Perform root cause analysis using RobustScorer.

    Parameters:
    - data : pandas.DataFrame
        The datas to perform RobustScorer.
    - inject_time : int, optional
        The time of fault injection time. Default is None.
    - dataset : str, optional
        The dataset name. Default is None.
    - num_loop : int, optional
        Number of loops. Default is None. Just for future API compatible
    - sli : int, optional
        SLI (Service Level Indicator). Default is None. Just for future API compatible
    - anomalies : list, optional
        List of anomalies. Default is None.
    - kwargs : dict
        Additional keyword arguments.

    Returns:
    - dict
        A dictionary containing node names and ranks. `ranks` is a ranked list of root causes.
    """
    if anomalies is None:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(min(anomalies[0], len(data)-3))
        # anomal_df is the rest
        anomal_df = data.tail(max(len(data) - anomalies[0], 3))
        # normal_df = data.head(anomalies[0])
        # # anomal_df is the rest
        # anomal_df = data.tail(len(data) - anomalies[0])

    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    anomal_df = preprocess(
        data=anomal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    # intersect
    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    ranks = []

    for col in normal_df.columns:
        a = normal_df[col].to_numpy()
        b = anomal_df[col].to_numpy()

        scaler = RobustScaler().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = max(zscores)
        ranks.append((col, score))

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    # print(ranks)
    # ranks = [x[0] for x in ranks]

    return ranks


