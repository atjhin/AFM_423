import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import ElasticNet

def get_equal_weights(test_data, weights=None):
    weights = np.ones(len(test_data), dtype=int) if weights is None else weights
    return pd.DataFrame({"stock_id":test_data['stock_id'], "weights":weights/weights.sum()})

def get_xgb_weights(train_data, test_data, features, xgb_params):
    # Data prep
    X_train = train_data[features]
    y_train = train_data["R12M_Usd"] / np.exp(train_data["Vol1Y_Usd"])
    X_test = test_data[features]
    
    # Train and fit model
    xgb_r = xgb.XGBRegressor(**xgb_params) 
    xgb_r.fit(X_train, y_train) 
    pred = xgb_r.predict(X_test) 
    weights = pred > np.median(pred)
    return get_equal_weights(test_data, weights)


def get_lr_weights(train_data, test_data, features, lr_params):
    # Data prep
    X_train = train_data[features]
    y_train = train_data["R12M_Usd"] / np.exp(train_data["Vol1Y_Usd"])
    X_test = test_data[features]

    # Train and fit model
    lasso_reg = ElasticNet(**lr_params)
    lasso_reg.fit(X_train, y_train)
    pred = lasso_reg.predict(X_test)
    weights = pred > np.median(pred)
    return get_equal_weights(test_data, weights)


## Backtesting function
def backtest(data_ml, features, t_oos, xgb_params={}, lr_params={}):
    m_offset = 12                                          
    train_size = 5 
    nb_port = 3
    ticks = data_ml['stock_id'].astype('category').cat.categories.tolist()
    ticks_df = pd.DataFrame({"stock_id":ticks}).reset_index()
    N = data_ml.stock_id.nunique()
    portf_weights = np.zeros((len(t_oos), nb_port, N))
    portf_returns = np.zeros((len(t_oos), nb_port))
    for t, curr_t in enumerate(t_oos):
        if t % 12 == 0:
            print(curr_t)
        train_data = data_ml[(data_ml['date'] < curr_t - pd.Timedelta(days=m_offset * 30)) &  \
                             (data_ml['date'] > curr_t - pd.Timedelta(days=m_offset * 30 + 365 * train_size))].reset_index(drop=True)
        test_data = data_ml[data_ml['date'] == curr_t].reset_index(drop=True)
        realized_returns = test_data['R1M_Usd']
        for i in range(nb_port):
            if i == 0:
                temp_weights = get_equal_weights(test_data)
            elif i == 1:
                temp_weights = get_lr_weights(train_data, test_data, features, lr_params)
            elif i == 2:
                temp_weights = get_xgb_weights(train_data, test_data, features, xgb_params)
            else:
                raise ValueError(f"Invalid number for nb_port: {nb_port}. Please enter a value from 1 to 3")
            ind = temp_weights.merge(ticks_df, how="left", on="stock_id").index.tolist()
            portf_weights[t,i,ind] = temp_weights["weights"]
            portf_returns[t,i] = np.multiply(temp_weights['weights'], realized_returns).sum()
    return portf_weights, portf_returns



def get_turnover(weights, data_ml, t_oos):
    ticks = data_ml['stock_id'].astype('category').cat.categories.tolist()
    ticks_df = pd.DataFrame({"stock_id":ticks}).reset_index()
    turnover = []
    for t in range(1, weights.shape[0]): 
        curr_return = data_ml.loc[data_ml['date']==t_oos[t], ['stock_id', 'R1M_Usd']]
        curr_return = ticks_df.merge(curr_return, how="left", on="stock_id")['R1M_Usd']
        prior_w = weights[t-1,:] * (1 + curr_return)  
        prior_w = prior_w / prior_w.sum()
        curr_turn = np.sum(np.abs(weights[t,:] - prior_w))  
        turnover.append(curr_turn)

    return np.mean(turnover)


def stationary_bootstrap(array, num, prob, seed):
    rng = np.random.default_rng(seed)
    N = len(array)
    result = []
    select_next = rng.choice([True, False], p=[1-prob, prob], size=num).tolist()
    next_ind = rng.integers(0, N, size=sum(select_next)).tolist()
    curr_ind = rng.integers(0, N)
    while len(select_next) > 0:
        result.append(array[curr_ind])
        if select_next.pop():
            curr_ind = (curr_ind + 1) % N
        else:
            curr_ind = next_ind.pop()
    return result
        
def get_bootstrap(returns, t, prob, N, seed=2025):
    initial_val = 1000
    portfolio_val = []
    rng = np.random.default_rng(seed)
    seed_ls = rng.integers(low=0, high=999999, size=N)
    for seed in seed_ls:
        bootstrap_returns = stationary_bootstrap(returns, t*12, prob, seed)
        portfolio_val.append(initial_val * np.prod(1+np.array(bootstrap_returns)))
    return portfolio_val


from scipy.stats import gaussian_kde
def plot_portf_cdf(portfolio_vals, strat_name, t, xlim):
    plt.figure(figsize=(8, 5))
    for name, vals in zip(strat_name, portfolio_vals):

        # Create and plot kernel density estimator for the CDF plot
        kde = gaussian_kde(vals)
        x_grid = np.linspace(np.min(vals), np.max(vals), 1000)
        density = kde(x_grid)
        dx = x_grid[1] - x_grid[0]
        cdf = np.cumsum(density) * dx
        cdf /= cdf[-1]
        line = plt.plot(x_grid, cdf, label=name)
        
        # Annotation
        color = line[0].get_color()
        mid_idx = len(x_grid) // 4
        x_mid = x_grid[mid_idx]
        y_mid = cdf[mid_idx]
        xytext = (x_mid * 1.05, y_mid) if name == "LR" else (x_mid * 0.9, y_mid)
        plt.annotate(
            name,
            xy=(x_mid, y_mid),
            xytext=xytext, 
            color=color,
            fontsize=10,
            ha='left',
            va='center'
        )
    plt.xlim(right=xlim)
    plt.ylim(bottom=0, top=1)
    plt.xlabel('Terminal Portfolio Value')
    plt.ylabel('CDF')
    plt.title(f'Empirical CDF of Terminal Portfolio Values ({t}-Year Horizon)')
    plt.show()



def perf_met_multi(portf_returns, portf_weights, asset_returns, t_oos, strat_name, t, p, xlim=8000, seed=2025):
    metrics_df = pd.DataFrame()
    num_bootstrap=1000
    bootstrap_portf_vals = []
    for j in range(portf_weights.shape[1]):
        turnover = get_turnover(portf_weights[:,j,:], asset_returns, t_oos)
        curr_bootstrap = get_bootstrap(portf_returns[:,j], t, p, num_bootstrap, seed=seed)
        curr_mean = np.mean(curr_bootstrap)
        curr_std = np.std(curr_bootstrap)
        bootstrap_portf_vals.append(curr_bootstrap)
        metrics_df = pd.concat([metrics_df, pd.DataFrame({"strat":[strat_name[j]], 
                                                          "turnover":[turnover],
                                                          "mean":[np.mean(portf_returns[:,j])],
                                                          "std":[np.std(portf_returns[:,j])], 
                                                          "sharpe":[np.mean(portf_returns[:,j])/np.std(portf_returns[:,j])],
                                                          "bt_mean":[curr_mean],
                                                          "bt_std":[curr_std]})], axis=0)
    plot_portf_cdf(bootstrap_portf_vals, strat_name, t, xlim=xlim)
    return metrics_df.reset_index(drop=True).round(4)


