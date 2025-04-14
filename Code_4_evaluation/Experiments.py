import numpy as np
import pandas as pd
import scipy as sp
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

def safety_stock(SL, sd, L):
    """
    Calculate safety stock based on the service level, 
    standard deviation of demand, and risk time.
    
    :param SL: Target service level (between 0 and 1)
    :param sd: Standard deviation of demand
    :param L: risk time
    :return: Rounded safety stock
    """
    if SL <= 0 or SL >= 1:
        raise ValueError("SL must be strictly between 0 and 1.")
    
    # z is the z-score for the service level
    z = sp.stats.norm.ppf(SL)
    
    # Compute safety stock
    st = z * sd * np.sqrt(L)
    
    # Round to two decimals
    return st



def get_demand_data(product_id):
    '''
    Very draft version
    '''
    file_path = os.path.join(current_dir, "Data_raw/sales_train_evaluation.csv")
    df = pd.read_csv(file_path, header=0)
    df = df.loc[df['id']==product_id].T
    df = df.drop(df.index[range(6)])
    df = df.reset_index()
    df.columns = ["step", "demand"]
    return(df)



def ARIMA_forecast(forecast_f):
    '''
    return a map, key is review day, value is a list for prediction t+1, t+2, ..., t+R+L
    '''
    file_path = os.path.join(current_dir, f"Data_raw/{forecast_f}")
    forecast_df = pd.read_csv(file_path, header=0)
    forecasts = forecast_df.groupby("review_day")["forecast"].apply(list).to_dict()
    return forecasts


def RS(FT=7, 
       L=3, 
       SL=0.95, 
       initial_stock=0, 
       actual_demand=None, 
       forecasts=None,
       S=None, 
       method="ARIMA",
       debug = False):
    """
    Simulation function for order-up-to policy.
    Only works on a given product (a single time series)
    Parameters:
      FT: review period (R in the paper)
      L: lead time
      SL: target service level
      initial_stock: initial inventory level (a number)
      actual_demand: a dataframe, two columns, [step, demand]
                     step looks like d_{a number}
      forecasts: a dictionary {d_{timme step}: list[predictions]}
      S: optional target inventory level; if None or 0 the forecast is used to set Q
      method: string defining which forecasting method to use.

    Returns:
      A list of performance metrics.
    """
    # Check for missing variables
    if any(v is None for v in [initial_stock, actual_demand, forecasts, FT, L, SL]):
        raise ValueError("Variable missing")
    
    # all time step in the code is consistent with d_{...} in demand dataframe,
    # the relationship between time step and index in demand dataframe is
    # time step - 1 = index

    review_periods = forecasts.keys()

    fh = FT + L   # forecast horizon = review period + lead time
    last_day = actual_demand.iloc[-1, 0] # should be d_{a number}
    n_days = actual_demand.shape[0] # row numbers
    assert int(last_day[2:])==n_days
    # initial/training demand period 
    # known_periods means from d_1 to d_known_periods is the training data
    known_periods = n_days - 393 

    # Known demand is the training data
    known_demand = actual_demand.iloc[:known_periods, 1]

    if debug:
        with open("ana.txt", "a") as f:
            print(known_periods, file=f)
            print("latest demand should be\n", actual_demand.iloc[known_periods-1, :], file=f)
            print(len(known_demand), file=f)

    # Initialize arrays (lists) with length = known_periods
    orders = [0] * known_periods
    sent = [0] * known_periods
    backorders = [0] * known_periods
    frst_diffs = [0] * known_periods
    forecasts_array = [0] * (known_periods+1)

    # inventory level over time 
    # s[t-1] is inventory at the end of period t
    s = [initial_stock] * known_periods  
    start = True
    count_ft = 0
    actual_demand_formetrics = []
    pinball_frc = []
    pinball_act = []
    #adjust to pure number
    review_days = [int(d[2:]) for d in review_periods] 

    for t in range(known_periods+1, n_days+1):
        # Append sent: actual dispatch is the minimum of demand and available stock at previous period.
        if debug:
            with open("ana.txt", "a") as f:
                print("period", f'd_{t}', file=f)

        sent_val = min(actual_demand.iloc[t-1, 1], s[-1])
        sent.append(sent_val)

        if t in review_days:
            count_ft += 1
            # Update known demand to current period
            known_demand = actual_demand.iloc[:t, 1]
            # Compute safety stock using the standard deviation of known demand
            try:
                sigma = np.std(known_demand) if len(known_demand) > 1 else 0
            except Exception as e:
                sigma = 0
            ss = safety_stock(SL, sigma, L + FT)

            if S is None or S == 0:
                # Select forecast method – all computations are wrapped in a try block.
                try:
                    if method == "ARIMA":
                        forecast = forecasts[f'd_{t}']
                    elif method == "LGBM":
                        # In the R code this reads a file. Here we raise an error.
                        raise NotImplementedError("LGBM forecast method not implemented.")
                    else:
                        pass
                except Exception as e:
                    forecast = [0] * fh
                # Ensure nonnegative forecasts
                forecast = [max(0, f) for f in forecast]
                # Q is the target inventory level = forecast sum + safety stock
                Q = sum(forecast) + ss
                # Record values for pinball loss computation
                pinball_frc.append(Q)
                # Sum actual demand over next fh periods (adjust slicing accordingly)
                pinball_act.append(sum(actual_demand.iloc[t : t+fh, 1]))
            else:
                Q = S
            # Append the forecast values – only the first fh values
            forecasts_array.extend(forecast[:fh])
            # Record the actual demand for metric calculations
            actual_demand_formetrics.extend(actual_demand.iloc[t : t+fh, 1])

            diff = np.diff(known_demand)
            mean_diff_sq = np.sum(np.square(diff))
            frst_diffs.extend([mean_diff_sq] * FT)

            # Calculate order amount.
            start_idx = max(t - L - 1, 0)  # converting to 0-based index
            sum_orders = sum(orders[start_idx : ])
            order_amt = max(Q - s[-1] + sent[-1] - sum_orders, 0)
            orders.append(order_amt)
        else:
            # No review: no order placed.
            orders.append(0)
            if count_ft == 0:
                forecasts_array.append(0)

        # Update inventory level.
        # If within lead time, no received order yet
        if t <= L:
            new_inventory = s[-1] - sent[-1]
        else:
            # orders[t-L] is the order placed L periods ago, now arriving.
            new_inventory = s[-1] - sent[-1] + orders[t - L - 1]
        s.append(new_inventory)

        if debug:
            with open("ana.txt", "a") as f:
                print("inventory level", file=f)
                print(s[-20:], file=f)

                print("order amount", file=f)
                print(orders[-20:], file=f)

                print("sent value", file=f)
                print(sent[-20:], file=f)

                print("period", f'd_{t}', file=f)
                print("current t:", t, file=f)

                print("aligned", file=f)
                assert(t==len(s)==len(orders)==len(sent))
                print(len(known_demand), file=f)

    # Post-simulation metric calculations

    forecasts_array = forecasts_array[known_periods+1 :]
    forecasts_array = forecasts_array[: len(actual_demand_formetrics)]

    forecasts_array = np.array(forecasts_array)
    actual_demand_formetrics = np.array(actual_demand_formetrics)

    # For inventory and orders, take the tail (dropping the first cutoff elements) last 365 days 
    inv = np.array(s[-365:])
    ords = np.array(orders[-365:])
    actual_demand_list = actual_demand.iloc[-365:, 1].to_numpy()
    # Pinball loss calculation
    temp_sum = []
    for act_val, frc_val in zip(pinball_act, pinball_frc):
        if act_val >= frc_val:
            temp_sum.append((act_val - frc_val) * SL)
        else:
            temp_sum.append((frc_val - act_val) * (1 - SL))
    # scaler is the mean of frst_diffs (dropping first cutoff elements)
    if len(frst_diffs[-365:]) > 0:
        scaler = np.mean(frst_diffs[-365:])
    else:
        scaler = 1  # avoid division by zero
    PLd = np.mean([d/scaler for d in temp_sum]) if scaler != 0 else 0
    PL = (np.mean(temp_sum) / scaler) if scaler != 0 else 0

    output = [
        FT,
        L,
        SL,
        np.sum(sent[-365:])/np.sum(actual_demand_list),
        np.mean(np.array(sent[-365:])[actual_demand_list>0]/actual_demand_list[actual_demand_list>0]),
        np.mean(s[-365:]),
        np.max(s[-365:]),
        np.sum(np.sum(actual_demand_list)-np.sum(sent[-365:])),
        len(inv[inv==0]),
        np.mean(inv)/np.mean(actual_demand_list),
        np.sum(ords),
        len(ords[ords>0]),
        np.sum(actual_demand_formetrics),
        np.sum(forecasts_array),
        np.mean(actual_demand_formetrics-forecasts_array),
        np.mean(np.abs(actual_demand_formetrics-forecasts_array)),
        np.sqrt(np.mean((actual_demand_formetrics-forecasts_array)**2)),
        np.sqrt(np.mean((actual_demand_formetrics-forecasts_array)**2)/np.mean((frst_diffs[-365:]))),
        np.sqrt(np.mean((actual_demand_formetrics-forecasts_array)**2)/(np.mean(actual_demand_formetrics)**2)),
        np.mean(np.abs(actual_demand_formetrics-forecasts_array))/np.mean(np.abs(frst_diffs[-365:])),
        PLd,
        PL
    ]

    output_col = ["FT","L","target_service_level","fill_rate","service_level","mean_inventory","max_inventory","lost_sales","out_of_stock_days",
                  "inventory_days","sum_orders","count_orders","sum_demand","sum_forecasts","ME","MAE","RMSE","RMSSE","RMSsE","FD2","PLd","PL"]
    
    return (zip(output_col,output), (sent, s, orders))


if __name__=="__main__":
    product_id = "FOODS_3_819_WI_3_evaluation"
    forecast_f = "rolling_forecasts_arima.csv"

    actual_demand = get_demand_data(product_id)
    forecasts = ARIMA_forecast(forecast_f)

    review_time = 7
    lead_time = 3
    service_level = 0.90
    method = "ARIMA"

    (output, (sent, inventory, orders)) = RS(FT=review_time, 
                                             L=lead_time, 
                                             SL=service_level, 
                                             initial_stock=0, 
                                             actual_demand=actual_demand, 
                                             forecasts=forecasts,
                                             S=None, 
                                             method="ARIMA",
                                             debug = False)
    
    output_df = pd.DataFrame(output)

    print(output_df)