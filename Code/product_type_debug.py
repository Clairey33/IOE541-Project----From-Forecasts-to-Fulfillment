import pandas as pd
import numpy as np
from datetime import timedelta
import pdb
import os
import json
import time

def load_data():
    # Load preprocessed data
    current_file = os.path.abspath("__file__")
    code_dir = os.path.dirname(current_file)
    data_dir = os.path.join(os.path.dirname(code_dir), "Data", "Processed")
    print("--------------- Data loaded successfully --------------- ", end="\n\n")
    
    sales_model = pd.read_pickle(os.path.join(data_dir, "lgbm_state_evaluation_data.pkl"))
    print("--------------- Model loaded successfully ---------------", )
    print(sales_model.head(), end="\n\n")

    return sales_model    


def classify_demo(sales_model):
    # get all unique values of the column 'item_id'
    # item_ids = sales_model['item_id'].unique()
    # state_ids = sales_model['state_id'].unique()
    start_time = time.time()

    item_ids = [
        "FOODS_3_819", "FOODS_3_090",
        "HOBBIES_1_234", "HOUSEHOLD_1_118"
    ]
    state_ids = ['CA', 'TX', 'WI'] 

    results = []

    grouped = sales_model.groupby(['item_id', 'state_id'])

    for (item, state), group in grouped:
        sales = group['sales'].values

        # Inter-demand intervals
        non_zero_indices = np.where(sales > 0)[0]
        if len(non_zero_indices) < 2:
            adi = np.inf
        else:
            intervals = np.diff(non_zero_indices)
            adi = np.mean(intervals)

        # CV^2 calculation
        demand_sizes = sales[sales > 0]
        if len(demand_sizes) < 2 or np.mean(demand_sizes) == 0:
            cv2 = np.inf
        else:
            cv2 = (np.std(demand_sizes) / np.mean(demand_sizes)) ** 2

        # Classification
        if cv2 < 0.5 and adi <= 4/3:
            demand_type = 'smooth'
        elif cv2 < 0.5 and adi > 4/3:
            demand_type = 'intermittent'
        elif cv2 >= 0.5 and adi > 4/3:
            demand_type = 'lumpy'
        else:
            demand_type = 'erratic'

        results.append({
            'id': f'{item}_{state}',
            'ADI': adi,
            'CV2': cv2,
            'demand_type': demand_type
        })

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("--------------- Classification finished successfully --------------- ")
    print(f"--------------- Finished in {elapsed_time:.2f} seconds ---------------")

    return pd.DataFrame(results)

   
def classify_old(sales_model):
    # get all unique values of the column 'item_id'
    # item_ids = sales_model['item_id'].unique()
    # state_ids = sales_model['state_id'].unique()
    start_time = time.time()

    item_ids = [
        "FOODS_3_819", "FOODS_3_090",
        "HOBBIES_1_234", "HOUSEHOLD_1_118"
    ]
    state_ids = ['CA', 'TX', 'WI'] 

    results = []
    for item in item_ids:
        for state in state_ids:
            item_sales = sales_model[(sales_model['item_id'] == item) & (sales_model['state_id'] == state)]
            sales = item_sales['sales'].values #

            # Inter-demand intervals: gaps between non-zero sales
            non_zero_indices = np.where(sales > 0)[0]
            if len(non_zero_indices) < 2:
                adi = np.inf  # No pattern
            else:
                intervals = np.diff(non_zero_indices)
                adi = np.mean(intervals)

            # CV^2 = (std / mean)^2 on non-zero sales
            demand_sizes = sales[sales > 0] 
            if len(demand_sizes) < 2 or np.mean(demand_sizes) == 0:
                cv2 = np.inf
            else:
                cv2 = (np.std(demand_sizes) / np.mean(demand_sizes)) ** 2

            # Classification
            if cv2 < 0.5 and adi <= 4/3:
                demand_type = 'smooth'
            elif cv2 < 0.5 and adi > 4/3:
                demand_type = 'intermittent'
            elif cv2 >= 0.5 and adi > 4/3:
                demand_type = 'lumpy'
            else:
                demand_type = 'erratic'
            
            results.append({
                'id': item + '_' + state,
                'ADI': adi,
                'CV2': cv2,
                'demand_type': demand_type
            })
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("--------------- Classification finished successfully --------------- ")
    print(f"--------------- Finished in {elapsed_time:.2f} seconds ---------------")

    return pd.DataFrame(results)
        

def classify_full(sales_model, verbose=False):
    item_ids = sales_model['item_id'].unique()
    state_ids = sales_model['state_id'].unique()

    # get all unique values of the column 'item_id'
    # item_ids = sales_model['item_id'].unique()
    # state_ids = sales_model['state_id'].unique()
    start_time = time.time()

    results = []

    grouped = sales_model.groupby(['item_id', 'state_id'])

    for (item, state), group in grouped:
        sales = group['sales'].values

        # Inter-demand intervals
        non_zero_indices = np.where(sales > 0)[0]
        if len(non_zero_indices) < 2:
            adi = np.inf
        else:
            intervals = np.diff(non_zero_indices)
            adi = np.mean(intervals)

        # CV^2 calculation
        demand_sizes = sales[sales > 0]
        if len(demand_sizes) < 2 or np.mean(demand_sizes) == 0:
            cv2 = np.inf
        else:
            cv2 = (np.std(demand_sizes) / np.mean(demand_sizes)) ** 2

        # Classification
        if cv2 < 0.5 and adi <= 4/3:
            demand_type = 'smooth'
        elif cv2 < 0.5 and adi > 4/3:
            demand_type = 'intermittent'
        elif cv2 >= 0.5 and adi > 4/3:
            demand_type = 'lumpy'
        else:
            demand_type = 'erratic'

        if verbose:
            print(f"---- Item: {item}, Demand Type: {demand_type} ----")
    
        results.append({
            'id': f'{item}_{state}',
            'ADI': adi,
            'CV2': cv2,
            'demand_type': demand_type
        })

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("--------------- Classification finished successfully --------------- ")
    print(f"--------------- Finished in {elapsed_time:.2f} seconds ---------------", end="\n\n")

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("--------------- Debugging started --------------- ", end="\n\n")
    # Step1: Test Load Data
    sales_model = load_data()

    # Step2: Test Classification
    # classify_old(sales_model)
    # classify_demo(sales_model)
    
    result = classify_full(sales_model)
    # result = classify_full(sales_model, verbose=True)
    

    result.to_csv("demand_classification.csv", index=False)
    result_list = result.to_dict(orient="records")
    with open("demand_classification.json", "w") as f:
        json.dump(result_list, f, indent=4)