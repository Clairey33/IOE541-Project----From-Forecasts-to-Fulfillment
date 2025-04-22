import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Constants from the paper
HOLDING_COST_RATE = 0.01  # 1% of unit price
LOST_SALE_COST_RATE = 0.125  # 12.5% of unit price
ORDER_COST_FIXED = 0.5  # fixed $0.5 per order
UNIT_PRICE = 1.0  # assume normalized price for analysis; update as needed

# get the file names of all files within evaluation_output folder
# def get_file_names():
#     current_file = os.path.abspath(__file__)
#     code_dir = os.path.dirname(current_file)
#     data_dir = os.path.join(os.path.dirname(code_dir), "evaluation_output")
#     # List all files in the directory
#     file_names = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    
#     print("------------------- File names retrieved successfully -------------------",end="\n\n")
#     return file_names


# def get_filepath(file):
#     # Get the directory of the current Python script
#     current_file = os.path.abspath(__file__)
#     code_dir = os.path.dirname(current_file)
#     data_dir = os.path.join(os.path.dirname(code_dir), "evaluation_output")
#     filename = file
#     filepath = os.path.join(data_dir, filename)
    
#     return filepath


# Get the file names of all files within evaluation_output folder
def get_file_names():
    current_file = os.path.abspath(__file__)
    code_dir = os.path.dirname(current_file)
    data_dir = os.path.join(os.path.dirname(code_dir), "evaluation_output")
    # List all files in the directory
    file_names = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    print("------------------- File names retrieved successfully -------------------\n")
    return file_names, data_dir

def get_filepath(data_dir, file):
    filepath = os.path.join(data_dir, file)
    print(f"Loaded: {file}")
    return filepath

def extract_efficiency_data(filepath):
    df = pd.read_csv(filepath, skiprows=3, sep=r"\s{2,}", engine='python', header=None, names=["metric", "value"])
    model = open(filepath).readline().strip()
    sku = open(filepath).readlines()[1].strip()

    mean_inventory = float(df[df["metric"] == "mean_inventory"]["value"].values[0])
    lost_sales = float(df[df["metric"] == "lost_sales"]["value"].values[0])
    num_orders = float(df[df["metric"] == "count_orders"]["value"].values[0])

    holding_cost = mean_inventory * UNIT_PRICE * HOLDING_COST_RATE
    lost_sales_cost = lost_sales * UNIT_PRICE * LOST_SALE_COST_RATE
    ordering_cost = num_orders * ORDER_COST_FIXED
    total_cost = holding_cost + lost_sales_cost + ordering_cost

    tsl_match = re.search(r"tsl_(\d+\.\d+)", os.path.basename(filepath)) 
    tsl = float(tsl_match.group(1)) if tsl_match else None
    
    return {
        "model": model,
        "sku": sku,
        "mean_inventory": mean_inventory,
        "lost_sales": lost_sales,
        "num_orders": num_orders,
        "holding_cost": holding_cost,
        "lost_sales_cost": lost_sales_cost,
        "ordering_cost": ordering_cost,
        "total_cost": total_cost,
        "tsl": tsl
    }

if __name__ == "__main__":
    print("------------------- Process started -------------------\n")
    file_names, data_dir = get_file_names()
    all_metrics = []

    for file in file_names:
        filepath = get_filepath(data_dir, file)
        metrics = extract_efficiency_data(filepath)
        # print(metrics)
        all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)

    # # Plot Efficiency Curve: Lost Sales vs Inventory
    # for (model, sku), group in df.groupby(['model', 'sku']):
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(group['mean_inventory'], group['lost_sales'], marker='o', label=f"{model}")
    #     plt.title(f"Efficiency Curve - {sku}")
    #     plt.xlabel("Mean Inventory")
    #     plt.ylabel("Lost Sales")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.legend()
    #     plt.show()

    # Plot Combined Efficiency Curve: Lost Sales vs Inventory
    for sku in df["sku"].unique():
        plt.figure(figsize=(8, 5))
        subset = df[df["sku"] == sku]
        sns.lineplot(data=subset, x="mean_inventory", y="lost_sales", hue="model", marker="o")
        plt.title(f"Efficiency Curve - {sku}")
        plt.xlabel("Mean Inventory")
        plt.ylabel("Lost Sales")
        plt.grid(True)
        plt.tight_layout()
        plt.legend(title="Model")
        plt.show()

    # # Plot Financial Cost Curve: Total Cost vs TSL
    # for (model, sku), group in df.groupby(['model', 'sku']):
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(group['tsl'], group['total_cost'], marker='o', label=f"{model}")
    #     plt.title(f"Total Cost vs TSL - {sku}")
    #     plt.xlabel("Target Service Level (TSL)")
    #     plt.ylabel("Total Inventory Cost")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.legend()
    #     plt.show()