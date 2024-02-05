# TSP Solver

## Environment Setup

To run the code, create a `conda` environment `tsp-solver` with the following command (It is assumed that `conda` is already installed on your system):

```bash
conda create -n tsp-solver python=3.12
```

Then activate the environment with the following command:

```bash
conda activate tsp-solver
```

Next, install the required packages with the below command:

```bash
pip install numpy pandas matplotlib scipy ortools ipykernel jupyterlab openpyxl networkx
```

To solve a particular TSP instance, all you need to do is to create an Excel file containing all the customers and their corresponding locations.
Note that the format of the file must follow the examples given in the 'inputs' folder.
Specifically, the first row of the file denotes the column names of the data - 'id' is the customer's identifier, 'latitude' and 'longitude' are the customer's location.
In addition, the name of the sheet must be 'customers'.
Feel free to change the sheet name if you're familiar with the code.

## Run the code in Jupyter Notebook

To run the code in Jupyter notebook, simply open the `tsp-solver.ipynb` and select the virtual environment you created for this problem.
In the last cell, replace the value of `filename` with the problem instance you want to solve.
For example, if your data is in file `customer-data.xlsx`, set `filename = ../inputs/customer-data.xlsx`.

Once the filename is set, simply run all the cells in the notebook and the optimal graph will be shown in the end.

## Run the code in Python Script

To solve a problem via command line, please run the command below:

```bash
python ./src/tsp-solver.py './inputs/tsp-customers-test1.xlsx'
```

Once the solver finishes running, the optimal route will be shown and a figure will pop up and could be saved to a file.
