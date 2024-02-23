import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List
from itertools import product, combinations
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

class TspVisualizer:
    """visualize a TSP tour
    Codes are taken from https://kunlei.github.io/python-ortools-notes/ip-tsp.html
    As the author of the codes, Kunlei Lian grants permission of using the codes for solving the TSP this client (Alan Aastorp) needs.
    """
    
    @staticmethod
    def show(locations, edges):
        """draw TSP tour
        adapted from https://stackoverflow.com/a/50682819
        
        examples:
        locations = {
            0: (5, 5),
            1: (4, 9),
            2: (6, 4),
        }

        edges = [
            (0, 1),
            (1, 2),
            (2, 0),
        ]

        Args:
            locations (dict): location id -> (lat, lon)
            edges (list): list of edges
        """
        G = nx.DiGraph()
        G.add_edges_from(edges)
        plt.figure(figsize=(15,10))
        
        colors = mpl.colormaps["Set1"].colors
        color_idx = 1
        color = np.array([colors[color_idx]])
        
        nx.draw_networkx_nodes(G, 
                                locations, 
                                nodelist=[x[0] 
                                        for x in edges], 
                                node_color=color)
        nx.draw_networkx_edges(G,
                                locations, 
                                edgelist=edges,
                                width=4, 
                                edge_color=color, 
                                style='dashed')
        
        # labels
        nx.draw_networkx_labels(G, locations, 
                                font_color='w', 
                                font_size=12, 
                                font_family='sans-serif')

        #print out the graph
        plt.axis('off')
        plt.show()

class TspModelBase:
    """base class for TSP models
    Codes are taken from https://kunlei.github.io/python-ortools-notes/ip-tsp.html
    As the author of the codes, Kunlei Lian grants permission of using the codes for solving the TSP this client (Alan Aastorp) needs.
    """
    
    def __init__(self, name: str):
        self._name: str = name
        self._node_list: List[int] = None
        self._node_coords: Dict[int, List[float, float]] = None
        
    def read_inputs(self, customers: pd.DataFrame) -> None:
        self._name = 'tsp'
        self._node_list = customers.index.to_list()
        self._node_coords = {
            id: customers.loc[id, :].to_list()
            for id in self._node_list
        }
        self._node_idx_to_id_dict = {}
        idx = 0
        for id in self._node_list:
            self._node_idx_to_id_dict[idx] = id
            idx += 1
            
        self._distance_matrix = []
        for i in self._node_list:
            dist = [int(np.sqrt(np.sum(np.square(np.array(self._node_coords[i]) - np.array(self._node_coords[j])))) * 100)
                for j in self._node_list]
            self._distance_matrix.append(dist)

    def get_combinations(self, size):
        return list(combinations(self._node_list, size))
        
    @property
    def name(self): return self._name
    
    @property
    def num_nodes(self): return len(self._node_coords)

class TspOrtools(TspModelBase):
    def __init__(self, name='TSP heuristic solver based on ortools'):
        super().__init__(name)
        
    def solve(self):
        # create routing index manager
        manager = pywrapcp.RoutingIndexManager(len(self._node_list), 1, 0)
        
        # create routing model
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return self._distance_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        
        # define arc cost
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # set solution heuristic
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        
        # solve the problem
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            print(f'solution found!')
            index = routing.Start(0)
            route_distance = 0
            route = []
            while not routing.IsEnd(index):
                route.append(self._node_idx_to_id_dict[manager.IndexToNode(index)])
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
            
            route_edges = []
            for idx in range(len(route) - 1):
                route_edges.append((route[idx], route[idx + 1]))
            route_edges.append((route[-1], route[0]))
            
            print(f"total distance: {route_distance}")
            print(f"route: {route}")
            print(f'route_edges: {route_edges}')
            TspVisualizer.show(self._node_coords, route_edges)
        else:
            print(f'No feasible solution found!')
    

if __name__ == '__main__':
    filename = sys.argv[1]
    print(f'filename: {filename}')

    df_customers = pd.read_excel(filename, sheet_name='customers')
    df_customers = df_customers.set_index('id')
    if df_customers.empty:
        print(f'empty input!')
    elif df_customers.index.nunique() < df_customers.shape[0]:
        print(f'duplicate location id exists, abort program...')
    else:        
        tsp_solver = TspOrtools()
        tsp_solver.read_inputs(df_customers)
        tsp_solver.solve()
