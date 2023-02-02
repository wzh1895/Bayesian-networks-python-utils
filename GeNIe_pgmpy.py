'''
Utility that reads Bayesian Networks created in GeNIe's .xdsl format and 
implements them with pgmpy, an open-source Python package for Bayesian Networks.
'''

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from bs4 import BeautifulSoup
from pathlib import Path
import networkx as nx
import numpy as np


def print_full(cpd):
    backup = TabularCPD._truncate_strtable
    TabularCPD._truncate_strtable = lambda self, x: x
    print(cpd)
    TabularCPD._truncate_strtable = backup


def parse_xdsl(xdsl_path, verbose=False):
    xdsl_path = Path(xdsl_path)
    if xdsl_path.exists():
        with open(xdsl_path) as f:
            xdsl_content = f.read().encode()
            f.close
    else:
        open(xdsl_path)
        return

    dependencies = dict()
    node_states = dict()
    node_pointvalues = dict()
    node_intervals = dict()
    cpds = dict()

    nodes_root = BeautifulSoup(xdsl_content, 'xml').find('nodes')
    cpts = nodes_root.findAll('cpt')

    for cpt in cpts:
        name = cpt.get('id')
        if verbose:
            print("Creating CPT for:",name)

        # create states
        states = cpt.findAll('state')
        self_state_names = list()
        for state in states:
            self_state_names.append(state.get('id'))
        
        node_states[name] = self_state_names
        
        state_names = dict()
        state_names[name] = self_state_names

        # create evidence & evidence_card
        parents = cpt.find('parents')

        if parents is not None:
            parents = parents.text.split(' ')
            dependencies[name] = parents
        
        if parents is None:
            evidence_card = None
        else:
            evidence_card = list()
            for parent in parents:
                evidence_card.append(len(node_states[parent]))
                state_names[parent] = node_states[parent]
        
        # create pointvalues or intervals (if any)
        pointvalues = cpt.find('pointvalues')
        if pointvalues:
            pointvalues = [float(pv) for pv in pointvalues.text.split(' ')]
            self_pointvalues = dict()
            for i in range(len(self_state_names)):
                self_pointvalues[self_state_names[i]] = pointvalues[i]
            node_pointvalues[name] = self_pointvalues
        
        intervals = cpt.find('intervals')
        if intervals:
            intervals = [float(iv) for iv in intervals.text.split(' ')]
            self_intervals = dict()
            for i in range(len(self_state_names)):
                self_intervals[self_state_names[i]] = (intervals[i], intervals[i+1])
            node_intervals[name] = self_intervals

        # create values
        probabilities = cpt.find('probabilities').text.split(' ')
        probabilities = [float(p) for p in probabilities]
        values = list()
        if evidence_card is not None:
            n_columns = np.product(evidence_card)
        else:
            n_columns = 1
        for i in range(n_columns):
            values.append(list(probabilities[i * len(self_state_names):(i+1) * len(self_state_names)]))
        values = np.transpose(values)

        cpd=TabularCPD(
            variable=name,
            variable_card=len(self_state_names),
            evidence=parents,
            evidence_card=evidence_card,
            state_names=state_names,
            values=values,
        )

        # print_full(cpd)

        cpds[name] = cpd

    edges = list()
    for child, parents in dependencies.items():
        for parent in parents:
            edges.append((parent, child))
    # print(edges)
    model = BayesianNetwork(edges)
    model.add_cpds(*list(cpds.values()))
    model.check_model()

    return model, node_pointvalues, node_intervals
