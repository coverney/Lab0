'''Code file for baseball elimination lab created for Advanced Algorithms
Spring 2020 at Olin College. The code for this lab has been adapted from:
https://github.com/ananya77041/baseball-elimination/blob/master/src/BaseballElimination.java'''

import sys
import math
import picos as pic
import networkx as nx
import itertools
import cvxopt
import matplotlib.pyplot as plt


class Division:
    '''
    The Division class represents a baseball division. This includes all the
    teams that are a part of that division, their winning and losing history,
    and their remaining games for the season.

    filename: name of a file with an input matrix that has info on teams &
    their games
    '''

    def __init__(self, filename):
        self.teams = {}
        self.G = nx.DiGraph()
        self.readDivision(filename)

    def readDivision(self, filename):
        '''Reads the information from the given file and builds up a dictionary
        of the teams that are a part of this division.

        filename: name of text file representing tournament outcomes so far
        & remaining games for each team
        '''
        f = open(filename, "r")
        lines = [line.split() for line in f.readlines()]
        f.close()

        lines = lines[1:]
        for ID, teaminfo in enumerate(lines):
            team = Team(int(ID), teaminfo[0], int(teaminfo[1]), int(teaminfo[2]), int(teaminfo[3]), list(map(int, teaminfo[4:])))
            self.teams[ID] = team

    def get_team_IDs(self):
        '''Gets the list of IDs that are associated with each of the teams
        in this division.

        return: list of IDs that are associated with each of the teams in the
        division
        '''
        return self.teams.keys()

    def is_eliminated(self, teamID, solver):
        '''Uses the given solver (either Linear Programming or Network Flows)
        to determine if the team with the given ID is mathematically
        eliminated from winning the division (aka winning more games than any
        other team) this season.

        teamID: ID of team that we want to check if it is eliminated
        solver: string representing whether to use the network flows or linear
        programming solver
        return: True if eliminated, False otherwise
        '''
        flag1 = False
        team = self.teams[teamID]

        temp = dict(self.teams)
        del temp[teamID]

        for _, other_team in temp.items():
            if team.wins + team.remaining < other_team.wins:
                flag1 = True

        if not flag1:
            saturated_edges = self.create_network(teamID)
            if solver == "Network Flows":
                flag1 = self.network_flows(saturated_edges)
            elif solver == "Linear Programming":
                flag1 = self.linear_programming(saturated_edges)

        return flag1

    def create_network(self, teamID):
        '''Builds up the network needed for solving the baseball elimination
        problem as a network flows problem & stores it in self.G. Returns a
        dictionary of saturated edges that maps team pairs to the amount of
        additional games they have against each other.

        teamID: ID of team that we want to check if it is eliminated
        return: dictionary of saturated edges that maps team pairs to
        the amount of additional games they have against each other
        '''
        saturated_edges = {}
        rightNodes = list(self.get_team_IDs())
        rightNodes.remove(teamID)
        num_teams = len(rightNodes)

        for i in range(num_teams):
            for j in range(i, num_teams):
                currID = rightNodes[i]
                againstID = rightNodes[j]
                if currID == teamID or againstID == teamID or currID == againstID:
                    continue
                gamesAgainst = self.teams[currID].get_against(againstID)
                saturated_edges[(currID,againstID)] = gamesAgainst

        # set up the graph to run the max flow algorithm on
        self.G.add_nodes_from(['S', 'T'] + list(saturated_edges.keys()) + rightNodes)
        edges = []

        for k,v in saturated_edges.items():
            # flow into node
            edges.append(('S', k,{'capacity':v,'flow':0}))
            # flow out of node
            edges.append((k, k[0],{'capacity':float("infinity"),'flow':0}))
            edges.append((k, k[1],{'capacity':float("infinity"),'flow':0}))

        # Add sink edges
        us = self.teams[teamID]
        ourWins = us.wins + us.remaining
        for n in rightNodes:
            them = self.teams[n]
            diff = ourWins - them.wins
            if diff < 0:
                print("Graph has negative capacity")
            edges.append((n, 'T',{'capacity':diff,'flow':0}))

        self.G.add_edges_from(edges)
        return saturated_edges

    def network_flows(self, saturated_edges):
        '''Uses network flows to determine if the team with given team ID
        has been eliminated. You can feel free to use the built in networkx
        maximum flow function or the maximum flow function you implemented as
        part of the in class implementation activity.

        saturated_edges: dictionary of saturated edges that maps team pairs to
        the amount of additional games they have against each other
        return: True if team is eliminated, False otherwise
        '''
        max_flow, flow_dict = nx.maximum_flow(self.G, 'S', 'T')
        # commented code used to visualize network
        # for edge in self.G.edges():
        #     u, v = edge
        #     self.G.edges[u, v]['flow'] = flow_dict[u][v]
        # self.layout=nx.bipartite_layout(self.G, ['S'])
        # self.draw_graph(self.layout)

        for edge in self.G.edges('S'):
            u, v = edge
            if flow_dict[u][v] < self.G.edges[u, v]['capacity']:
                return True
        return False

    def draw_graph(self, layout):
        """Draws a nice representation of a networkx graph object.
        Source: https://notebooks.azure.com/coells/projects/100days/html/day%2049%20-%20ford-fulkerson.ipynb"""

        plt.figure(figsize=(12, 4))
        plt.axis('off')

        nx.draw_networkx_nodes(self.G, layout, node_color='lightblue', node_size=800)
        nx.draw_networkx_edges(self.G, layout, edge_color='gray', arrowsize=30)
        nx.draw_networkx_labels(self.G, layout, font_color='black', font_size=16)

        for u, v, e in self.G.edges(data=True):
            label = '{}/{}'.format(e['flow'], e['capacity'])
            color = 'green' if e['flow'] < e['capacity'] else 'red'
            x = layout[u][0] * .6 + layout[v][0] * .4
            y = layout[u][1] * .6 + layout[v][1] * .4
            t = plt.text(x, y, label, size=16, color=color,
                         horizontalalignment='center', verticalalignment='center')
        plt.show()

    def linear_programming(self, saturated_edges):
        '''Uses linear programming to determine if the team with given team ID
        has been eliminated. We recommend using a picos solver to solve the
        linear programming problem once you have it set up.
        Do not use the flow_constraint method that Picos provides (it does all of the work for you)
        We want you to set up the constraint equations using picos (hint: add_constraint is the method you want)

        saturated_edges: dictionary of saturated edges that maps team pairs to
        the amount of additional games they have against each other
        returns True if team is eliminated, False otherwise
        '''
        maxflow=pic.Problem()
        # Generate edge capacities.
        c={}
        for e in self.G.edges():
            u, v = e
            capacity = self.G.edges[u, v]['capacity']
            c[(u, v)]  = capacity
        # Convert the capacities to a PICOS expression.
        cc=pic.new_param('c',c)
        s = 'S'
        t = 'T'
        # add the flow variables
        f = {}
        # setting the lower and upper bounds also enforces the edge capacities
        # and nonnegativity constraints
        for e in self.G.edges():
            if c[e] > sys.maxsize:
                f[e] = maxflow.add_variable('f[{0}]'.format(e),1, lower=0)
            else:
                f[e] = maxflow.add_variable('f[{0}]'.format(e),1, lower=0, upper=c[e])
        # add another variable for the total flow
        F = maxflow.add_variable('F',1, lower=0)
        # enforce flow conservation
        for i in self.G.nodes:
            if i == s:
                # this constraint makes sure the flow of what comes out of the
                # source equals the flow
                maxflow.add_constraint(
                    pic.sum([f[p,i] for p in self.G.predecessors(i)],'p','pred(i)') + F
                    == pic.sum([f[i,j] for j in self.G.successors(i)],'j','succ(i)'))
            elif i != t:
                # this constraint makes sure what goes into a node is the same
                # as what comes out of it (if it isn't the source or sink)
                maxflow.add_constraint(
                    pic.sum([f[p,i] for p in self.G.predecessors(i)],'p','pred(i)')
                        == pic.sum([f[i,j] for j in self.G.successors(i)],'j','succ(i)'))
        # Set the objective.
        maxflow.set_objective('max',F)
        # Solve the problem.
        sol = maxflow.solve(verbose=0,solver='cvxopt')
        flow = pic.tools.eval_dict(f) # dictionary mapping edges to flow
        for edge in self.G.edges('S'):
            u, v = edge
            # account for floating point rounding issues
            if not abs(flow[u,v]- self.G.edges[u, v]['capacity']) < 1e-5:
                return True
        return False


    def checkTeam(self, team):
        '''Checks that the team actually exists in this division.
        '''
        if team.ID not in self.get_team_IDs():
            raise ValueError("Team does not exist in given input.")

    def __str__(self):
        '''Returns pretty string representation of a division object.
        '''
        temp = ''
        for key in self.teams:
            temp = temp + f'{key}: {str(self.teams[key])} \n'
        return temp

class Team:
    '''
    The Team class represents one team within a baseball division for use in
    solving the baseball elimination problem. This class includes information
    on how many games the team has won and lost so far this season as well as
    information on what games they have left for the season.

    ID: ID to keep track of the given team
    teamname: human readable name associated with the team
    wins: number of games they have won so far
    losses: number of games they have lost so far
    remaining: number of games they have left this season
    against: dictionary that can tell us how many games they have left against
    each of the other teams
    '''

    def __init__(self, ID, teamname, wins, losses, remaining, against):
        self.ID = ID
        self.name = teamname
        self.wins = wins
        self.losses = losses
        self.remaining = remaining
        self.against = against

    def get_against(self, other_team=None):
        '''Returns number of games this team has against this other team.
        Raises an error if these teams don't play each other.
        '''
        try:
            num_games = self.against[other_team]
        except:
            raise ValueError("Team does not exist in given input.")

        return num_games

    def __str__(self):
        '''Returns pretty string representation of a team object.
        '''
        return f'{self.name} \t {self.wins} wins \t {self.losses} losses \t {self.remaining} remaining'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        division = Division(filename)
        for (ID, team) in division.teams.items():
            print(f'{team.name}: Eliminated? {division.is_eliminated(team.ID, "Linear Programming")}')
    else:
        print("To run this code, please specify an input file name. Example: python baseball_elimination.py teams2.txt.")
