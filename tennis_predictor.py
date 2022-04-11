import pandas as pd
import numpy as np
from scipy import stats

class TennisOddsSim:
    def __init__(self, df, p1_name, p2_name, p1_rank_points=-1, p2_rank_points=-1, data_range=(0, -1), n_sims=1000, treshold=10):
        df = df.iloc[data_range[0]:data_range[1]].copy()
        self.p1_name = p1_name
        self.p2_name = p2_name
        self.total_sets = []
        self.total_games = []
        self.p1_won = 0 # counter for p1 wins
        self.n_sims = n_sims
        self.current_iteration = 0
        self.can_start = True
        
        p1w_df = df[df['winner_name'] == p1_name].copy()
        p1l_df = df[df['loser_name'] == p1_name].copy()
        p2w_df = df[df['winner_name'] == p2_name].copy()
        p2l_df = df[df['loser_name'] == p2_name].copy()
        
        self.p1_df = self.merge_w_l(p1w_df, p1l_df) # indexes aren't reset after this, correct it
        self.p2_df = self.merge_w_l(p2w_df, p2l_df)
        
        
        if p1_rank_points > 0:
            self.p1_rank_points = p1_rank_points
        elif len(self.p1_df) > 0:
            self.p1_rank_points = self.p1_df.iloc[-1]['p_rank_points']
        else:
            self.p1_rank_points = 1
            
        if p1_rank_points > 0:
            self.p2_rank_points = p2_rank_points
        elif len(self.p2_df) > 0:
            self.p2_rank_points = self.p2_df.iloc[-1]['p_rank_points']
        else:
            self.p2_rank_points = 1
            
        self.treshold = treshold
        
        self.p1_reg = self.set_regressions(self.p1_df,p2_rank_points)
        self.p2_reg = self.set_regressions(self.p2_df,p1_rank_points)
    
    
    def merge_w_l(self, win_df, loss_df):
        """
        win_df: DataFrame containting players winning matches
        loss_df: DataFrame containting players losing matches
        """
        
        # format the column names in the database,
        # so all names are equal in a way that they start with w or winner for current player
        loss_df_cols = loss_df.columns.values.copy()
        for i in range(len(loss_df_cols)):
            word = loss_df_cols[i].split('_')[0]
            if word == 'l' or word == 'loser':
                loss_df_cols[i] = ('w' if word == 'l' else 'winner') + loss_df_cols[i][len(word):]
            if word == 'w' or word == 'winner':
                loss_df_cols[i] = ('l' if word == 'w' else 'loser') + loss_df_cols[i][len(word):]
        loss_df.columns = loss_df_cols
        
        player_df = pd.concat([win_df, loss_df])
        cols = player_df.columns.values.copy()
        pid = 'p'
        oppid = 'o'
        for i in range(len(cols)):
            word = cols[i].split('_')[0]
            if word == 'w' or word == 'winner':
                cols[i] = pid + cols[i][len(word):]
            elif word == 'l' or word == 'loser':
                cols[i] = oppid + cols[i][len(word):]
        player_df.columns = cols
        
        player_df = player_df.sort_values(by='tourney_date').drop('tourney_date', axis=1).copy()
        self.to_relative(player_df)
        
        return player_df
    
    
    def to_relative(self, data):
        data['p_ace'] = data['p_ace'] / data['p_svpt']
        data['p_df'] = data['p_df'] / (data['p_svpt'] - data['p_1stIn']) # fault given it's a second serve
        data['p_1stWon'] = data['p_1stWon'] / data['p_1stIn']
        data['p_2ndWon'] = data['p_2ndWon'] / (data['p_svpt'] - data['p_1stIn'])
        data['p_bpSaved'] = data['p_bpSaved'] / data['p_bpFaced']
        data['p_1stIn'] = data['p_1stIn'] / data['p_svpt']

        data['o_1stWon'] = data['o_1stWon'] / data['o_1stIn']
        data['o_2ndWon'] = data['o_2ndWon'] / (data['o_svpt'] - data['o_1stIn'])
        data['o_bpSaved'] = data['o_bpSaved'] / data['o_bpFaced']
    
    
    def set_regressions(self, player_data, opp_rank_points):
        stat_regressions = {}
        stat_types = ['p_1stIn', 'p_df', 'p_ace', 'p_bpSaved', 'p_1stWon', 'p_2ndWon', 'o_bpSaved', 'o_1stWon', 'o_2ndWon']
        
        for stat in stat_types:
            x = player_data[[stat, 'o_rank_points']].dropna().iloc[-50:].values
            if len(x) < self.treshold:
                self.p1_won = np.nan
                self.can_start = False
                break
            y = x[:,0].copy()
            x = x[:,1].copy()
            stat_regressions[stat] = stats.linregress(x, y)
        
        return stat_regressions
    
    

    def get_values(self, reg, opp_rank_points):
        values = {}
        
        for key in reg.keys():
            value_mean = np.random.normal(reg[key].slope, reg[key].stderr) * opp_rank_points + reg[key].intercept
            value_std = reg[key].intercept_stderr
            values[key] = np.random.normal(loc=value_mean,scale=value_std)
        
        return values
    
    def sim_match(self):
        matches = [] # move this out to object
        p1 = self.get_values(self.p1_reg,self.p2_rank_points)
        p2 = self.get_values(self.p2_reg,self.p1_rank_points)
        match = TennisMatch([p1.copy(),p2.copy()])
        while match.status != 'end':
            match.generate_point()
#         print(match.score)
        matches += [match.score]
        self.p1_won += 1 if match.p[0]['sets'] == 2 else 0
        
        # ADD THIS LATER!!!!
#         total_sets += [match.p[0]['sets'] + match.p[1]['sets']]
#         total_games += [match.total_games]
    
    def start(self):
        if self.can_start == False:
            return np.nan
        for i in range(self.n_sims):
#             print(i) # remove this later
            self.current_iteration += 1
            self.sim_match()
    
    def reset(self):
        self.current_iteration = 0
        self.p1_won = 0


class TennisMatch():
    """
    Class used to track tennis match score
    """
    def __init__(self, players, verbose=False):
        self.p = [{'sets': 0, 'games': 0, 'points':0}, {'sets': 0, 'games': 0, 'points':0}]
        self.serve = 1
        self.status = 'game'
        self.bp = False
        self.score = ''
        self.player_serving = np.random.randint(2)
        self.players = players.copy()
        for i in range(2):
            # estimate the statistics for each player, 1st serve won, 2nd serve won and break points saved
            self.players[i]['p_bpSaved'] = (self.players[i]['p_bpSaved'] + self.players[-i + 1]['o_bpSaved']) / 2
            self.players[i]['p_1stWon'] = (self.players[i]['p_1stWon'] + self.players[-i + 1]['o_1stWon']) / 2
            self.players[i]['p_2ndWon'] = (self.players[i]['p_2ndWon'] + self.players[-i + 1]['o_2ndWon']) / 2
        self.total_games = 0
    
    def add_point(self, player):
        if self.status == 'end':
            return
        self.p[player]['points'] += 1
        if self.p[-self.player_serving + 1]['points'] >= 4 and self.p[-self.player_serving + 1]['points'] - self.p[self.player_serving]['points'] == 1:
            self.bp = True
        else:
            self.bp = False
        
        # check if game over
        if self.p[player]['points'] >= 4 and self.p[player]['points'] >= self.p[-player + 1]['points'] and self.p[player]['points'] - self.p[-player + 1]['points'] > 1:
            self.p[player]['points'] = 0
            self.p[-player + 1]['points'] = 0
            self.p[player]['games'] += 1
            self.player_serving = -self.player_serving + 1
        
        # check if set over
        if (self.p[player]['games'] == 6 and self.p[-player + 1]['games'] <= 4) or self.p[player]['games'] == 7:
            y = self.p[player]['games']
            z = self.p[-player + 1]['games']
            self.score += f'{y}-{z},' if player == 0 else f'{z}-{y},'
            self.total_games += self.p[player]['games'] + self.p[-player + 1]['games']
            self.p[player]['games'] = 0
            self.p[-player + 1]['games'] = 0
            self.p[player]['sets'] += 1
        if self.p[player]['sets'] == 2:
            self.status = 'end'
            self.winner = player
    
    def generate_point(self):
        if self.serve_result() == False:
            self.serve = 2
            if self.serve_result() == False:
                self.add_point(-self.player_serving + 1)
                self.serve = 1
                return
        if self.server_wins_point() == True:
            self.add_point(self.player_serving)
        else:
            self.add_point(-self.player_serving + 1)
        self.serve = 1
    
    def server_wins_point(self):
        rand = np.random.uniform()
        if self.bp == True:
            odds = self.players[self.player_serving]['p_bpSaved']
            return odds >= rand
        serve = 'p_1stWon' if self.serve == 1 else 'p_2ndWon'
        odds = self.players[self.player_serving][serve]
        return odds > rand
    
    def serve_result(self):
        rand = np.random.uniform()
        serve = 'p_1stIn' if self.serve == 1 else 'p_df'
        odds = self.players[self.player_serving][serve]
        return odds > rand