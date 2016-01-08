# Given a file of team data and a file of champion data, generates an ARFF file to classify everything.
# 
# Example champion data:
# CHAMPION,TYPE,POKE,AOE,HEAL,CC,DIFFICULTY
# AATROX,FIGHTER,1,1,1,1,1
#
# Sample game data:
# GameId: 1985817861
# Team1(100): Sion, Blitzcrank, Nami, Vel'Koz, Viktor, 
# Team2(200): Corki, Sona, Cho'Gath, Shaco, Jinx, 
# Winner: 200


game_file = 'game.txt'
champ_file = 'champ.txt'
arff_file = 'lol.arff'

# Read champ file into list
open_file = open(champ_file)
champs = [line.rstrip() for line in open_file]
open_file.close()

# Store champ headers in a list
champ_headers = champs[0].split(',')

# Store champ data in a map. ['name' => 'x, y, z...]
champ_data = {}
for i in range(1, len(champs)):
    split_champ = champs[i].split(',')
    champ_data[split_champ[0].lower().replace('_',' ')] = [int(c) for c in split_champ[2:]]
    
# Read game file
with open(game_file) as f:
    open_file = open(game_file)
    games_raw = [line.rstrip() for line in open_file]
    open_file.close()

# Parse game data
# Each team consists of a newline followed by four lines of data, thus: 
    # 
    # GameId: 1985817861
    # Team1(100): Sion, Blitzcrank, Nami, Vel'Koz, Viktor,
    # Team2(200): Corki, Sona, Cho'Gath, Shaco, Jinx,
    # Winner: 200
games = []
for i in range(0, len(games_raw), 5):
    # We won't use the first two lines 
    games.append({})
    games[-1]['team1'] = []
    games[-1]['team2'] = []
    games[-1]['winner'] = '' 
    
    # parse team 1
    team1 = games_raw[i+2][12:-1] # skip 'Team1(100): ' and trailing comma
    team1 = [c.strip().lower() for c in team1.split(',')]
    games[-1]['team1'] = team1

    # parse team 2
    team2 = games_raw[i+3][12:-1] # skip 'Team2(100): ' and trailing comma
    team2 = [c.strip().lower() for c in team2.split(',')]
    games[-1]['team2'] = team2

    # parse the winner of the game
    # print games_raw[i+4][8:]
    games[-1]['winner'] = 'team1' if games_raw[i+4][8:] == '100' else 'team2'
    

# print games

# Open arff file to be written
f = open(arff_file, 'w')

# Write the arff attributes and data
f.write('@RELATION lol\r\n\r\n')

# f.write('@ATTRIBUTE poke1 NUMERIC\r\n')
# f.write('@ATTRIBUTE aoe1  NUMERIC\r\n')
# f.write('@ATTRIBUTE heal1 NUMERIC\r\n')
# f.write('@ATTRIBUTE cc1   NUMERIC\r\n')
# f.write('@ATTRIBUTE diff1 NUMERIC\r\n')
f.write('@ATTRIBUTE tier1 NUMERIC\r\n')
# f.write('@ATTRIBUTE poke2 NUMERIC\r\n')
# f.write('@ATTRIBUTE aoe2  NUMERIC\r\n')
# f.write('@ATTRIBUTE heal2 NUMERIC\r\n')
# f.write('@ATTRIBUTE cc2   NUMERIC\r\n')
# f.write('@ATTRIBUTE diff2 NUMERIC\r\n')
f.write('@ATTRIBUTE tier2 NUMERIC\r\n')
f.write('@ATTRIBUTE victor {team1,team2}\r\n')

f.write('@DATA\r\n') 
for i in range(0, len(games)):
    # Find the average of each team's attribute
    poke1 = aoe1 = heal1 = cc1 = diff1 = tier1 = 0
    poke2 = aoe2 = heal2 = cc2 = diff2 = tier2 = 0

    team1 = games[i]['team1']
    team2 = games[i]['team2']
    for j in range(0, 5):

        if team1[j] not in champ_data:
            print('Missing champion', team2[j])
            continue
        if team2[j] not in champ_data:
            print('Missing champion', team2[j])
            continue

        poke1 += champ_data[team1[j]][0]
        aoe1  += champ_data[team1[j]][1]
        heal1 += champ_data[team1[j]][2]
        cc1   += champ_data[team1[j]][3]
        diff1 += champ_data[team1[j]][4]
        tier1 += champ_data[team1[j]][5]

        poke2 += champ_data[team2[j]][0]
        aoe2  += champ_data[team2[j]][1]
        heal2 += champ_data[team2[j]][2]
        cc2   += champ_data[team2[j]][3]
        diff2 += champ_data[team2[j]][4]
        tier2 += champ_data[team2[j]][5]

    victor = games[i]['winner']

    # f.write('%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%s\r\n' % (poke1/5.0, aoe1/5.0, heal1/5.0,
    #         cc1/5.0, diff1/5.0, tier1/5.0, poke2/5.0, aoe2/5.0, heal2/5.0, cc2/5.0, diff2/5.0, tier2/5.0, victor))
    # f.write('%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%s\r\n' % (poke2/5.0, aoe2/5.0, heal2/5.0,
    #         cc2/5.0, diff2/5.0, tier2/5.0, poke1/5.0, aoe1/5.0, heal1/5.0, cc1/5.0, diff1/5.0, tier1/5.0,
    #                                                         'team1' if victor == 'team2' else 'team2'))
    f.write('%f,%f,%s\r\n' % (tier1, tier2, victor))

f.close()
