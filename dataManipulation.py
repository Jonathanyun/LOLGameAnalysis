import pandas as pd 
import os 
import csv

PATH = os.getcwd()
PATH = PATH + "/games.csv"


data = pd.read_csv(PATH)
print(data.head())
df = data[['gameId', 'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald', 'winner']]
df.to_csv('trainingGames.csv')

df2 = data[['gameId', 'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald']]
df2['winner'] = 0
df2.to_csv('testingGames.csv')