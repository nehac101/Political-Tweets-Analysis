project = "TweetAnalysis" # don't edit this

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np

def makeDataFrame(filename):
    return pd.read_csv(filename)

def parseName(fromString):
    start = fromString.find(":")
    end = fromString.find("(")
    name = fromString[start+1:end]
    return name.strip()

def parsePosition(fromString):
    start = fromString.find("(")
    end = fromString.find("from")
    position = fromString[start+1:end]
    return position.strip()

def parseState(fromString):
    start = fromString.find("from")
    end = fromString.find(")")
    state = fromString[start+5:end]
    return state

def getRegionFromState(stateDf, state):
    row = stateDf.loc[stateDf['State'] == state, 'Region']
    return row.values[0]

def addColumns(data, stateDf):
    names = []
    positions = []
    states = []
    regions = []
    for index, row in data.iterrows():
        value = row["label"]
        name = parseName(value)
        position = parsePosition(value)
        state = parseState(value)
        region = getRegionFromState(stateDf, state)
        names.append(name) 
        positions.append(position)
        states.append(state)
        regions.append(region)
    data["Name"] = names
    data["Position"] = positions
    data["State"] = states
    data["Region"] = regions
    return None


def doWeek1():
    df = makeDataFrame("data/politicaldata.csv")
    stateDf = makeDataFrame("data/statemappings.csv")
    addColumns(df, stateDf)
    print("Updated dataframe:")
    print(df)

#test functions taken from cmu 15-110 website

def testMakeDataFrame():
    print("Testing makeDataFrame()...", end="")
    df = makeDataFrame("data/politicaldata.csv")
    assert(type(df) == pd.core.frame.DataFrame)
    assert(df.size == 89640)
    stateDf = makeDataFrame("data/statemappings.csv")
    assert(type(stateDf) == pd.core.frame.DataFrame)
    assert(stateDf.size == 204)
    print("... done!")

def testParseName():
    print("Testing parseName()...", end="")
    assert(parseName("From: Steny Hoyer (Representative from Maryland)") == "Steny Hoyer")
    assert(parseName("From: Mitch (Senator from Kentucky)") == "Mitch")
    assert(parseName("From: Stephanie Rosenthal (Prof from PA)") == "Stephanie Rosenthal")
    assert(parseName("From: Kelly (Senator from Pennsylvania)") == "Kelly")
    print("...done!")

def testParsePosition():
    print("Testing parsePosition()...", end="")
    assert(parsePosition("From: Steny Hoyer (Representative from Maryland)") == "Representative")
    assert(parsePosition("From: Mitch (Senator from Kentucky)") == "Senator")
    assert(parsePosition("From: Stephanie Rosenthal (Prof from PA)") == "Prof")
    assert(parsePosition("From: Kelly (Senator from Pennsylvania)") == "Senator")
    print("...done!")

def testParseState():
    print("Testing parseState()...", end="")
    assert(parseState("From: Steny Hoyer (Representative from Maryland)") == "Maryland")
    assert(parseState("From: Mitch (Senator from Kentucky)") == "Kentucky")
    assert(parseState("From: Stephanie Rosenthal (Prof from PA)") == "PA")
    assert(parseState("From: Kelly (Senator from Pennsylvania)") == "Pennsylvania")
    print("...done!")

def testGetRegionFromState():
    print("Testing getRegionFromState()...", end="")
    stateDf = makeDataFrame("data/statemappings.csv")
    assert(str(getRegionFromState(stateDf, "California")) == "West")
    assert(str(getRegionFromState(stateDf, "Maine")) == "Northeast")
    assert(str(getRegionFromState(stateDf, "Nebraska")) == "Midwest")
    assert(str(getRegionFromState(stateDf, "Texas")) == "South")
    print("...done!")

def testAddColumns():
    print("Testing addColumns()...", end="")
    df = makeDataFrame("data/politicaldata.csv")
    stateDf = makeDataFrame("data/statemappings.csv")
    addColumns(df, stateDf)
    assert(df["Name"][1] == "Mitch McConnell")
    assert(df["Name"][4979] == "Ted Yoho")
    assert(df["Position"][1] == "Senator")
    assert(df["Position"][4979] == "Representative")
    assert(df["State"][1] == "Kentucky")
    assert(df["State"][4979] == "Florida")
    assert(df["Region"][1] == "South")
    assert(df["Region"][4979] == "South")
    print("... done!")

def testWeek1():
    testMakeDataFrame()
    testParseName()
    testParsePosition()
    testParseState()
    testGetRegionFromState()
    testAddColumns()

testWeek1()
doWeek1()

#### CHECK-IN 2 ####

def findSentiment(classifier, tweet):
    score = classifier.polarity_scores(tweet)['compound']
    if score < -0.1: 
        return "negative"
    elif score > 0.1: 
        return "positive"
    else: 
        return "neutral"

def addSentimentColumn(data):
    classifier = SentimentIntensityAnalyzer()
    sentiments = []
    for index, row in data.iterrows():
        tweet = row["text"]
        senti = findSentiment(classifier, tweet)
        sentiments.append(senti)
    data["Sentiment"] = sentiments
    return None

def getNegSentimentByState(data):
    d = dict()
    for index, row in data.iterrows():
        state = row["State"]
        senti = row["Sentiment"]
        if senti == "negative":
            if state not in d: 
                d[state] = 1
            else:
                d[state]+=1
    return d

def getAttacksByState(data):
    d = dict()
    for index, row in data.iterrows():
        state = row["State"]
        msg = row["message"]
        if msg == "attack":
            if state not in d: 
                d[state] = 1
            else:
                d[state]+=1
    return d

def getPartisanByState(data):
    d = dict()
    for index, row in data.iterrows():
        state = row["State"]
        bias = row["bias"]
        if bias == "partisan":
            if state not in d: 
                d[state] = 1
            else:
                d[state]+=1
    return d  

def getMessagesByRegion(data):
    d = dict()
    for index, row in data.iterrows():
        region = row["Region"]
        msg = row["message"]
        if region not in d:
            d[region] = {msg: 1}
        else:
            if msg not in d[region]:
                d[region][msg] =1
            else:
                d[region][msg] += 1
    return d

def getAudienceByRegion(data):
    d = dict()
    for index, row in data.iterrows():
        region = row["Region"]
        audience = row["audience"]
        if region not in d:
            d[region] = {audience: 1}
        else:
            if audience not in d[region]:
                d[region][audience] =1
            else:
                d[region][audience] += 1
    return d

def doWeek2():
    df = makeDataFrame("data/politicaldata.csv")
    stateDf = makeDataFrame("data/statemappings.csv")
    addColumns(df, stateDf)
    addSentimentColumn(df)

    negSentiments = getNegSentimentByState(df)
    attacks = getAttacksByState(df)
    parisanship = getPartisanByState(df)
    messages = getMessagesByRegion(df)
    audiences = getAudienceByRegion(df)

#test functions taken from cmu 15-110 website

def testFindSentiment():
    print("Testing findSentiment()...", end="")
    classifier = SentimentIntensityAnalyzer()
    assert(findSentiment(classifier, "great") == "positive")
    assert(findSentiment(classifier, "bad") == "negative")
    assert(findSentiment(classifier, "") == "neutral")
    print("...done!")

def testAddSentimentColumn():
    print("Testing addSentimentColumn()...", end="")
    df = makeDataFrame("data/politicaldata.csv")
    addSentimentColumn(df)
    assert(df["Sentiment"][0] == "neutral")
    assert(df["Sentiment"][1] == "negative")
    assert(df["Sentiment"][4978] == "positive")
    print("... done!")

def testGetNegSentimentByState(df):
    print("Testing getNegSentimentByState()...", end="")
    d = getNegSentimentByState(df)
    assert(len(d) == 49)
    assert(d["Pennsylvania"] == 48)
    assert(d["North Dakota"] == 3)
    assert(d["Louisiana"] == 20)
    print("...done!")

def testGetAttacksByState(df):
    print("Testing getAttacksByState()...", end="")
    d = getAttacksByState(df)
    assert(len(d) == 37)
    assert(d["Pennsylvania"] == 9)
    assert(d["Maryland"] == 4)
    assert(d["Nevada"] == 1)
    print("...done!")

def testGetPartisanByState(df):
    print("Testing getPartisanByState()...", end="")
    d = getPartisanByState(df)
    assert(len(d) == 50)
    assert(d["Pennsylvania"] == 40)
    assert(d["Maryland"] == 44)
    assert(d["Nevada"] == 10)
    print("...done!")

def testGetMessagesByRegion(df):
    print("Testing getMessagesByRegion()...", end="")
    d = getMessagesByRegion(df)
    assert(len(d) == 4)
    assert(len(d["South"]) == 9)
    assert(d["South"]["policy"] == 563)
    assert(d["Northeast"]["attack"] == 23)
    print("...done!")

def testGetAudienceByRegion(df):
    print("Testing getAudienceByRegion()...", end="")
    d = getAudienceByRegion(df)
    assert(len(d) == 4)
    assert(len(d["South"]) == 2)
    assert(d["South"]["national"] == 1561)
    assert(d["Midwest"]["constituency"] == 265)
    assert(d["Northeast"]["national"] == 682)
    print("...done!")

def testWeek2():
    testFindSentiment()
    testAddSentimentColumn()
    
    df = makeDataFrame("data/politicaldata.csv")
    stateDf = makeDataFrame("data/statemappings.csv")
    addColumns(df, stateDf)
    addSentimentColumn(df)
    
    testGetNegSentimentByState(df)
    testGetAttacksByState(df)
    testGetPartisanByState(df)
    testGetMessagesByRegion(df)
    testGetAudienceByRegion(df)

testWeek2()
doWeek2()


def graphAttacksAllStates(dict):
    barPlot(dict, "Graph Attacks Across All States")
    return

def graphTopN(dict, n, title):
    newD = {}
    numbers = list(dict.values())
    sortedVals = list(reversed(sorted(numbers)))
    for i in range(n):
        value = sortedVals[i]
        index = numbers.index(value)
        newD[list(dict.keys())[index]] = value
    barPlot(newD, "Attack Tweets for top N States")


def graph2Regions(dict, r1, r2, title):
    allKeys = list(dict[r1].keys())
    r1Vals =[]
    r2Vals = []
    for key in dict[r2]: 
        if key not in allKeys:
            allKeys.append(key)
    for key in allKeys:
        if key in dict[r1]:
            r1Vals.append(dict[r1][key])
            if key in dict[r2]:
                r2Vals.append(dict[r2][key])
            else:
                r2Vals.append(0)
        else:
            r2Vals.append(dict[r2][key])
            if key in dict[r1]:
                r1Vals.append(dict[r1][key])
            else:
                r1Vals.append(0)
    sideBySideBarPlots(allKeys, r1Vals, r2Vals, r1, r2, "Comparing Types of Tweets from Two American Regions")
    return

def graphSentCountAttackCount(sentiments, attacks, title):
    allKeys = list(sentiments.keys())
    senti =[]
    attac = []
    for key in attacks: 
        if key not in allKeys:
            allKeys.append(key)
    for key in allKeys:
        if key in sentiments:
            senti.append(sentiments[key])
            if key in attacks:
                attac.append(attacks[key])
            else:
                attac.append(0)
        else:
            attac.append(attacks[key])
            if key in sentiments:
                senti.append(sentiments[key])
            else:
                senti.append(0)
    sideBySideBarPlots(allKeys, senti, attac, 'sentiments', 'attacks', "Sentiment Counts vs Attack Counts ")
    return


#test functions taken from cmu 15-110 website
def doWeek3():
    df = makeDataFrame("data/politicaldata.csv")
    stateDf = makeDataFrame("data/statemappings.csv")
    addColumns(df, stateDf)
    addSentimentColumn(df)
    
    nsbs = getNegSentimentByState(df)
    abs = getAttacksByState(df)
    pbs = getPartisanByState(df)
    mbr = getMessagesByRegion(df)
    abr = getAudienceByRegion(df)
    
    graphAttacksAllStates(abs)
    graphTopN(abs, 5, "Top Attacks")
    graphTopN(pbs, 5, "Top Partisan Messages")
    graph2Regions(mbr, "West", "South", "Messages by Region")
    graph2Regions(abr, "West", "South", "Audience by Region")
    graphSentCountAttackCount(nsbs, abs, "Sentiment vs Attacks by State")
    
#following functions taken from cmu 15-110 website

"""
Expects a dictionary of states as keys with counts as values, and a title.
Plots the states on the x axis, counts as the y axis and puts a title on top.
"""
def barPlot(dict, title):
    names = list(dict.keys())
    values = list(dict.values())
    plt.bar(names, values)
    plt.xticks(names, rotation='vertical')
    plt.title(title)
    plt.show()

"""
Expects 3 lists - one of names, and two of values such that the index of a name
corresponds to a value at the same index in both lists. Category1 and Category2
are the labels for the different colors in the graph. For example, you may use
it to graph two categories of counts side by side to look at the differences.
"""
def sideBySideBarPlots(names, values1, values2, category1, category2, title):
    x = list(range(len(names)))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    pos1 = []
    pos2 = []
    for i in x:
        pos1.append(i - width/2)
        pos2.append(i + width/2)
    rects1 = ax.bar(pos1, values1, width, label=category1)
    rects2 = ax.bar(pos2, values2, width, label=category2)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    plt.title(title)
    plt.xticks(rotation="vertical")
    fig.tight_layout()
    plt.show()

#### WEEK 3 TESTS ####

# Instead of running individual tests, check the new graph generated by doWeek3
# after you finish each function.

doWeek3()