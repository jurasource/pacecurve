# Goal

1. To be able to predict, in “real time”, what distance someone will run in a timed track race (24 hours is the main use case) using intermediate distance updates at random intervals
2. Using PCA analysis (or whatever technique is appropriate) to determine ~3 “pace/fatigue” profile curves from the historical lap times of previous races (currently have 5 years of data, each race has ~45 runners)
3. Create a prediction model and use the historical data to test it against

# Supplementary/intermediate goals

1. Visualise the existing dataset (and all derived pace curves and backtesting results) in an interactive website using pure python (eg streamlit or a similar library/framework). 
2. Deploy everything to an external server so it can be made available on the web

# Future goals 

1. Incorporate historical weather conditions into the model based on the date and location
2. Scrape additional historical race results to use for training
3. Use Strava data to create a “fitness” factor for the model and incorporate that into real time predictions 
4. Extend the capability to include fixed distance races, and use both historical results and elevation profile to determine pace/fatigue curves
