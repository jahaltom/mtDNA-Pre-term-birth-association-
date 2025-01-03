import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
#import pyarrow

cOm=["C","M"]

for f in cOm:

    df_merged=pd.read_csv('forPlotting.'+f+'csv',sep='\t')
    
    
    
    ##########################################Population  
    # Step 3: Map colors to races
    race_to_color = {'SouthAsian': 'blue', 'African': 'green'}  # Define color mapping
    df_merged['Color'] = df_merged['population'].map(race_to_color)
    
    # Step 4: Plot the data
    plt.figure(figsize=(25, 6))
    
    # Scatter plot colored by race
    for race in df_merged['population'].unique():
        subset = df_merged[df_merged['population'] == race]
        plt.scatter(subset['POS'], subset['Frequency'], 
                    label=race, c=subset['Color'], alpha=0.7, s=50)
    
    # Add labels, legend, and title
    plt.xlabel('Position')
    plt.ylabel('Frequency')
    plt.title('Position vs Frequency Colored by population Metadata')
    plt.legend(title='population')
    plt.grid(True)
    plt.xticks(np.arange(0, 17001, 1000)) 
    # Show the plot
    plt.show()
    plt.savefig("Heteroplasmy."+f+".Population.png", bbox_inches="tight")
    plt.clf()      
    ##########################################PTB
    # Step 3: Map colors to ptb status
    ptb = {0.0: 'blue', 1.0: 'red'}  # Define color mapping
    df_merged['Color'] = df_merged['PTB'].map(ptb)
    
    # Step 4: Plot the data
    plt.figure(figsize=(25, 6))
    
    # Scatter plot colored by ptb
    for ps in df_merged['PTB'].unique():
        subset = df_merged[df_merged['PTB'] == ps]
        plt.scatter(subset['POS'], subset['Frequency'], 
                    label=ps, c=subset['Color'], alpha=0.7, s=50)
    
    # Add labels, legend, and title
    plt.xlabel('Position')
    plt.ylabel('Frequency')
    plt.title('Position vs Frequency Colored by PTB Metadata')
    plt.legend(title='PTB')
    plt.grid(True)
    plt.xticks(np.arange(0, 17001, 1000))
    # Show the plot
    plt.show()
    plt.savefig("Heteroplasmy."+f+".PTB.png", bbox_inches="tight")
    plt.clf()   
