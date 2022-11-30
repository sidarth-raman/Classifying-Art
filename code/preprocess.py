import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import tensorflow as tf
import csv

def get_paintings():
    # Read in CSV of Artists
    single_category_artists = set()
    artist_to_genre = {}
    artists = pd.read_csv('../data/artists.csv')

    data = []
    label = []

    for row in artists.iterrows():
        # If there are multiple genres for a painter then don't add the artist
        if "," not in row[1]["genre"]:
            # Replace space with underscore to match data in data folder
            underscore_name = row[1]["name"].replace(' ', '_')
            single_category_artists.add(underscore_name)
            artist_to_genre[underscore_name] = row[1]["genre"]

    images_directory = '../data/images/images/'
    print(artist_to_genre)


    for artist in single_category_artists:
        new_dir = images_directory + artist
        current_genre = artist_to_genre[artist]
        if os.path.exists(new_dir):
            print("Found -->", new_dir)
            for x in os.listdir(new_dir):
                image = plt.imread(new_dir + "/" + x)
                label.append(current_genre)
                data.append(image)


    # Need X0, Y0, X1, Y1 
    print(len(data))
    print(len(label))
    
if __name__ == '__main__':
    get_paintings()

    