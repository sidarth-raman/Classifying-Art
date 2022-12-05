import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import tensorflow as tf
import csv
import os


#write a function that takes in a directory of images and places images into new directory based on genre

def preprocess_genre():
    pre_preprocess_clean()
    artist_to_genre = {}
    artists = pd.read_csv('../data/artists.csv')

    #remove artists with multiple genres
    artists = artists[~artists["genre"].str.contains(",")].reset_index()
    #keep only name and genre cols in df
    artists = artists[["name", "genre"]]
    #replace spaces with underscores
    artists["name"] = artists["name"].str.replace(" ", "_")
    artists["genre"] = artists["genre"].str.replace(" ", "_")
    artists_names = artists["name"].tolist()
    genre_names = artists["genre"].unique().tolist()


    #create dictionary of artist to genre
    artist_to_genre = {artists["name"][i]: artists["genre"][i] for i in range(len(artists))}

    # print(artist_to_genre)
    #create a new directory for each genre
    for genre in artist_to_genre.values():
        if not os.path.exists("../data/genre_images/" + genre):
            os.makedirs("../data/genre_images/" + genre)
            print("PROCESS: Created genre_images/" + genre)

    # print(artists["name"].tolist())
    #iterate through each image in the directory
    for filename in os.listdir("../data/images"):
        # print(filename)
        if filename in artists_names:
            #get the genre of the artist
            genre = artist_to_genre[filename]
            #copy and move the images to the genre directory
            # print(filename, genre)
            os.system("cp ../data/resized/" + filename + "_* ../data/genre_images/" + genre)
            # os.system("cp ../data/images/" + filename + "/* ../data/genre_images/" + genre + "/")
            print("PROCESS: Copied resized images of " + filename + " to genre_images/" + genre)

    remove_img_genre("../data/genre_images/Impressionism")
    remove_img_genre("../data/genre_images/Post-Impressionism")
    return None

#function randomly removes half of images from given directory
def remove_img_genre(genre_dir):
    #get list of all files in genre directory
    files = os.listdir(genre_dir)
    #randomly select half of the files to remove
    remove = random.sample(files, int(len(files)*(2/3)))
    #remove the files
    for file in remove:
        os.remove(genre_dir + "/" + file)
        print("PROCESS: Removed " + file + " from " + genre_dir)
    return None
    


def pre_preprocess_clean():
    if os.path.exists("../data/genre_images"):
        os.system("rm -r ../data/genre_images")
        print("PROCESS: Removed genre_images")
        print()


if __name__ == '__main__':
    preprocess_genre()
    

    