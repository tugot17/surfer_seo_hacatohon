from typing import Optional

from fastapi import FastAPI

import osmnx as ox
import numpy as np
import itertools
import pandas as pd
import numpy as np
import json
from scipy.spatial import distance
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()
origins = [
    "http://192.168.43.17",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv('places.csv.gz', compression='gzip')
places = df[(df['language'] == 'pl')]

def get_queries_location(places_df, category, city_bounding_box):
    df = places_df[places_df.category == category]
    df = df[(df['audit_latitude'] > city_bounding_box[1]) & (df['audit_latitude'] < city_bounding_box[3])]
    df = df[(df['audit_longitude'] < city_bounding_box[2]) & (df['audit_longitude'] > city_bounding_box[0])]

    return df


def get_places_location(places_df, category, city_bounding_box):
    df = places_df[places_df.category == category]
    df = df[(df['place_latitude'] > city_bounding_box[1]) & (df['place_latitude'] < city_bounding_box[3])]
    df = df[(df['place_longitude'] < city_bounding_box[2]) & (df['place_longitude'] > city_bounding_box[0])]

    places_array = []
    for i, row in df.iterrows():
        places_array.append([row['place_latitude'], row['place_longitude']])
    return places_array


def get_closest_place(query_location, places):
    min_distance = float("inf")
    closest_location = ''
    for place_location in places:
        d = distance.euclidean(place_location, query_location)
        if d < min_distance:
            min_distance = d
            closest_location = place_location
    return closest_location


def get_best_location(places_df, city_bbox, category, grids):
    # city_item = read_item(category, city, grid_size)
    # grids = city_item["grid_points"]
    # city_bbox = city_item["city_bbox"]

    grids_counts = [0] * len(grids)  # list of grids with counts of closest queries
    queries_locations_df = get_queries_location(places_df, category, city_bbox)

    places = grids.copy()
    places.extend(get_places_location(places_df, category, city_bbox))

    for _, query_location in queries_locations_df.iterrows():
        query_latitude = query_location['audit_latitude']
        query_longitude = query_location['audit_longitude']

        query_location = [query_longitude, query_latitude]
        place = get_closest_place(query_location, places)
        if place in grids:
            index = -1
            for i, g in enumerate(grids):
                if place == g:
                    index = i
            grids_counts[index] = grids_counts[index] + 1

    if len(set(grids_counts)) == 1:
        return -1

    return grids[np.argmax(grids_counts)]


@app.get("/")
def read_item(category: str, city : str, grid_size: int = 10):
    """
    :param category: which business category
    :param city: in which city
    :param grid_size:  we will divide the city into a grid of size: (grid_size x grid_size)
    :return:
    """
    gdf = ox.geocode_to_gdf({'city': city})
    geom = gdf.loc[0, 'geometry']
    city_bbox = geom.bounds

    longitudes = list(np.linspace(city_bbox[0], city_bbox[2], grid_size))
    latitudes = list(np.linspace(city_bbox[1], city_bbox[3], grid_size))

    grid_points = list(itertools.product(longitudes, latitudes))

    location = get_best_location(places, city_bbox, category, grid_points)

    return {"category": category, "city": city, "location": location, "center": grid_points[len(grid_points)//2]}