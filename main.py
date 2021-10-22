from typing import Optional

from fastapi import FastAPI

import osmnx as ox
import numpy as np
import itertools

app = FastAPI()

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

    return {"category": category, "city": city, "grid_points": grid_points}