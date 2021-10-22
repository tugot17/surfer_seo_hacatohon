from typing import Optional

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_item(category: str, city : str, grid_size=10):
    """
    :param category: which business category
    :param city: in which city
    :param grid_size:  we will divide the city into a grid of size: (grid_size x grid_size)
    :return:
    """
    return {"category": category, "city": city}