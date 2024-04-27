import datetime
from typing import Union, List

import numpy as np 

from fastapi import FastAPI
from pydantic import BaseModel


class Sample(BaseModel):
    obs: List[List[float]]
    policy_tgt: List[List[float]]
    value_tgt: List[int]

app = FastAPI()


@app.post("/game/{game_id}/")
async def game(sample: Sample):
    #assert abs(np.sum(sample.policy_tgt) - len(sample.policy_tgt)) <= 1e-05
    assert sample.value_tgt[-1] != -1

    print(f"Number of new samples: {len(sample.obs)}")

    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=0)))
    np.savez_compressed(f'data/run1/training-run1-{now.strftime("%Y%m%d")}-{now.strftime("%H%M")}', obs=sample.obs, policy_tgt=sample.policy_tgt, value_tgt=sample.value_tgt)

    return {"message": "Success!"}
