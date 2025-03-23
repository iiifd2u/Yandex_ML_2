import os
import json
from datetime import datetime
from configparser import ConfigParser
from typing import List

import numpy as np
import torch
import torch.nn as nn

def get_date_id():
    now = datetime.now()
    date_id = now.strftime("%Y%m%d_%H%M%S")
    return date_id


# def save_submission(submission:List, save_dir:str):
#   os.makedirs(save_dir, exist_ok=True)
#   filename = os.path.join(f"submission_{get_date_id()}.txt")
#   with open(os.path.join(save_dir, filename), "w") as f:
#       for line in submission:
#           f.write("{}\n".format(" ".join(list(map(str, map(int, line))))))

def save_submission(submission:List[dict], save_dir:str):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(f"submission_{get_date_id()}.jsonl")
    with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
        for di in submission:
            line = json.dumps(di, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
            f.write(line+"\n")

def save_models(model:nn.Module, optimizer, model_name:str, save_dir:str):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(f"{model_name}_{get_date_id()}.pt")
    try:
        # torch.save(model.state_dict(), os.path.join(save_dir, filename))
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            # 'scheduler_step_dict': scheduler.state_dict()
            }, os.path.join(save_dir, filename))
    except Exception as e:
        print(f"ошибка в сохранении модели {e}")

def load_weights(model:nn.Module, path:str, device:torch.device):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        print(f"Модель успешно загружена из {path}")
    else:
        print("Путь модели не существует!")


def load_config(path: str) ->ConfigParser:
    try:
        config = ConfigParser()
        config.read(path)
        return config
    except Exception as e:
        print(f"ошибка чтения конфига {e}")