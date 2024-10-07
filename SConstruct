import os
import os.path
from steamroller import Environment
from os import path
from glob import glob


vars = Variables("custom.py")
vars.AddVariables(
    ("SOURCE_DIR", "", "source"),
    ("PROCESSED_DIR", "", "work/processed_data"),
    ("CHECKPOINT_DIR", "", "work/checkpoints"),
    ("MAX_EPOCHS", "", "100")
)

env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[],
    
    BUILDERS={
        "GetLines" : Builder(
            action="python scripts/get_lines.py "
            "--input_dir ${SOURCE} --output_dir ${PROCESSED_DIR} "
        ),
        "TrainModel" : Builder(
            action="python scripts/train.py "
            "--input_dir ${SOURCES[0]} "
            "--lines_dir ${SOURCES[1]} "
            "--output ${TARGET} "
            "--checkpoint_dir ${CHECKPOINT_DIR} "
            "--max_epochs ${MAX_EPOCHS}"
        ),
    }
)

lines_data = env.GetLines(
    Dir(env['PROCESSED_DIR']),
    Dir(env['SOURCE_DIR'])
    
)

model = env.TrainModel(
    "work/model.pkl",
    [Dir(env['SOURCE_DIR']), lines_data],
)
