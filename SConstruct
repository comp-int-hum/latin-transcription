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
    ("ERROR_ANALYSIS_DIR", "", "work/error_analysis"),
    ("MAX_EPOCHS", "", "100")
    ("RANDOM_SEED", "", "40")
    ("TRAIN_PROPORTION", "", "0.9")
    ("GPU_DEVICES", "", [0]) # GPU device number
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
            "--max_epochs ${MAX_EPOCHS} "
            "--random_seed ${RANDOM_SEED} "
            "--train_proportion ${TRAIN_PROPORTION} "
            "--devices ${DEVICES}"
        ),
        "ExtractErrors" : Builder(
            action="python scripts/extract_errors.py "
            "--input_dir ${SOURCES[0]} "
            "--lines_dir ${SOURCES[1]} "
            "--model ${SOURCES[2]} "
            "--output_dir ${TARGETS[0]} "
            "--random_seed ${RANDOM_SEED} "
            "--train_proportion ${TRAIN_PROPORTION} "
            "--gpu_devices ${GPU_DEVICES}"
        ),

    }
)

lines_data = env.GetLines(
    Dir(env['PROCESSED_DIR']),
    Dir(env['SOURCE_DIR']),
)

model = env.TrainModel(
    "work/model.pkl",
    [Dir(env['SOURCE_DIR']), lines_data],
)

evaluation = env.EvaluateModel(
    [Dir(env['ERROR_ANALYSIS_DIR'])],
    [Dir(env['SOURCE_DIR']), lines_data, model],
)

