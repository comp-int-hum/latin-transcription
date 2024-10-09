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
    ("MODEL_RESULTS_DIR", "", "work/results"),
    ("MAX_EPOCHS", "", 10),
    ("RANDOM_SEED", "", 40),
    ("TRAIN_PROPORTION", "", 0.9),
    ("GPU_DEVICES", "", [0, 1]) # GPU device number
)

env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[],
    
    BUILDERS={
        "GetLines" : Builder(
            action="python scripts/get_lines.py "
            "--input_dir ${SOURCE} --output_dir ${TARGET} "
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
            "--gpu_devices ${GPU_DEVICES}"
        ),
        "ApplyModel" : Builder(
            action="python scripts/apply_model.py "
            "--input_dir ${SOURCES[0]} "
            "--lines_dir ${SOURCES[1]} "
            "--model ${SOURCES[2]} "
            "--output_dir ${MODEL_RESULTS_DIR} "
            "--random_seed ${RANDOM_SEED} "
            "--train_proportion ${TRAIN_PROPORTION} "
            "--gpu_devices ${GPU_DEVICES}"
        ),

        "EvaluateResults" : Builder(
            action="python scripts/evaluate_results.py "
            "--input_dir ${SOURCE} "
            "--output ${TARGET} "
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

# Note - steamroller does not support creating dir nodes automatically
evaluation = env.ApplyModel(
    [Dir(env['MODEL_RESULTS_DIR']+ "/val"), Dir(env['MODEL_RESULTS_DIR']+ "/train")],
    [Dir(env['SOURCE_DIR']), lines_data, model],
)

val_results, train_results = evaluation

val_report = env.EvaluateResults(
    "${MODEL_RESULTS_DIR}/val_results.txt",
    val_results,
)

train_report = env.EvaluateResults(
    "${MODEL_RESULTS_DIR}/train_results.txt",
    train_results,
)



