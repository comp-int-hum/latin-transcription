import os
import os.path
from steamroller import Environment



vars = Variables("custom.py")
vars.AddVariables(
    # Directories
    ("SOURCE_DIR", "", "source"),
    ("WORK_DIR", "", "work"),
    ("UPDATED_LINE_POLYGONS_DIR", "", "/updated_polygons"),
    ("PROCESSED_DIR", "", "/processed_data"),
    ("CHECKPOINT_DIR", "", "/checkpoints"),
    ("MODEL_RESULTS_DIR", "", "/results"),
    ("MODEL_ERRORS_DIR", "", "/errors"),

    # Data 
    ("DATA_CUTOFF", "", -1),
    ("RANDOM_SEED", "", 40),

    # Model 
    ("MAX_EPOCHS", "", 20),
    ("TRAIN_PROPORTION", "", 0.9),
    ("GPU_DEVICES", "", [0]), # GPU device number
    
    # Logging 
    ("WANDB", "", True),

    # Steamroller 
    ("STEAMROLLER_ENGINE", "", "slurm"),
    ("CPU_QUEUE", "", "parallel"),
    ("CPU_ACCOUNT", "", "tlippin1"),    
    ("GPU_QUEUE", "", "a100"),
    ("GPU_ACCOUNT", "", "tlippin1_gpu"),
    ("GPU_COUNT", "", 1),
    ("MEMORY", "", "64GB"),
)

env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[],
    
    BUILDERS={

        "GetLines" : Builder(
            action="python scripts/get_lines.py "
            "--image_dir ${SOURCES[0]} "
            "--xml_dir ${SOURCES[1]} "
            "--output_dir ${TARGET} "
        ),
        "TrainModel" : Builder(
            action="python scripts/train.py "
            "--input_dir ${SOURCES[0]} "
            "--lines_dir ${SOURCES[1]} "
            "--output ${TARGETS[0]} "
            "--checkpoint_dir ${TARGETS[1]} "
            "--max_epochs ${MAX_EPOCHS} "
            "--random_seed ${RANDOM_SEED} "
            "--train_proportion ${TRAIN_PROPORTION} "
            "--gpu_devices ${GPU_DEVICES} "
            "--data_cutoff ${DATA_CUTOFF} "
            "--use_wandb ${WANDB} "
            "--wandb_name ${WANDB_NAME} "
        ),
        "ApplyModel" : Builder(
            action="python scripts/apply_model.py "
            "--input_dir ${SOURCES[0]} "
            "--lines_dir ${SOURCES[1]} "
            "--model ${SOURCES[2]} "
            "--output_dir ${MODEL_RESULTS_DIR} "
            "--random_seed ${RANDOM_SEED} "
            "--train_proportion ${TRAIN_PROPORTION} "
            "--gpu_devices ${GPU_DEVICES} "
        ),

        "GenerateReport" : Builder(
            action="python scripts/generate_report.py "
            "--input_dir ${SOURCE} "
            "--output ${TARGET} "
        ),

        "ExtractErrors" : Builder(
            action="python scripts/extract_errors.py "
            "--input_dir ${SOURCE} "
            "--output_dir ${TARGET} "
        ),

    }
)

def cpu_task_config(name, time_required, memory_required=env["MEMORY"]):
    return {
        "STEAMROLLER_ACCOUNT": env["CPU_ACCOUNT"],
        "STEAMROLLER_QUEUE": env["CPU_QUEUE"],
        "STEAMROLLER_TIME": time_required,
        "STEAMROLLER_MEMORY": memory_required,
        "STEAMROLLER_NAME_PREFIX": f"{name}",
        "STEAMROLLER_ENGINE": env["STEAMROLLER_ENGINE"],
    }

def gpu_task_config(name, time_required, memory_required=env["MEMORY"]):
    return {
        "STEAMROLLER_ACCOUNT": env["GPU_ACCOUNT"],
        "STEAMROLLER_QUEUE": env["GPU_QUEUE"],
        "STEAMROLLER_TIME": time_required,
        "STEAMROLLER_MEMORY": memory_required,
        "STEAMROLLER_NAME_PREFIX": f"{name}",
        "STEAMROLLER_ENGINE": env["STEAMROLLER_ENGINE"],
        "STEAMROLLER_GPU_COUNT": env["GPU_COUNT"],
    }

path = env["WORK_DIR"]

lines_data = env.GetLines(
    Dir(path + env['PROCESSED_DIR']),
    [Dir(env['SOURCE_DIR']), Dir(env['SOURCE_DIR'])],
    **cpu_task_config("get_lines", "1:00:00"),
)
for max_data in [-1]: #[500, 1000, 1500, -1]:
    path = f"{env['WORK_DIR']}/max_data_{max_data}"
    model = env.TrainModel(
        [f"{path}/model.pkl", Dir(path + '/' + env['CHECKPOINT_DIR'])],
        [Dir(env['SOURCE_DIR']), lines_data],
        DATA_CUTOFF=max_data,
        WANDB_NAME=f"max_data_{max_data}",
        **gpu_task_config(f"train_model_{max_data}", "12:00:00"),
    )

    # Note - steamroller does not yet support creating dir nodes automatically
    evaluation = env.ApplyModel(
        [Dir(path + '/'+ env['MODEL_RESULTS_DIR']+ "/val"), Dir(path + '/'+env['MODEL_RESULTS_DIR']+ "/train")],
        [Dir(env['SOURCE_DIR']), lines_data, model],
        MODEL_RESULTS_DIR=Dir(path + '/'+ env['MODEL_RESULTS_DIR']),
        **gpu_task_config(f"apply_model_{max_data}", "2:00:00"),
    )

    val_results, train_results = evaluation

    val_report = env.GenerateReport(
        "${PATH}/val_report.json",
        Dir(val_results),
        PATH = path,
        **cpu_task_config("generate_report_val", "1:00:00"),
    )

    train_report = env.GenerateReport(
        "${PATH}/train_report.json",
        Dir(train_results),
        PATH = path,
        **cpu_task_config("generate_report_train", "1:00:00"),
    )

    errors_val = env.ExtractErrors(
        Dir(path + '/'+ env['MODEL_ERRORS_DIR'] +"/val"),
        Dir(val_results),
        PATH=path,
        **cpu_task_config("extract_errors_val", "1:00:00"),

    )

    errors_train = env.ExtractErrors(
        Dir(path + '/'+ env['MODEL_ERRORS_DIR'] +"/train"),
        Dir(train_results),
        PATH=path,
        **cpu_task_config("extract_errors_train", "1:00:00"),
    )



