import os
import os.path
from steamroller import Environment


vars = Variables("custom.py")
vars.AddVariables(
    ("SOURCE_DIR", "", "source"),
    ("WORK_DIR", "", "work"),
    ("UPDATED_LINE_POLYGONS_DIR", "", "/updated_polygons"),
    ("PROCESSED_DIR", "", "/processed_data"),
    ("CHECKPOINT_DIR", "", "/checkpoints"),
    ("MODEL_RESULTS_DIR", "", "/results"),
    ("MAX_EPOCHS", "", 100),
    ("RANDOM_SEED", "", 40),
    ("TRAIN_PROPORTION", "", 0.9),
    ("GPU_DEVICES", "", [1]), # GPU device number
    ("MODEL_ERRORS_DIR", "", "/errors"),
    ("DATA_CUTOFF", "", -1)
)

env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[],
    
    BUILDERS={
        "UpdateLinePolygons" : Builder(
            action="python scripts/update_line_polygons.py "
            "--input_dir ${SOURCE} "
            "--output ${TARGET} "
        ),
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
            "--data_cutoff ${DATA_CUTOFF}"
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
path = env["WORK_DIR"]
updated_xmls = env.UpdateLinePolygons(
    Dir(path + env['UPDATED_LINE_POLYGONS_DIR']),
    Dir(env['SOURCE_DIR']),
)
lines_data = env.GetLines(
    Dir(path + env['PROCESSED_DIR']),
    [Dir(env['SOURCE_DIR']), updated_xmls],
)
for max_data in [500, 1000, 1500, -1]:
    path = f"{env['WORK_DIR']}/max_data_{max_data}"
    model = env.TrainModel(
        [f"{path}/model.pkl", Dir(path + '/' + env['CHECKPOINT_DIR'])],
        [Dir(env['SOURCE_DIR']), lines_data],
        DATA_CUTOFF=max_data
    )

    # Note - steamroller does not yet support creating dir nodes automatically
    evaluation = env.ApplyModel(
        [Dir(path + '/'+ env['MODEL_RESULTS_DIR']+ "/val"), Dir(path + '/'+env['MODEL_RESULTS_DIR']+ "/train")],
        [Dir(env['SOURCE_DIR']), lines_data, model],
        MODEL_RESULTS_DIR=Dir(path + '/'+ env['MODEL_RESULTS_DIR']), 
    )

    val_results, train_results = evaluation

    val_report = env.GenerateReport(
        "${PATH}/val_report.json",
        val_results,
        PATH = path
    )

    train_report = env.GenerateReport(
        "${PATH}/train_report.json",
        train_results,
        PATH = path
    )

    errors_val = env.ExtractErrors(
        "${PATH}/{MODEL_ERRORS_DIR}/val",
        val_results,
        PATH=path
    )

    errors_train = env.ExtractErrors(
        "${PATH}/{MODEL_ERRORS_DIR}/train",
        train_results,
        PATH=path
    )



