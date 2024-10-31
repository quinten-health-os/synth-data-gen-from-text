import os
import sys
import warnings
from datetime import datetime
import logging

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

import config as conf 
from src.logger import init_logger
from src import loading
from src.parsers.pipeline_parser import pipeline_parser
from src.utils.utils_sdv import get_metadata_from_dict

from src.modelling.sdv_copula import fit_copula
from src.modelling.sdv_ctgan import fit_ctgan
from src.modelling.sdv_tvae import fit_tvae, get_loss_tvae


def main():

    # Initiate parser
    parser = pipeline_parser()
    args = parser.parse_args()

    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")

    # load data
    if args.real_dataset:
        # case when training data is specified in args
        path_file = args.real_dataset
    else:
        path_file = os.path.join(conf.PATH_PREPARED_DATA, conf.FILE_PREPARED_DATA)
    df_prepared = loading.read_data(bucket_name=conf.BUCKET_NAME,
                                    path_file=path_file)    
    # load metadata
    dict_metadata = loading.read_dict(
        conf.BUCKET_NAME, os.path.join(conf.PATH_METADATA, conf.FILE_METADATA)
    )
    sdv_metadata = get_metadata_from_dict(dict_metadata=dict_metadata)
    
    # modelling 
    start_time = datetime.now()
    fig = None # plot loss fig
    if conf.SDG_MODEL == "ctgan":
        model = fit_ctgan(df=df_prepared,
                        sdv_metadata=sdv_metadata,
                        epochs=conf.EPOCHS,
                        batch_size=conf.BATCH_SIZE)
        
    elif conf.SDG_MODEL == "copula":
        model = fit_copula(df=df_prepared,
                            sdv_metadata=sdv_metadata)
        
    elif conf.SDG_MODEL == "tvae":
        model = fit_tvae(df=df_prepared,
                        sdv_metadata=sdv_metadata,
                        epochs=conf.EPOCHS,
                        batch_size=conf.BATCH_SIZE)
    
        # loss function 
        fig = get_loss_tvae(synthesizer=model)
    else:
        raise ValueError("Model not implemented")
    
    # sample synthetic data
    df_synth = model.sample(num_rows=conf.N_SAMPLE)
    time = datetime.now() - start_time
    logging.info(f"Execution time: {time}")

    # saving
    if args.save:
        if args.synth_dataset:
            path_file = args.synth_dataset
        else:
            path_file = os.path.join(conf.PATH_SYNTH_DATA, conf.FILE_SYNTHESIZED_DATA)
        loading.save_csv(df_synth,
                         conf.BUCKET_NAME,
                         path_file=path_file)
        
        if fig:
            loading.save_figure_s3(
            fig,
            conf.BUCKET_NAME,
            conf.PATH_SYNTH_DATA,
            f"training/loss_{conf.SDG_MODEL}_{conf.DATABASE}_{conf.DATE}.png")
    
if __name__ == "__main__":
    main()
    