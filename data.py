import os

def create_h5(h5file, overwrite=False):
    if overwrite or not os.path.exists(h5file):
    training_files, subject_ids = fetch_training_data_files(return_subject_ids=True)

    write_data_to_file(training_files, 
                        config["data_file"], 
                        image_shape=config["image_shape"], 
                        modality_names = config['all_modalities'],
                        subject_ids=subject_ids,
                       mean_std_file = config['mean_std_file'])
    data_file_opened = open_data_file(config["data_file"])