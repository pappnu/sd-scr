import os
import pathlib
import wave


def list_audio_files_in_subfolders(root_folder, exclude=[], file_suffixes=['.wav']):
    audio_files = []
    subfolder_names = []
    with os.scandir(root_folder) as root_iter:
        for i in root_iter:
            if not any(exc in i.name for exc in exclude) and i.is_dir():
                with os.scandir(i.path) as sub_iter:
                    n_files_before = len(audio_files)

                    for j in sub_iter:
                        j_path = pathlib.Path(j.path)
                        if any(j_path.suffix in suf for suf in file_suffixes):
                            audio_files.append(str(j_path))

                    if len(audio_files) > n_files_before:
                        subfolder_names.append(i.name)

    return [audio_files, subfolder_names]


def list_files_in_directory(dir_path, file_suffixes=['.wav']):
    files = []
    with os.scandir(dir_path) as iter:
        for j in iter:
            j_path = pathlib.Path(j.path)
            if any(j_path.suffix in suf for suf in file_suffixes):
                files.append(str(j_path))
    
    return files


def form_train_val_and_test_lists(file_list, config):
    output = {
        'validation': [],
        'test': [],
        'train': [],
    }

    file_list_copy = file_list.copy()

    for label in config:
        for out_list in output:
            for i in range(config[label][out_list]):
                for index, file in enumerate(file_list_copy):
                    if label + os.path.sep in file:
                        output[out_list].append(file)
                        file_list_copy.pop(index)
                        break

    return output


def file_lines_to_list(file_path):
    with open(file_path, 'r') as file_open:
        return file_open.read().splitlines()


def matching_files(main_list, search_list):
    matches = []
    others = []

    temp_list = [str(pathlib.Path(j)) for j in search_list]
    for i in main_list:
        match = ''
        for j in temp_list:
            if j in i:
                match = i
                matches.append(i)
                temp_list.remove(j)
                break

        if not match:
            others.append(i)

    return [matches, others]


def find_longest_wav(file_paths):
    longest_duration = 0
    for file_path in file_paths:
        with wave.open(file_path, 'rb') as wav:
            frames = wav.getnframes()
            framerate = wav.getframerate()
            duration = frames / float(framerate)
            if duration > longest_duration:
                longest_duration = duration

    print(longest_duration)


def speaker_dependent_sets(file_paths):
    file_dict = {}
    for file_path in file_paths:
        path_obj = pathlib.PurePath(file_path)
        speaker = path_obj.name.split('_')[0]
        command = path_obj.parent.name

        if speaker not in file_dict:
            file_dict[speaker] = {}
        if command not in file_dict[speaker]:
            file_dict[speaker][command] = []

        file_dict[speaker][command].append(file_path)

    return file_dict


def utterances_per_speaker(speaker_sets):
    n_utterances_by_speaker = {}
    for speaker in speaker_sets:
        utterances_total = 0
        n_utterances_by_speaker[speaker] = {}
        for command in speaker_sets[speaker]:
            utterances_total = utterances_total + len(speaker_sets[speaker][command])
            n_utterances_by_speaker[speaker][command] = len(
                speaker_sets[speaker][command]
            )
        n_utterances_by_speaker[speaker]['total'] = utterances_total

    return {
        dict_key: dict_value
        for dict_key, dict_value in sorted(
            n_utterances_by_speaker.items(),
            key=lambda item: item[1]['total'],
            reverse=True
        )
    }
