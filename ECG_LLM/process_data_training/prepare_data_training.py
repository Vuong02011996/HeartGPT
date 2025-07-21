import os
import random
import re
from glob import glob
import shutil
import wfdb as wf
from tqdm import tqdm
from wfdb.processing import resample_singlechan, resample_sig
import numpy as np
from reprocessing import butter_bandpass_filter, beat_annotations, norm
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
import sys
import copy
import json
import csv
from multiprocessing import Process, JoinableQueue, Lock
from multiprocessing import Lock, Process, Queue, current_process
import time
import itertools
import queue  # imported for using queue.Empty exception
from config_dataset import (SAMPLING_RATE,
                            FEATURE_LEN,
                            OFFSET_LABEL,
                            GROUP_LABEL_LEN,
                            USED_NORM,
                            MIN_RR_LEN,
                            HEARTBEAT_TYPE,
                            EVENT_TYPE,
                            ROOT_DISK,
                            DISK_NAME, scale_number, )

import traceback
from sklearn.preprocessing import minmax_scale  # for rescaling

from ECG_LLM.process_data_training.helper import plot_numpy_data, ecg_to_tokens


def initiate_process_parameters(res_db_dict, ranges):
    return_dict = dict()
    return_dict['process_res_db_dict'] = [0] * len(ranges)
    return_dict['process_lst_file_to_handle'] = [0] * len(ranges)

    for i in range(len(ranges)):
        return_dict['process_res_db_dict'][i] = copy.deepcopy(res_db_dict)
        return_dict['process_res_db_dict'][i]["eval"]["total_sample"] = 0
        return_dict['process_res_db_dict'][i]["eval"]["study_ids"] = []
        return_dict['process_res_db_dict'][i]["train"]["total_sample"] = 0
        return_dict['process_res_db_dict'][i]["train"]["study_ids"] = []
        for key in res_db_dict["beat_class"].keys():
            return_dict['process_res_db_dict'][i]["eval"][key] = 0
            return_dict['process_res_db_dict'][i]["train"][key] = 0

        for key in res_db_dict["event_class"].keys():
            return_dict['process_res_db_dict'][i]["eval"][key] = 0
            return_dict['process_res_db_dict'][i]["train"][key] = 0

        return_dict['process_lst_file_to_handle'][i] = list()

    return return_dict


def process_study_batch(ds_type,
                        queue,
                        lock,
                        process_index,
                        ranges,
                        study_names,
                        output_directory):
    """

    :param queue:
    :param lock:
    :param process_index:
    :param ranges:
    :param study_names:
    :param output_directory:
    :return:
    """
    global process_res_db_dict
    global event_type_dict
    # Initial parameter for each process
    res_db_dict = process_res_db_dict[process_index]
    event_counter = 0
    study_counter = ranges[process_index][1] - ranges[process_index][0]
    study_in_shard = np.arange(ranges[process_index][0], ranges[process_index][1], dtype=int)
    for i in tqdm(study_in_shard):
        study_path = study_names[i]
        # a = os.path.basename(os.path.dirname(study_path))
        # if a != '380341':
        #     continue
        try:
            all_sub = []
            all_label = []
            # all_event_label = []
            # all_process_label = []
            all_events = glob(study_path + '*/*.hea')
            for event_path in all_events:
                file_name = event_path[:-4]
                header = wf.rdheader(file_name)
                fs_origin = header.fs
                channel = min(1, len(header.sig_name) - 1)
                startSample = 0
                stopSample = header.sig_len
                com = header.comments
                event_type = EVENT_TYPE["OTHER"]
                for m in com:
                    if "channel:" in m:
                        channel = int(m[-2:])
                    elif "startSample:" in m:
                        startSample = (int(re.findall(r'\d+', m)[0]) * SAMPLING_RATE) // fs_origin
                    elif "stopSample:" in m:
                        stopSample = (int(re.findall(r'\d+', m)[0]) * SAMPLING_RATE) // fs_origin
                    elif "eventType" in m:
                        # lock.acquire()
                        # if str(m).replace("eventType: ", "") not in event_type_dict.keys():
                        #     event_type_dict[str(m).replace("eventType: ", "")] = 1
                        # else:
                        #     event_type_dict[str(m).replace("eventType: ", "")] += 1
                        #
                        # lock.release()
                        if "SVT" in m:
                            event_type = EVENT_TYPE["SVT"]
                        elif "VT" in m:
                            event_type = EVENT_TYPE["VT"]
                        elif "SVE_" in m or "_SVE" in m:
                            event_type = EVENT_TYPE["SVE"]
                        elif "VE_" in m or "_VE" in m:
                            event_type = EVENT_TYPE["VE"]
                        elif "AFIB" in m:
                            event_type = EVENT_TYPE["AF"]
                        elif "TACHY" in m or "BRADY" in m or "PAUSE" in m or "AVB" in m:
                            event_type = EVENT_TYPE["SINUS"]


                ann = wf.rdann(file_name, extension='atr')
                record = wf.rdsamp(file_name, channels=[channel])
                buf_record = np.nan_to_num(record[0][:, 0])

                if fs_origin != SAMPLING_RATE:
                    buf_ecg, ann = resample_singlechan(buf_record,
                                                       ann,
                                                       ann.fs,
                                                       SAMPLING_RATE)

                else:
                    buf_ecg = buf_record.copy()

                beats, types = beat_annotations(ann)
                buf_ecg = butter_bandpass_filter(buf_ecg, 0.5, 40, SAMPLING_RATE)
                if USED_NORM:
                    buf_ecg = norm(buf_ecg, int(0.5 * SAMPLING_RATE))

                buf_ecg = ecg_to_tokens(buf_ecg)

                buf_ecg = buf_ecg[:(len(buf_ecg) // FEATURE_LEN) * FEATURE_LEN]
                buf_ecg_size = len(buf_ecg)
                index_label = np.arange(buf_ecg_size)
                buf_label = np.zeros(buf_ecg_size, np.int64)
                chk = list(np.where(np.diff(beats) < MIN_RR_LEN))
                if len(chk[0]) > 0:
                    beats = np.delete(beats, chk[0])
                    types = np.delete(types, chk[0])

                for beat, btype in zip(beats, types):
                    # buf_label += (np.astype(((index_label - OFFSET_LABEL < beat) ==
                    #                          (beat <= index_label + OFFSET_LABEL)), np.int64) *
                    #               (HEARTBEAT_TYPE[btype]))

                    buf_label += (
                            ((index_label - OFFSET_LABEL < beat) & (beat <= index_label + OFFSET_LABEL)).astype(np.int64) * HEARTBEAT_TYPE[btype]
                    )


                if ((stopSample - startSample) < (10 * SAMPLING_RATE) or
                        stopSample > buf_ecg_size or
                        (stopSample - startSample) > (20 * SAMPLING_RATE)):
                    frame_ecg = []
                    frame_label = []
                    # process_frame_label = []
                    # process_event_label = []
                else:
                    if (stopSample - startSample) % FEATURE_LEN != 0 and (
                            stopSample - startSample) // FEATURE_LEN == 1:
                        stopSample = startSample + FEATURE_LEN
                    elif (stopSample - startSample) % FEATURE_LEN != 0 and (
                            stopSample - startSample) // FEATURE_LEN == 2:
                        stopSample = startSample + FEATURE_LEN * 2

                    frame_ecg = buf_ecg[startSample: stopSample]
                    frame_label = buf_label[startSample: stopSample]
                    # frame_len = len(frame_ecg)
                    # label_index = np.arange(GROUP_LABEL_LEN)[None, :] + \
                    #               np.arange(0, frame_len, GROUP_LABEL_LEN)[:, None]
                    # frame_label_index = frame_label[label_index]
                    # process_frame_label = np.asarray([np.max(lbl, axis=-1) for lbl in frame_label_index],
                    #                                  dtype=int).flatten()
                    #
                    # full_label_index = np.arange(FEATURE_LEN)[None, :] + \
                    #               np.arange(0, frame_len, FEATURE_LEN)[:, None]
                    # process_event_label = np.asarray([event_type for _ in frame_label[full_label_index]],
                    #                                  dtype=int).flatten()

                # if len(process_frame_label) > 0:
                #     for key in res_db_dict["beat_class"].keys():
                #         lbl_pos = np.where(process_frame_label == res_db_dict["beat_class"][key])[0]
                #         if len(lbl_pos) > 0:
                #             res_db_dict[ds_type][key] += len(lbl_pos)
                #
                #     for key in res_db_dict["event_class"]:
                #         lbl_pos = np.where(process_event_label == res_db_dict["event_class"][key])[0]
                #         if len(lbl_pos) > 0:
                #             res_db_dict[ds_type][key] += len(lbl_pos)

                if len(frame_ecg) > 0:
                    # frame_ecg = np.round((scale_number) * minmax_scale(frame_ecg), 0)
                    frame_ecg = np.expand_dims(frame_ecg, axis=-1)
                    all_sub.append(frame_ecg)
                    all_label.append(frame_label)
                    # plot_numpy_data(frame_ecg, frame_label)

                    # all_process_label.append(process_frame_label)
                    # all_event_label.append(process_event_label)

                event_counter += 1

            if len(all_sub) > 0:
                all_sub = np.concatenate(all_sub, axis=0)
                all_label = np.concatenate(all_label, axis=0)
                # all_process_label = np.concatenate(all_process_label, axis=0)
                # all_event_label = np.concatenate(all_event_label, axis=0)

            study_id = os.path.basename(os.path.dirname(study_path))

            if len(all_sub) > 0:
                res_db_dict[ds_type]["study_ids"].append(int(study_id))
                res_db_dict[ds_type]["total_sample"] += len(all_sub) // FEATURE_LEN

                output_file = os.path.join(OUT_DATA_PATH, ds_type, "sub_{}.npy".format(study_id))
                np.save(output_file, all_sub)
                output_file = os.path.join(OUT_DATA_PATH, ds_type, "lab_{}.npy".format(study_id))
                np.save(output_file, all_label)
                # output_file = os.path.join(OUT_DATA_PATH, ds_type, "process_lab_{}.npy".format(study_id))
                # np.save(output_file, all_process_label)
                # output_file = os.path.join(OUT_DATA_PATH, ds_type, "event_lab_{}.npy".format(study_id))
                # np.save(output_file, all_event_label)

            process_res_db_dict[process_index] = res_db_dict

        except Exception as e:
            traceback.print_exc()
            lock.acquire()
            try:
                print(e)
                print('SKIPPED: Unexpected error while decoding %s.' % file_name)
            finally:
                lock.release()
            continue

    lock.acquire()
    try:
        print('%s [processor %d]: Process %d studys with %d events' %
              (datetime.now(), process_index, study_counter, event_counter))

        sys.stdout.flush()
    finally:
        lock.release()


    queue.put({'process_res_db_dict': process_res_db_dict[process_index]})
    queue.task_done()



def build_numpy(ds_type,
                db_process_info,
                datastore_dict,
                output_directory,
                # num_processes=os.cpu_count()):
                num_processes=1):
    """

    :param db_process_info:
    :param datastore_dict:
    :param output_directory:
    :return:
    """
    global process_res_db_dict
    print(output_directory)
    spacing = np.linspace(0, len(db_process_info["study_ids"]), num_processes + 1).astype(np.int64)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a processor for each batch.
    # print('Launching %d processors for spacings: %s' % (num_processes, ranges))
    sys.stdout.flush()

    # Initiate parameter for each process
    process_data_dict = initiate_process_parameters(datastore_dict, ranges)
    process_res_db_dict = process_data_dict['process_res_db_dict']
    processors = list()
    process_queue = [list() for _ in range(num_processes)]
    process_lock = [list() for _ in range(num_processes)]

    for process_index in range(len(ranges)):
        process_queue[process_index] = JoinableQueue()
        process_lock[process_index] = Lock()
        args = (ds_type,
                process_queue[process_index],
                process_lock[process_index],
                process_index,
                ranges,
                db_process_info["study_ids"],
                output_directory)
        t = Process(target=process_study_batch, args=args)
        t.start()
        processors.append(t)

    # Get output of processes
    for process_index in range(len(ranges)):
        process_returned_data = process_queue[process_index].get()
        process_res_db_dict[process_index] = process_returned_data['process_res_db_dict']
        processors[process_index].terminate()

    # Concatenate processes returned output !!!
    datastore_dict[ds_type]["total_sample"] += int(
        np.asarray([t[ds_type]["total_sample"] for t in process_res_db_dict]).sum())


    for key in datastore_dict["beat_class"].keys():
        datastore_dict[ds_type][key] += int(np.asarray([t[ds_type][key] for t in process_res_db_dict]).sum())

    for key in datastore_dict["event_class"].keys():
        datastore_dict[ds_type][key] += int(np.asarray([t[ds_type][key] for t in process_res_db_dict]).sum())

    datastore_dict[ds_type]["study_ids"] += list(
        itertools.chain(*[t[ds_type]["study_ids"] for t in process_res_db_dict]))
    sys.stdout.flush()
    return datastore_dict




def buil_sub():
    datastore_dict = dict()
    datastore_dict["train"] = dict()
    datastore_dict["eval"] = dict()
    datastore_dict["GROUP_LABEL_LEN"] = GROUP_LABEL_LEN
    datastore_dict["USED_NORM"] = USED_NORM
    datastore_dict["OFFSET_LABEL"] = OFFSET_LABEL
    datastore_dict["SAMPLING_RATE"] = SAMPLING_RATE
    datastore_dict["FEATURE_LEN"] = FEATURE_LEN
    datastore_dict["beat_class"] = HEARTBEAT_TYPE
    datastore_dict["event_class"] = EVENT_TYPE
    datastore_dict["train"]["total_sample"] = 0
    datastore_dict["eval"]["total_sample"] = 0
    for key in datastore_dict["beat_class"].keys():
        datastore_dict["eval"][key] = 0
        datastore_dict["train"][key] = 0

    for key in datastore_dict["event_class"].keys():
        datastore_dict["eval"][key] = 0
        datastore_dict["train"][key] = 0

    datastore_dict["train"]["study_ids"] = []
    datastore_dict["eval"]["study_ids"] = []

    all_study_ids = dict()
    all_study_ids["train"] = []
    all_study_ids["eval"] = []

    all_study = glob(DATA_PATH_LABEL + '/*/')
    x_train = int(0.8 * len(all_study))
    random.shuffle(all_study)
    train_study = all_study[:x_train]
    eval_study = all_study[x_train:]

    with open(os.path.join(OUT_DATA_PATH, "train_sub_study_ids.txt"), 'w') as file:
        for study_path in train_study:
            file.write("{}\n".format(study_path))
            all_study_ids["train"].append((str(study_path)))

    with open(os.path.join(OUT_DATA_PATH, "eval_sub_study_ids.txt"), 'w') as file:
        for study_path in eval_study:
            file.write("{}\n".format(study_path))
            all_study_ids["eval"].append((str(study_path)))


    for t in ["train", "eval"]:
        out_dir = '{}/{}/'.format(OUT_DATA_PATH, t)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # else:
        #     shutil.rmtree(out_dir)
        #     os.makedirs(out_dir)

        db_process_info = dict()
        db_process_info['study_ids'] = all_study_ids[t].copy()
        datastore_dict = build_numpy(ds_type=t,
                                     db_process_info=db_process_info,
                                     datastore_dict=datastore_dict,
                                     output_directory=out_dir)

        print('num_{}_samples = {}'.format(t, datastore_dict[t]['total_sample']))


    log_datastore = open(OUT_DATA_PATH + '/datastore_sub.txt', 'w')
    json.dump(datastore_dict, log_datastore)
    log_datastore.close()

    return True


def check_data_numpy():
    eval_file = "/media/server2/MegaDataset1/DataTraining/NumpyData/eval/lab_373712.npy"
    train_file = "/media/server2/MegaDataset1/DataTraining/NumpyData/eval/sub_373712.npy"
    train_file_ = "/media/server2/MegaDataset1/DataTraining/NumpyData/eval/process_lab_373712.npy"
    data1 = np.load(eval_file)
    data1 = np.reshape(data1, (-1, 1280, data1.shape[-1]))
    data2 = np.load(train_file)
    data2 = np.reshape(data2, (-1, 1280, data2.shape[-1]))
    data3 = np.load(train_file_)
    data3 = np.reshape(data3, (-1, 1280, data3.shape[-1]))
    a = 0


def load_data_bcty_numpy():
    SAMPLING_RATE = 128
    subseq_size = int(10 * SAMPLING_RATE)
    data_type = "labseq"
    data_path = OUT_DATA_PATH

    if data_type == "labseq":
        all_train_data = glob(data_path + '/train/sub*.npy')
        all_eval_data = glob(data_path + '/eval/sub*.npy')
        for train_file in tqdm(all_train_data, leave=True, desc="Load Training Dataset Progress"):
            print(f"Processing in file: {train_file}")
            data = np.load(train_file)
            label = np.load(train_file.replace("sub_", "lab_"))
            label = np.reshape(label, (-1, subseq_size))

            strips_one_study = np.reshape(data, (-1, subseq_size))
            for i, strip in enumerate(strips_one_study):
                plot_numpy_data(strip, label[i])

        # for eval_file in tqdm(all_eval_data, leave=True, desc="Load Testing Dataset Progress"):
        #     data = np.load(eval_file)
        #     data = np.reshape(data, (-1, subseq_size, data.shape[-1]))
        #     label = np.load(eval_file.replace("sub_", "lab_"))
        #     label = np.reshape(label, (-1, subseq_size))


if __name__ == '__main__':
    DATA_PATH_LABEL = "/media/server2/MegaDataset/DataTraining/LabelCorrect/"
    OUT_DATA_PATH = "/media/server2/MegaDataset/DataTraining/NumpyData_tokenization_512/"
    os.makedirs(OUT_DATA_PATH, exist_ok=True)
    buil_sub()

    # check_data_numpy()
    # load_data_bcty_numpy()



    """
    # How to cut data: Flow
    + 
    # How to training:
    + 
    # Have new model
    + How to inference with data training.
    + Plot result new model with label.
        + Get event id that error label with result model. 
        + Check 
    
    """